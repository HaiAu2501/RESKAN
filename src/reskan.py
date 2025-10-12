import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# ----------------------------
#  Utilities: Charbonnier + TV
# ----------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def tv_l1(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()

# --------------------------------------
#  Fixed Haar Wavelet DWT / iDWT blocks
# --------------------------------------
class HaarDWT2D(nn.Module):
    """
    Depthwise DWT với Haar (stride=2). In: (B, F, H, W) -> Out: (B, 4F, H/2, W/2)
    """
    def __init__(self, channels: int):
        super().__init__()
        Fch = channels
        h = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        g = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])
        ll = torch.einsum('i,j->ij', h, h)
        lh = torch.einsum('i,j->ij', h, g)
        hl = torch.einsum('i,j->ij', g, h)
        hh = torch.einsum('i,j->ij', g, g)
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).repeat(Fch, 1, 1, 1)  # (4F,1,2,2)
        self.register_buffer("weight", weight)
        self.conv = nn.Conv2d(Fch, 4 * Fch, kernel_size=2, stride=2, padding=0, groups=Fch, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(self.weight)
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        return self.conv(x)

class HaarIDWT2D(nn.Module):
    """
    Depthwise iDWT với Haar (stride=2). In: (B, 4F, H, W) -> Out: (B, F, 2H, 2W)
    """
    def __init__(self, channels: int):
        super().__init__()
        Fch = channels
        h = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        g = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])
        ll = torch.einsum('i,j->ij', h, h)
        lh = torch.einsum('i,j->ij', h, g)
        hl = torch.einsum('i,j->ij', g, h)
        hh = torch.einsum('i,j->ij', g, g)
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).repeat(Fch, 1, 1, 1)  # (4F,1,2,2)
        self.register_buffer("weight", weight)
        self.deconv = nn.ConvTranspose2d(4 * Fch, Fch, kernel_size=2, stride=2, padding=0, groups=Fch, bias=False)
        with torch.no_grad():
            self.deconv.weight.copy_(self.weight)
        for p in self.deconv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.deconv(x)

# ---------------------------------------------------------
#  KAN cubic dictionary (low-rank, shared across channels)
# ---------------------------------------------------------
class KANCubicDict1D(nn.Module):
    """
    Low-rank KAN: sử dụng R spline bậc-3 (dictionary dùng chung) và trộn theo kênh.
    - Dictionary tham số hoá bằng nn.Embedding (K -> R) để gather cực kỳ gọn.
    - Mỗi kênh c có vector trộn w_{c,:} \in R^R, cùng id_gain_c và bias_c.
    - Vẫn có affine map per-channel để phủ [-1,1].
    """
    def __init__(self, channels: int, K: int = 32, R: int = 8, clamp_val: float = 1.5):
        super().__init__()
        assert K >= 4, "Cubic cần K >= 4"
        self.C = channels
        self.K = K
        self.R = R
        self.clamp_val = clamp_val

        # Map per-channel: a*x + b để đưa về [-1,1]
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))

        # Dictionary alpha: (K, R); Embedding giúp gather theo chỉ số i0..i3 nhanh/gọn.
        self.alpha_table = nn.Embedding(K, R)
        nn.init.zeros_(self.alpha_table.weight)  # khởi tạo gần tuyến tính qua id_gain

        # Mixing per channel: (C, R)
        self.mix = nn.Parameter(torch.zeros(channels, R))

        # Skip linear + bias per channel
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        # Biên miền [-1,1]
        self.register_buffer("low", torch.tensor(-1.0), persistent=False)
        self.register_buffer("high", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _cubic_basis(t: torch.Tensor):
        # t in [0,1], trả về (b0,b1,b2,b3) broadcast với shape t
        t2 = t * t
        t3 = t2 * t
        b0 = (1 - 3*t + 3*t2 - t3) / 6.0
        b1 = (4 - 6*t2 + 3*t3) / 6.0
        b2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
        b3 = t3 / 6.0
        return b0, b1, b2, b3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  --> y: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C

        # affine + clamp
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        # map [-1,1] -> [0, K-1]
        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)                # (B,C,H,W)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)

        # 4 indices with boundary clamp
        i0 = (i - 1).clamp(0, self.K - 1)
        i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1)
        i3 = (i + 2).clamp(0, self.K - 1)

        # cubic basis
        b0, b1, b2, b3 = self._cubic_basis(t)  # (B,C,H,W)

        # gather alpha for each index via embedding: returns (..., R)
        A0 = self.alpha_table(i0)  # (B,C,H,W,R)
        A1 = self.alpha_table(i1)
        A2 = self.alpha_table(i2)
        A3 = self.alpha_table(i3)

        # spline value per basis r
        # shape (B,C,H,W,R)
        spline_per_r = (
            A0 * b0.unsqueeze(-1) +
            A1 * b1.unsqueeze(-1) +
            A2 * b2.unsqueeze(-1) +
            A3 * b3.unsqueeze(-1)
        )

        # mix across R with per-channel weights: (C,R)
        # output (B,C,H,W)
        y_nl = torch.einsum('bchwr,cr->bchw', spline_per_r, self.mix)

        y = self.id_gain.view(1, C, 1, 1) * x + y_nl + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self) -> torch.Tensor:
        """
        Sobolev penalty trên các spline toàn cục (theo k).
        alpha_table.weight: (K, R)
        """
        w = self.alpha_table.weight  # (K,R)
        d1 = w[1:, :] - w[:-1, :]
        d2 = d1[1:, :] - d1[:-1, :]
        return (d2 ** 2).mean()

    def group_lasso_penalty(self) -> torch.Tensor:
        """
        L_{2,1} trên mix (khuyến khích kênh dùng ít basis, giảm rank hiệu dụng).
        """
        # mix: (C,R) -> sum_c ||mix_c||_2
        eps = 1e-8
        norms = torch.sqrt((self.mix ** 2).sum(dim=1) + eps)  # (C,)
        return norms.mean()

# -------------------------
#  Depthwise-Separable Conv
# -------------------------
class DSConv(nn.Module):
    """
    Depthwise 3x3 + Pointwise 1x1; chuẩn hoá nhẹ bằng GroupNorm để ổn định Lipschitz.
    """
    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(ch, ch, kernel_size, padding=padding, groups=ch, bias=True)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

    def forward(self, x):
        y = self.dw(x)
        y = F.relu(self.gn(self.pw(y)), inplace=False)
        return y

# -------------------------
#  ResKAN Lite Residual Block
# -------------------------
class ResKANLiteBlock(nn.Module):
    """
    - DWT/iDWT cố định (Haar).
    - 2 DSConv nhẹ + KAN dictionary.
    - SE gating + residual scaling s \in (0,1) để đảm bảo tính co.
    """
    def __init__(self, feat_ch: int, kan_K: int = 32, kan_R: int = 8):
        super().__init__()
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)

        in_ch = 4 * feat_ch

        self.conv1 = DSConv(in_ch, kernel_size=3)
        self.kan = KANCubicDict1D(in_ch, K=kan_K, R=kan_R)
        self.conv2 = DSConv(in_ch, kernel_size=3)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(in_ch // 8, 8), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

        # Residual scaling parameter s in (0,1)
        self.res_scale_param = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 khởi tạo vừa phải

    def forward(self, x):
        s = torch.sigmoid(self.res_scale_param)  # (scalar)
        z = self.dwt(x)            # (B, 4F, H/2, W/2)
        z = self.conv1(z)
        z = self.kan(z)
        z = self.conv2(z)
        z = z * self.se(z)
        y = self.idwt(z)
        return x + s * y

    def regularizers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sob, gl = self.kan.sobolev_penalty(), self.kan.group_lasso_penalty()
        return sob, gl

# -------------------------
#  Full Slim ResKAN Network
# -------------------------
class SlimResKANNet(nn.Module):
    """
    - Một khối ResKANLite lặp T lần (weight sharing) để giảm params.
    - Có thể bật 'share_block=False' nếu muốn khối khác nhau (tăng capacity).
    """
    def __init__(self, in_ch: int = 3, feat_ch: int = 64, num_steps: int = 8,
                 kan_K: int = 32, kan_R: int = 8, share_block: bool = True):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)
        self.num_steps = num_steps
        self.share_block = share_block

        if share_block:
            self.block = ResKANLiteBlock(feat_ch, kan_K=kan_K, kan_R=kan_R)
        else:
            self.blocks = nn.ModuleList([
                ResKANLiteBlock(feat_ch, kan_K=kan_K, kan_R=kan_R)
                for _ in range(num_steps)
            ])

        self.recon = nn.Conv2d(feat_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        inp = x
        f = self.shallow(x)
        if self.share_block:
            for _ in range(self.num_steps):
                f = self.block(f)
        else:
            for blk in self.blocks:
                f = blk(f)
        out = self.recon(f)
        return torch.clamp(inp + out, 0.0, 1.0)

    def regularizers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sob = 0.0
        gl = 0.0
        if self.share_block:
            s, g = self.block.regularizers()
            sob = sob + s
            gl = gl + g
        else:
            for blk in self.blocks:
                s, g = blk.regularizers()
                sob = sob + s
                gl = gl + g
        return sob, gl

# ---------------------------------------
#  LightningModule: training + validation
# ---------------------------------------
class SlimResKANLightning(pl.LightningModule):
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        num_steps: int = 8,
        kan_K: int = 32,
        kan_R: int = 8,
        share_block: bool = True,
        lr: float = 2e-4,
        wd: float = 1e-8,
        lambda_charb: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_tv: float = 0.0,
        lambda_sobolev: float = 5e-4,
        lambda_group: float = 1e-4,   # L_{2,1} cho mix (giảm rank hiệu dụng)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = SlimResKANNet(
            in_ch=in_ch, feat_ch=feat_ch, num_steps=num_steps,
            kan_K=kan_K, kan_R=kan_R, share_block=share_block
        )

        self.loss_charb = CharbonnierLoss(eps=1e-3)
        self.loss_l1 = nn.L1Loss()

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.lr = lr
        self.wd = wd
        self.lambda_charb = lambda_charb
        self.lambda_l1 = lambda_l1
        self.lambda_tv = lambda_tv
        self.lambda_sobolev = lambda_sobolev
        self.lambda_group = lambda_group

    def forward(self, x):
        return self.net(x)

    def _regularization_terms(self):
        sob, gl = self.net.regularizers()
        return sob, gl

    def training_step(self, batch, batch_idx):
        lq, gt = batch  # (B,C,H,W) in [0,1]
        pred = self(lq)

        loss = 0.0
        if self.lambda_charb > 0:
            loss = loss + self.lambda_charb * self.loss_charb(pred, gt)
        if self.lambda_l1 > 0:
            loss = loss + self.lambda_l1 * self.loss_l1(pred, gt)
        if self.lambda_tv > 0:
            loss = loss + self.lambda_tv * tv_l1(pred)

        sob, gl = self._regularization_terms()
        if self.lambda_sobolev > 0:
            loss = loss + self.lambda_sobolev * sob
        if self.lambda_group > 0:
            loss = loss + self.lambda_group * gl

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=lq.size(0))
        self.log("reg_sobolev", sob, prog_bar=False, on_step=True, on_epoch=True, batch_size=lq.size(0))
        self.log("reg_group", gl, prog_bar=False, on_step=True, on_epoch=True, batch_size=lq.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1)
        gt = gt.clamp(0, 1)

        psnr = self.val_psnr(pred, gt)
        ssim = self.val_ssim(pred, gt)

        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("val_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)

        val_loss = self.loss_charb(pred, gt)
        self.log("val_loss", val_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0))
        return {"val_psnr": psnr, "val_ssim": ssim}

    def test_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1)
        gt = gt.clamp(0, 1)

        psnr = self.val_psnr(pred, gt)
        ssim = self.val_ssim(pred, gt)
        
        self.log("test_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("test_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        return {"test_psnr": psnr, "test_ssim": ssim}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=self.lr * 0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
