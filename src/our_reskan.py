# reskan.py 
import math
from typing import Tuple, Optional

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


# ---------------------------------------------------------
#  KAN: cubic B-spline 1D (degree=3), separable & memory-safe
# ---------------------------------------------------------
class KANCubic1D(nn.Module):
    """
    phi_c(x) = id_gain_c * x + bias_c + sum_{j in {i-1,i,i+1,i+2}} alpha[c, j] * B_j(t),
    i = floor(u), t = u - i, u = map([-1,1]->[0,K-1]).
    Khởi tạo "hiền": id_gain=0, alpha=0, bias=0 => KAN ~ 0 lúc đầu.
    """
    def __init__(self, channels: int, K: int = 32, clamp_val: Optional[float] = 6.0):
        super().__init__()
        assert K >= 4, "Cubic cần K >= 4"
        self.C = channels
        self.K = K
        self.clamp_val = clamp_val

        self.a = nn.Parameter(torch.ones(channels))       # giữ 1.0 để không bóp méo lớn
        self.b = nn.Parameter(torch.zeros(channels))
        self.alpha = nn.Parameter(torch.zeros(channels, K))
        self.id_gain = nn.Parameter(torch.zeros(channels))  # <<< KHỞI TẠO 0.0
        self.bias = nn.Parameter(torch.zeros(channels))

        self.register_buffer("low", torch.tensor(-1.0), persistent=False)
        self.register_buffer("high", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _cubic_basis(t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        t2 = t * t
        t3 = t2 * t
        b0 = (1 - 3*t + 3*t2 - t3) / 6.0
        b1 = (4 - 6*t2 + 3*t3) / 6.0
        b2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
        b3 = t3 / 6.0
        return b0, b1, b2, b3

    def _gather_alpha_linear(self, idx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = idx.shape
        K = self.K
        c = torch.arange(C, device=idx.device).view(1, C, 1, 1).expand(B, C, H, W)
        lin = c * K + idx
        return torch.take(self.alpha.view(-1), lin.view(-1)).view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.C
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)

        i0 = (i - 1).clamp(0, self.K - 1)
        i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1)
        i3 = (i + 2).clamp(0, self.K - 1)

        b0, b1, b2, b3 = self._cubic_basis(t)

        a0 = self._gather_alpha_linear(i0)
        a1 = self._gather_alpha_linear(i1)
        a2 = self._gather_alpha_linear(i2)
        a3 = self._gather_alpha_linear(i3)

        spline_val = a0*b0 + a1*b1 + a2*b2 + a3*b3
        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self) -> torch.Tensor:
        d1 = self.alpha[:, 1:] - self.alpha[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        return (d2 ** 2).mean()


# -------------------------
#  ResKAN Residual Block (sửa rất nhỏ)
# -------------------------
class ResKANBlock(nn.Module):
    def __init__(self, feat_ch: int, kernel_size: int = 3, kan_K: int = 16, kan_degree: int = 3):
        super().__init__()
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)

        in_ch = 4 * feat_ch
        padding = kernel_size // 2

        # GIỮ groups=1 như bản gốc (không đổi kiến trúc)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)
        self.kan = KANCubic1D(in_ch, K=kan_K)  # dùng đúng kan_K người dùng truyền vào
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)

        # SE giữ nguyên, nhưng bias cuối khởi tạo dương để gate≈1 lúc đầu
        self.se_conv1 = nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1)
        self.se_conv2 = nn.Conv2d(max(in_ch // 8, 8), in_ch, kernel_size=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            self.se_conv1,
            nn.ReLU(),  # không inplace
            self.se_conv2,
            nn.Sigmoid()
        )

        # Residual scaling để gần-identity nhưng có gradient
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # học được; init nhỏ

        # Init cẩn thận để nhánh residual nhỏ nhưng không tắc gradient
        self._init_weights()

    def _init_weights(self):
        # conv1, conv2: Kaiming, nhưng scale nhỏ cho conv2
        nn.init.kaiming_normal_(self.conv1.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        nn.init.kaiming_normal_(self.conv2.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        with torch.no_grad():
            self.conv2.weight.mul_(0.1)  # scale nhỏ để không phá identity ở init
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # SE gate ≈ 1 ở init (đừng bóp méo nhánh)
        nn.init.kaiming_normal_(self.se_conv1.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(self.se_conv1.bias)
        nn.init.zeros_(self.se_conv2.weight)
        nn.init.constant_(self.se_conv2.bias, 2.0)  # sigmoid(2)~0.88

        # KAN đã init "hiền": id_gain=0, alpha=0, bias=0 (ở lớp KAN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dwt(x)            # (B, 4F, H/2, W/2)
        z = self.conv1(z)
        z = self.kan(z)
        z = self.conv2(z)
        z = z * self.se(z)
        y = self.idwt(z)
        return x + self.res_scale * y


# -------------------------
#  Full ResKAN Network
# -------------------------
class ResKANNet(nn.Module):
    def __init__(self, in_ch: int = 3, feat_ch: int = 64, num_blocks: int = 8,
                 kan_K: int = 16, kan_degree: int = 3):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResKANBlock(feat_ch, kernel_size=3, kan_K=kan_K, kan_degree=kan_degree)
            for _ in range(num_blocks)
        ])
        self.recon = nn.Conv2d(feat_ch, in_ch, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Shallow & recon: nhỏ để gần-identity nhưng vẫn cho gradient
        nn.init.kaiming_normal_(self.shallow.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if self.shallow.bias is not None:
            nn.init.zeros_(self.shallow.bias)

        nn.init.kaiming_normal_(self.recon.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        with torch.no_grad():
            self.recon.weight.mul_(1e-2)  # rất nhỏ để out≈0 lúc đầu
        if self.recon.bias is not None:
            nn.init.zeros_(self.recon.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        f = self.shallow(x)
        for blk in self.blocks:
            f = blk(f)
        out = self.recon(f)
        return torch.clamp(inp + out, 0.0, 1.0)


# ---------------------------------------
#  LightningModule: training + validation
# ---------------------------------------
class ResKANLightning(pl.LightningModule):
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        num_blocks: int = 8,
        kan_K: int = 16,
        kan_degree: int = 3,
        lr: float = 2e-4,
        wd: float = 1e-8,
        lambda_charb: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_tv: float = 0.0,
        lambda_sobolev: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = ResKANNet(
            in_ch=in_ch, feat_ch=feat_ch, num_blocks=num_blocks,
            kan_K=kan_K, kan_degree=kan_degree
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _kan_sobolev_penalty(self) -> torch.Tensor:
        pen = 0.0
        for m in self.net.modules():
            if isinstance(m, KANCubic1D):
                pen = pen + m.sobolev_penalty()
        return pen

    def training_step(self, batch, batch_idx):
        lq, gt = batch  # (B,C,H,W) in [0,1]
        pred = self(lq)

        loss = 0.0
        if self.lambda_charb > 0:
            loss = loss + self.lambda_charb * self.loss_charb(pred, gt)
        if self.lambda_l1 > 0:
            loss = loss + self.lambda_l1 * nn.functional.l1_loss(pred, gt)
        if self.lambda_tv > 0:
            loss = loss + self.lambda_tv * tv_l1(pred)
        if self.lambda_sobolev > 0:
            loss = loss + self.lambda_sobolev * self._kan_sobolev_penalty()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0))
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
