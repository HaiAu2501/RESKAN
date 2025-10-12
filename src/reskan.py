# reskan_lite.py
import math
from typing import List, Optional

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
    # anisotropic TV-L1
    loss = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    return loss


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
        if (H % 2 != 0) or (W % 2 != 0):
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
#  KAN cubic basis (cardinal) utilities
# ---------------------------------------------------------
def _cubic_basis(t: torch.Tensor):
    # t in [0,1], return b0..b3 with shape broadcastable to t
    t2 = t * t
    t3 = t2 * t
    b0 = (1 - 3*t + 3*t2 - t3) / 6.0
    b1 = (4 - 6*t2 + 3*t3) / 6.0
    b2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
    b3 = t3 / 6.0
    return b0, b1, b2, b3


# ---------------------------------------------------------
#  KAN-LOWRANK 1D cubic spline (shared dictionary)
# ---------------------------------------------------------
class KANLowRankCubic1D(nn.Module):
    """
    KAN 1D cubic cardinal B-spline (degree=3) với phân rã hạng thấp alpha = W @ D.
    - C: số kênh
    - K: số điểm điều khiển (knots)
    - M: hạng (số "atoms") << C

    y = id_gain * x + bias + sum_{j=0..3} alpha_{c, i+j-1} * B_j(t)
    với i = floor(u), t = u - i, u map từ [-1,1] -> [0, K-1].
    """
    def __init__(self, channels: int, K: int = 32, rank_m: int = 8, clamp_val: float = 1.5):
        super().__init__()
        assert K >= 4, "Cubic cần K >= 4"
        self.C = channels
        self.K = K
        self.M = rank_m
        self.clamp_val = clamp_val

        # affine map per-channel: a*x + b (đưa về [-1,1])
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        # low-rank factors
        self.W = nn.Parameter(torch.randn(channels, rank_m) * 0.02)
        self.D = nn.Parameter(torch.randn(rank_m, K) * 0.02)
        # skip linear + bias
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        # biên miền [-1,1]
        self.register_buffer("low", torch.tensor(-1.0), persistent=False)
        self.register_buffer("high", torch.tensor(1.0), persistent=False)

    def _alpha(self) -> torch.Tensor:
        # (C,K)
        return self.W @ self.D

    def sobolev_penalty(self) -> torch.Tensor:
        # phạt độ cong trên D (atoms), ổn định & trơn
        d1 = self.D[:, 1:] - self.D[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        return (d2 ** 2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        assert C == self.C

        # affine + clamp
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        # map [-1,1] -> [0, K-1]
        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)              # (B,C,H,W)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)        # [0,1]

        # 4 neighbor indices, clamp
        i0 = (i - 1).clamp(0, self.K - 1)
        i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1)
        i3 = (i + 2).clamp(0, self.K - 1)

        # cubic basis
        b0, b1, b2, b3 = _cubic_basis(t)  # (B,C,H,W)

        # gather alpha at indices without materializing big tensors (only a view expand)
        alpha = self._alpha().view(1, C, 1, 1, self.K).expand(B, C, H, W, self.K)  # view
        a0 = torch.gather(alpha, -1, i0.unsqueeze(-1)).squeeze(-1)
        a1 = torch.gather(alpha, -1, i1.unsqueeze(-1)).squeeze(-1)
        a2 = torch.gather(alpha, -1, i2.unsqueeze(-1)).squeeze(-1)
        a3 = torch.gather(alpha, -1, i3.unsqueeze(-1)).squeeze(-1)

        spline_val = a0*b0 + a1*b1 + a2*b2 + a3*b3
        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y


# ---------------------------------------------------------
#  Depthwise-Separable Conv block (DW + PW)
# ---------------------------------------------------------
class DW_PW_Conv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        return x


# -------------------------
#  KAN-Lite Residual Block
# -------------------------
class KANLiteBlock(nn.Module):
    """
    Block nhẹ chạy ở độ phân giải thấp:
    z -> DW-PW -> KANLowRankCubic1D -> DW-PW -> SE -> +res
    """
    def __init__(self, feat_ch: int, kan_K: int = 32, kan_rank: int = 8):
        super().__init__()
        self.conv1 = DW_PW_Conv(feat_ch, 3)
        self.kan = KANLowRankCubic1D(feat_ch, K=kan_K, rank_m=kan_rank)
        self.conv2 = DW_PW_Conv(feat_ch, 3)

        # SE nhỏ gọn
        r = max(feat_ch // 16, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, r, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(r, feat_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.conv1(x)
        z = self.kan(z)
        z = self.conv2(z)
        z = z * self.se(z)
        return x + z


# -------------------------
#  Full ResKAN-Lite Network
# -------------------------
class ResKANLiteNet(nn.Module):
    """
    Kiến trúc: Shallow -> DWT (1 lần) -> bottleneck (1x1) -> [N blocks @ low-res]
               -> expand (1x1) -> IDWT -> Recon -> residual to input
    """
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        num_blocks: int = 8,
        kan_K: int = 32,
        kan_rank: int = 8,
        bottleneck_ch: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.F = feat_ch
        self.use_checkpoint = use_checkpoint

        self.shallow = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)

        # One-shot wavelet analysis/synthesis
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)

        # Bottleneck ở low-res để giảm kênh
        in_low = 4 * feat_ch
        self.Fm = bottleneck_ch if bottleneck_ch is not None else feat_ch  # thường = F
        self.reduce = nn.Conv2d(in_low, self.Fm, kernel_size=1, bias=True)

        # Stack blocks @ low-res
        self.blocks = nn.ModuleList([
            KANLiteBlock(self.Fm, kan_K=kan_K, kan_rank=kan_rank) for _ in range(num_blocks)
        ])

        # Expand kênh về 4F để iDWT
        self.expand = nn.Conv2d(self.Fm, in_low, kernel_size=1, bias=True)

        # Recon @ full-res
        self.recon = nn.Conv2d(feat_ch, in_ch, kernel_size=3, padding=1)

    def forward_lowres_stage(self, z):
        if not self.use_checkpoint:
            for blk in self.blocks:
                z = blk(z)
            return z
        else:
            # checkpoint từng block để giảm RAM đỉnh
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                z = checkpoint(blk, z)
            return z

    def forward(self, x):
        inp = x
        f = self.shallow(x)       # (B,F,H,W)
        z = self.dwt(f)           # (B,4F,H/2,W/2)
        z = self.reduce(z)        # (B,Fm,H/2,W/2)
        z = self.forward_lowres_stage(z)
        z = self.expand(z)        # (B,4F,H/2,W/2)
        f2 = self.idwt(z)         # (B,F,H,W)
        out = self.recon(f2)      # (B,in_ch,H,W)
        return torch.clamp(inp + out, 0.0, 1.0)


# ---------------------------------------
#  LightningModule: training + validation
# ---------------------------------------
class ResKANLiteLightning(pl.LightningModule):
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        num_blocks: int = 8,
        kan_K: int = 32,
        kan_rank: int = 8,
        bottleneck_ch: Optional[int] = None,
        use_checkpoint: bool = False,
        lr: float = 2e-4,
        wd: float = 1e-8,
        lambda_charb: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_tv: float = 0.0,
        lambda_sobolev: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = ResKANLiteNet(
            in_ch=in_ch,
            feat_ch=feat_ch,
            num_blocks=num_blocks,
            kan_K=kan_K,
            kan_rank=kan_rank,
            bottleneck_ch=bottleneck_ch,
            use_checkpoint=use_checkpoint,
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

    def forward(self, x):
        return self.net(x)

    def _sobolev_penalty(self):
        pen = 0.0
        for m in self.net.modules():
            if isinstance(m, KANLowRankCubic1D):
                pen = pen + m.sobolev_penalty()
        return pen

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
        if self.lambda_sobolev > 0:
            loss = loss + self.lambda_sobolev * self._sobolev_penalty()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=lq.size(0))
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
