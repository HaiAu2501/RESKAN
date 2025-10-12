# reskan_lightning.py
import math
from typing import List

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


def tv_l1(x):
    """Total variation L1 (nhẹ, optional)."""
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
        ll = torch.einsum('i,j->ij', h, h)  # 2x2
        lh = torch.einsum('i,j->ij', h, g)
        hl = torch.einsum('i,j->ij', g, h)
        hh = torch.einsum('i,j->ij', g, g)
        # Make 4 filters per channel
        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        weight = weight.unsqueeze(1)  # (4, 1, 2, 2)
        weight = weight.repeat(Fch, 1, 1, 1)  # (4F, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.groups = Fch
        self.conv = nn.Conv2d(in_channels=Fch, out_channels=4 * Fch, kernel_size=2, stride=2,
                              padding=0, groups=Fch, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(self.weight)
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            # pad reflect nếu kích thước lẻ
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
        # transpose-conv weight shape: (in_channels, out_channels_per_group, kH, kW)
        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        weight = weight.unsqueeze(1)  # (4, 1, 2, 2)
        weight = weight.repeat(Fch, 1, 1, 1)  # (4F, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.groups = Fch
        # in_channels = 4F, out_channels = F, groups=F => weight (4F, 1, 2, 2)
        self.deconv = nn.ConvTranspose2d(in_channels=4 * Fch, out_channels=Fch, kernel_size=2, stride=2,
                                         padding=0, groups=Fch, bias=False)
        with torch.no_grad():
            self.deconv.weight.copy_(self.weight)
        for p in self.deconv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.deconv(x)


# --------------------------------
#  KAN 1D spline (per-feature)
# --------------------------------
def _open_uniform_knots(K: int, degree: int, low: float = -1.0, high: float = 1.0, device=None):
    """
    Open-uniform knot vector: [low,...(p+1), interior..., high...(p+1)], total length = K+degree+1.
    Requires K >= degree+1; if not, we still build clamped extremes.
    """
    p = degree
    assert K >= p + 1, f"K must be >= degree+1; got K={K}, degree={p}"
    n_interior = K - p - 1
    if n_interior > 0:
        interior = torch.linspace(low, high, n_interior + 2, device=device)[1:-1]
        knots = torch.cat([
            torch.full((p + 1,), low, device=device),
            interior,
            torch.full((p + 1,), high, device=device)
        ], dim=0)
    else:
        # No interior knots, just clamped ends
        knots = torch.cat([
            torch.full((p + 1,), low, device=device),
            torch.full((p + 1,), high, device=device)
        ], dim=0)
    return knots  # shape: (K + p + 1,)


def _bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Cox–de Boor. x: (...), knots: (K+p+1,), degree=p.
    Returns basis B: (..., K)
    """
    p = degree
    K = knots.numel() - p - 1  # number of basis
    *spatial, = x.shape
    # N for p=0 has length (K+p)
    L = K + p
    # Expand dims for broadcasting with i-index
    x_exp = x.unsqueeze(-1)  # (..., 1)
    # Indicator for N_{i,0}
    t0 = knots[:-1]  # (K+p,)
    t1 = knots[1:]   # (K+p,)
    # shape-broadcast to (..., L)
    N = ((x_exp >= t0) & (x_exp < t1)).to(x.dtype)
    # Special case include right boundary at the last knot:
    N = torch.where((x_exp == knots[-1]), torch.cat([torch.zeros_like(N[..., :L-1]), torch.ones_like(N[..., :1])], dim=-1), N)

    # recursion
    for r in range(1, p + 1):
        Lr = K + p - (r - 1)      # current length
        # after update, new length will be Lr-1
        left_den = (knots[r: r + Lr - 1] - knots[:Lr - 1])  # (Lr-1,)
        right_den = (knots[r + 1: r + 1 + Lr - 1] - knots[1: Lr])  # (Lr-1,)

        left_num = x_exp - knots[:Lr - 1]           # (..., Lr-1)
        right_num = knots[r + 1: r + 1 + Lr - 1] - x_exp  # (..., Lr-1)

        left = torch.zeros_like(N[..., :Lr - 1])
        right = torch.zeros_like(N[..., :Lr - 1])

        nonzero = left_den != 0
        if nonzero.any():
            left = torch.where(
                nonzero,
                (left_num / left_den) * N[..., :Lr - 1],
                left
            )

        nonzero2 = right_den != 0
        if nonzero2.any():
            right = torch.where(
                nonzero2,
                (right_num / right_den) * N[..., 1:Lr],
                right
            )
        N = left + right  # (..., Lr-1)

    # N now has shape (..., K)
    return N


class KANSpline1D(nn.Module):
    """
    KAN 1D per-feature: phi_c(x) = id_gain_c * x + bias_c + sum_k alpha_{c,k} * B_{k,p}(a_c * x + b_c)
    - Shared over spatial positions; separate per-channel parameters.
    - Stable init: identity (id_gain=1, alpha=0, a=1, b=0).
    """
    def __init__(self, channels: int, K: int = 16, degree: int = 3):
        super().__init__()
        assert K >= degree + 1, "K must be >= degree+1 for open-uniform spline."
        self.C = channels
        self.K = K
        self.degree = degree

        # per-channel affine to map input to [-1,1] domain
        self.a = nn.Parameter(torch.ones(channels))   # scale
        self.b = nn.Parameter(torch.zeros(channels))  # shift

        # spline weights alpha (C, K)
        self.alpha = nn.Parameter(torch.zeros(channels, K))

        # linear skip + bias
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        # knot vector (fixed, on device later)
        self.register_buffer("knots_base", None, persistent=False)

    def _get_knots(self, device):
        if (self.knots_base is None) or (self.knots_base.device != device):
            self.knots_base = _open_uniform_knots(self.K, self.degree, low=-1.0, high=1.0, device=device)
        return self.knots_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) with C=self.C
        """
        B, C, H, W = x.shape
        assert C == self.C
        device = x.device
        knots = self._get_knots(device)

        # per-channel affine: x' = a_c * x + b_c
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        # clamp (optional, ổn định đầu kỳ):
        x_aff = torch.clamp(x_aff, -1.5, 1.5)

        # B-spline basis at every pixel, per channel
        # basis: (B, C, H, W, K)
        basis = _bspline_basis(x_aff, knots, self.degree)  # (B, C, H, W, K)

        # weight alpha per channel: (1, C, 1, 1, K)
        alpha = self.alpha.view(1, C, 1, 1, self.K)
        spline_val = (basis * alpha).sum(dim=-1)  # (B, C, H, W)

        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self):
        """
        Sobolev-like smoothness on spline coefficients: L2 of discrete 2nd derivative of alpha along k.
        """
        # alpha: (C, K)
        d1 = self.alpha[:, 1:] - self.alpha[:, :-1]          # (C, K-1)
        d2 = d1[:, 1:] - d1[:, :-1]                          # (C, K-2)
        return (d2 ** 2).mean()


# -------------------------
#  ResKAN Residual Block
# -------------------------
class ResKANBlock(nn.Module):
    def __init__(self, feat_ch: int, kernel_size: int = 3, kan_K: int = 16, kan_degree: int = 3):
        super().__init__()
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)

        in_ch = 4 * feat_ch
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)
        self.kan = KANSpline1D(in_ch, K=kan_K, degree=kan_degree)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)

        # lightweight channel attention (SE) — ổn định, phổ biến trong restoration
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // 8, 8), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, F, H, W)
        z = self.dwt(x)                        # (B, 4F, H/2, W/2)
        z = self.conv1(z)
        z = self.kan(z)
        z = self.conv2(z)
        z = z * self.se(z)
        y = self.idwt(z)                       # (B, F, H, W)
        return x + y                           # residual


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

    def forward(self, x):
        inp = x
        f = self.shallow(x)
        for blk in self.blocks:
            f = blk(f)
        out = self.recon(f)
        return torch.clamp(inp + out, 0.0, 1.0)  # global residual + clamp an toàn


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

    def forward(self, x):
        return self.net(x)

    def _kan_sobolev_penalty(self):
        pen = 0.0
        for m in self.net.modules():
            if isinstance(m, KANSpline1D):
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
            loss = loss + self.lambda_sobolev * self._kan_sobolev_penalty()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=lq.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1)
        gt = gt.clamp(0, 1)

        psnr = self.val_psnr(pred, gt)
        ssim = self.val_ssim(pred, gt)

        # yêu cầu: log val_psnr
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("val_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)

        # cũng log val_loss để theo dõi
        val_loss = self.loss_charb(pred, gt)
        self.log("val_loss", val_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0))

        return {"val_psnr": psnr, "val_ssim": ssim}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=self.lr * 0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
