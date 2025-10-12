# reskan_lightning_memopt.py
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
#  Memory-efficient local B-spline evaluation (p+1 only)
# ---------------------------------------------------------
def _open_uniform_knots(K: int, degree: int, low: float = -1.0, high: float = 1.0, device=None, dtype=None):
    p = degree
    assert K >= p + 1, f"K must be >= degree+1; got K={K}, degree={p}"
    n_interior = K - p - 1
    if n_interior > 0:
        interior = torch.linspace(low, high, n_interior + 2, device=device, dtype=dtype)[1:-1]
        knots = torch.cat([
            torch.full((p + 1,), low, device=device, dtype=dtype),
            interior,
            torch.full((p + 1,), high, device=device, dtype=dtype)
        ], dim=0)
    else:
        knots = torch.cat([
            torch.full((p + 1,), low, device=device, dtype=dtype),
            torch.full((p + 1,), high, device=device, dtype=dtype)
        ], dim=0)
    return knots  # shape: (K + p + 1,)


def _find_span_bucketize(x: torch.Tensor, knots: torch.Tensor, K: int, p: int):
    """
    Return span index i in [p, K-1] s.t. knots[i] <= x < knots[i+1].
    Vectorized via torch.bucketize.
    """
    # edges for bucketize correspond to knots[p:K]
    edges = knots[p:K]  # length K-p
    i = torch.bucketize(x, edges, right=False) + p - 1
    # Handle right boundary x == knots[-1]
    i = torch.where(x >= knots[K], torch.full_like(i, K - 1), i)
    # Clamp just in case
    return i.clamp(min=p, max=K - 1)


def _bspline_local_values(x: torch.Tensor, knots: torch.Tensor, K: int, p: int):
    """
    Compute local B-spline basis values N_{i-p..i,p}(x) of length (p+1) and the start index (i-p),
    for every element of x. All vectorized over x; loops only over degree p (small).
    Returns:
      N: (*, p+1)
      start: (*)  (int64), such that these (p+1) values correspond to basis indices start..start+p
    """
    device = x.device
    dtype = x.dtype
    i = _find_span_bucketize(x, knots, K, p)                  # (*)
    start = i - p                                             # (*)

    # Build left[j] = x - knots[i+1-j], right[j] = knots[i+j] - x for j=1..p
    # Prepare index tensors for gather
    shape = x.shape
    # indices for left/right per j
    js = torch.arange(1, p + 1, device=device)                # (p,)
    idx_left = (i.unsqueeze(-1) + 1 - js)                     # (*, p)
    idx_right = (i.unsqueeze(-1) + js)                        # (*, p)

    # gather knots
    t_left = knots.index_select(0, idx_left.reshape(-1)).reshape(*shape, p)   # (*, p)
    t_right = knots.index_select(0, idx_right.reshape(-1)).reshape(*shape, p) # (*, p)

    x_exp = x.unsqueeze(-1)                                   # (*, 1)
    left = x_exp - t_left                                     # (*, p)
    right = t_right - x_exp                                   # (*, p)

    # Cox–de Boor local recursion
    N = x.new_zeros(*shape, p + 1)                            # (*, p+1)
    N[..., 0] = 1.0
    for j in range(1, p + 1):
        saved = x.new_zeros(*shape)
        for r in range(j):
            denom = right[..., r] + left[..., j - r - 1]      # (*)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # avoid div by zero
            temp = N[..., r] / denom
            N[..., r] = saved + right[..., r] * temp
            saved = left[..., j - r - 1] * temp
        N[..., j] = saved
    return N, start


# --------------------------------
#  KAN 1D spline (per-feature)
#  ==> Memory-efficient version
# --------------------------------
class KANSpline1D(nn.Module):
    """
    KAN 1D per-feature (memory-efficient):
    phi_c(x) = id_gain_c * x + bias_c + sum_{j=0..p} alpha_{c, start+j} * N_j(x)
    - Only (p+1) local basis values per pixel (vs. K previously).
    - Shared knots across channels; per-channel scale/shift 'a','b' map inputs to [-1,1].
    """
    def __init__(self, channels: int, K: int = 16, degree: int = 3, clamp_val: float = 1.5):
        super().__init__()
        assert K >= degree + 1, "K must be >= degree+1 for open-uniform spline."
        self.C = channels
        self.K = K
        self.p = degree
        self.clamp_val = clamp_val

        # per-channel affine to map input to [-1,1]
        self.a = nn.Parameter(torch.ones(channels))   # scale
        self.b = nn.Parameter(torch.zeros(channels))  # shift

        # spline weights alpha (C, K)
        self.alpha = nn.Parameter(torch.zeros(channels, K))

        # linear skip + bias
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.register_buffer("knots", None, persistent=False)

    def _ensure_knots(self, device, dtype):
        if (self.knots is None) or (self.knots.device != device) or (self.knots.dtype != dtype):
            self.knots = _open_uniform_knots(self.K, self.p, low=-1.0, high=1.0, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) with C=self.C
        """
        B, C, H, W = x.shape
        assert C == self.C
        self._ensure_knots(x.device, x.dtype)

        # per-channel affine: x' = a_c * x + b_c, then clamp (ổn định)
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        # --- Local B-spline values (only p+1 per pixel) ---
        # shape of N: (B, C, H, W, p+1); start: (B, C, H, W)
        N, start = _bspline_local_values(x_aff, self.knots, self.K, self.p)

        # --- Gather local alpha weights per-channel ---
        # Build indices for K dimension: start + [0..p]
        offs = torch.arange(0, self.p + 1, device=x.device).view(*(1,)*4 + (self.p + 1,))  # (1,1,1,1,p+1)
        idx = (start.unsqueeze(-1) + offs).clamp_(0, self.K - 1)                            # (B,C,H,W,p+1)

        # alpha: (C, K) -> (1,C,1,1,K)
        alpha = self.alpha.view(1, C, 1, 1, self.K)
        alpha_loc = torch.gather(alpha.expand(B, -1, H, W, -1), dim=-1, index=idx)         # (B,C,H,W,p+1)

        spline_val = (N * alpha_loc).sum(dim=-1)  # (B,C,H,W)

        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self):
        # alpha: (C, K)
        d1 = self.alpha[:, 1:] - self.alpha[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
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

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // 8, 8), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.dwt(x)            # (B, 4F, H/2, W/2)
        z = self.conv1(z)
        z = self.kan(z)
        z = self.conv2(z)
        z = z * self.se(z)
        y = self.idwt(z)
        return x + y


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

        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("val_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)

        val_loss = self.loss_charb(pred, gt)
        self.log("val_loss", val_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0))

        return {"val_psnr": psnr, "val_ssim": ssim}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=self.lr * 0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
