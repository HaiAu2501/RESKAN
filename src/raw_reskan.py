# reskan.py 
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

class KANCubic1D(nn.Module):
    """
    KAN 1D cubic cardinal B-spline (degree=3), memory-safe & no in-place.
    phi_c(x) = id_gain_c * x + bias_c + sum_{j=0..3} alpha_{c, i+j-1} * B_j(t),
    với i = floor(u), t = u - i, u map từ [-1,1] -> [0, K-1].
    """
    def __init__(self, channels: int, K: int = 32, clamp_val: float = 1.5):
        super().__init__()
        assert K >= 4, "Cubic cần K >= 4"
        self.C = channels
        self.K = K
        self.clamp_val = clamp_val
        # map per-channel: a*x + b để phủ [-1,1]
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        # spline weights
        self.alpha = nn.Parameter(torch.zeros(channels, K))
        # skip linear + bias
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        # biên miền [-1,1]
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
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        assert C == self.C
        # affine + clamp (không in-place)
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        # map [-1,1] -> [0, K-1]
        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)              # (B,C,H,W)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)        # [0,1]

        # 4 chỉ số lân cận, clamp biên (KHÔNG in-place)
        i0 = (i - 1).clamp(0, self.K - 1)
        i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1)
        i3 = (i + 2).clamp(0, self.K - 1)

        # 4 weights cubic theo t
        b0, b1, b2, b3 = self._cubic_basis(t)          # (B,C,H,W) each

        # gather alpha theo 4 chỉ số (dùng take_along_dim/gather — không in-place)
        alpha = self.alpha.view(1, C, 1, 1, self.K).expand(B, C, H, W, self.K)  # view an toàn (chỉ đọc)
        a0 = torch.gather(alpha, -1, i0.unsqueeze(-1)).squeeze(-1)
        a1 = torch.gather(alpha, -1, i1.unsqueeze(-1)).squeeze(-1)
        a2 = torch.gather(alpha, -1, i2.unsqueeze(-1)).squeeze(-1)
        a3 = torch.gather(alpha, -1, i3.unsqueeze(-1)).squeeze(-1)

        spline_val = a0*b0 + a1*b1 + a2*b2 + a3*b3      # (B,C,H,W)
        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self):
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
        self.kan = KANCubic1D(in_ch, K=32)  # degree=3 fixed, mạnh & gọn
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1),
            nn.ReLU(),  # <- KHÔNG inplace
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
            loss = loss + self.lambda_l1 * self.loss_l1(pred, gt)
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
