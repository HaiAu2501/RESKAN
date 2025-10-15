# reskan_universal.py
# Universal KAN Restoration (task-agnostic): Learn a convex data energy Dθ(x; y)
# and a separable 1D KAN prior in a forward-backward (unrolled) scheme.
# - Data gradient: ∇_x Dθ(x;y) = K^T (Kx - B(y)), with K depthwise conv (learned) and B(y) a learned predictor.
# - Prior prox: Ψ (Haar DWT) → KAN (1D per-coefficient) → grouped 1×1 → Ψ^T.
# - No explicit A. Train once, apply to dehaze/derain/denoise/deblur/... with the SAME architecture.
# -----------------------------------------------------------------------------------------------

import math
from typing import Optional, Tuple
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def tv_l1(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


# ----------------------------
#  Haar DWT / iDWT (depthwise)
# ----------------------------
class HaarDWT2D(nn.Module):
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
        with torch.no_grad(): self.conv.weight.copy_(self.weight)
        for p in self.conv.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        return self.conv(x)


class HaarIDWT2D(nn.Module):
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
        self.tconv = nn.ConvTranspose2d(4 * Fch, Fch, kernel_size=2, stride=2, padding=0, groups=Fch, bias=False)
        with torch.no_grad(): self.tconv.weight.copy_(self.weight)
        for p in self.tconv.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv(x)


# -------------------------------------------
#  1D Cubic B-spline KAN (separable per-chan)
# -------------------------------------------
class KANCubic1D(nn.Module):
    def __init__(self, channels: int, K: int = 32, clamp_val: float = 1.5):
        super().__init__()
        assert K >= 4
        self.C, self.K, self.clamp_val = channels, K, clamp_val
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.alpha = nn.Parameter(torch.zeros(channels, K))
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("low", torch.tensor(-1.0))
        self.register_buffer("high", torch.tensor(1.0))

    @staticmethod
    def _cubic_basis(t: torch.Tensor):
        t2 = t * t; t3 = t2 * t
        B0 = (1 - 3*t + 3*t2 - t3) / 6.0
        B1 = (4 - 6*t2 + 3*t3) / 6.0
        B2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
        B3 = t3 / 6.0
        return B0, B1, B2, B3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape; assert C == self.C
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)
        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)

        i0 = (i - 1).clamp(0, self.K - 1); i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1); i3 = (i + 2).clamp(0, self.K - 1)

        HW = H * W
        i0f = i0.view(B, C, HW); i1f = i1.view(B, C, HW)
        i2f = i2.view(B, C, HW); i3f = i3.view(B, C, HW)

        alpha = self.alpha.unsqueeze(0)  # (1,C,K)
        a0 = torch.gather(alpha, 2, i0f).view(B, C, H, W)
        a1 = torch.gather(alpha, 2, i1f).view(B, C, H, W)
        a2 = torch.gather(alpha, 2, i2f).view(B, C, H, W)
        a3 = torch.gather(alpha, 2, i3f).view(B, C, H, W)

        B0, B1, B2, B3 = self._cubic_basis(t)
        spline_val = a0 * B0 + a1 * B1 + a2 * B2 + a3 * B3
        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self) -> torch.Tensor:
        d1 = self.alpha[:, 1:] - self.alpha[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        return (d2 ** 2).mean()


# ----------------------------------------------------------
#  Prox-like block: DWT → (depthwise 3×3) → KAN → group-1×1
# ----------------------------------------------------------
class ProxKANSubbandBlock(nn.Module):
    def __init__(self, feat_ch: int, kan_K: int = 32, use_depthwise_context: bool = True, conv3_ks: int = 3):
        super().__init__()
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)
        in_ch = 4 * feat_ch
        pad = conv3_ks // 2
        self.dw = nn.Identity() if not use_depthwise_context else nn.Conv2d(in_ch, in_ch, kernel_size=conv3_ks, padding=pad, groups=in_ch, bias=True)
        self.kan = KANCubic1D(in_ch, K=kan_K)
        self.mix = nn.Conv2d(in_ch, in_ch, kernel_size=1, groups=4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dwt(x)
        z = self.dw(z)         # local context (no mixing across subbands)
        z = self.kan(z)        # separable 1D shrinkage
        z = self.mix(z)        # grouped mixing within subband
        y = self.idwt(z)
        return x + y

    def sobolev_penalty(self) -> torch.Tensor:
        return self.kan.sobolev_penalty()


# ------------------------------------------------------
#  Universal "Data Gradient": ∇_x Dθ(x;y) = K^T (Kx - B(y))
#  - K: depthwise conv (learned). SPD Hessian = K^T K  ⪰ 0.
#  - B(y): learned predictor (small CNN) with same channels as Kx.
#  Optional per-channel positive gate s(y) (Softplus) for adaptivity.
# ------------------------------------------------------
class LearnedDataGradient(nn.Module):
    def __init__(self, channels: int, ksize: int = 7, use_gate: bool = True):
        super().__init__()
        assert ksize % 2 == 1
        self.C = channels
        self.ks = ksize
        self.pad = ksize // 2

        # Depthwise kernel K (shared across space; learned)
        # shape (C,1,ks,ks); small init ~ Gaussian-like
        k = torch.linspace(-1, 1, ksize)
        xx, yy = torch.meshgrid(k, k, indexing="ij")
        kern = torch.exp(-(xx**2 + yy**2) / (2 * 0.6**2))
        kern = kern / kern.sum()
        w0 = kern.view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1)
        self.kernel = nn.Parameter(w0)

        # Predictor B(y): small CNN → (B,C,H,W)
        hidden = max(16, channels // 3)
        self.pred = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1)
        )

        # Optional per-channel positive gate s(y) ≥ 0
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(8, channels // 8), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, channels // 8), channels, 1),
                nn.Softplus()  # ≥ 0
            )

    def _conv_dw(self, x: torch.Tensor, flip: bool = False) -> torch.Tensor:
        k = self.kernel
        if flip: k = torch.flip(k, dims=[2, 3])
        xpad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(xpad, k, bias=None, stride=1, padding=0, groups=self.C)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = self._conv_dw(x, flip=False)      # K x
        b = self.pred(y).to(x.dtype)          # B(y)
        r = q - b                              # residual in K-space
        if self.use_gate:
            s = self.gate(y).to(x.dtype)      # (B,C,1,1), ≥0
            r = r * s
        g = self._conv_dw(r, flip=True)       # K^T r   (adjoint via flipped kernel + reflect pad)
        return g                               # ∇_x Dθ(x;y)


# ---------------------------------------
#  Unrolled universal network (task-agn.)
# ---------------------------------------
class ResKANUniversal(nn.Module):
    """
    T steps of forward-backward:
      x_{k+1} = prox_{τR} ( x_k - τ ∇_x Dθ(x_k; y) ), realized as:
        v  = x - τ * DataGrad(x,y)
        u  = ProxKAN(v)   # Ψ → KAN 1D → group-1x1 → Ψ^T (+res)
        x  = u + γ x
    """
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 48,
        T: int = 8,
        kan_K: int = 32,
        tau: float = 1.0,
        gamma: float = 0.0,
        ksize_data: int = 7,
        use_gate: bool = True
    ):
        super().__init__()
        self.T, self.tau, self.gamma = T, tau, gamma
        self.head = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.tail = nn.Conv2d(feat_ch, in_ch, 3, padding=1)
        self.block = ProxKANSubbandBlock(feat_ch, kan_K=kan_K, use_depthwise_context=True, conv3_ks=3)
        self.data_grad = LearnedDataGradient(in_ch, ksize=ksize_data, use_gate=use_gate)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Warm-start in feature space
        x = self.head(y)
        for _ in range(self.T):
            # map to image space for data gradient
            img = self.tail(x)
            g = self.data_grad(img, y)     # ∇_x Dθ(img; y)
            g = self.head(g)               # bring to feature space
            v = x - self.tau * g           # forward step (data)
            u = self.block(v)              # backward step (prox-KAN)
            x = u + self.gamma * x         # relaxation
        out = self.tail(x)
        return out.clamp(0.0, 1.0)

    def sobolev_penalty(self) -> torch.Tensor:
        return self.block.sobolev_penalty()


# --------------------------
#  Feed-forward (Agnostic)
# --------------------------
class ResKANNet(nn.Module):
    """
    Pure feed-forward (task-agnostic): stack of ProxKANSubbandBlock
    """
    def __init__(self, in_ch: int = 3, feat_ch: int = 48, num_blocks: int = 8, kan_K: int = 32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.blocks = nn.ModuleList([ProxKANSubbandBlock(feat_ch, kan_K=kan_K) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(feat_ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = feat
        for blk in self.blocks:
            res = blk(res)
        out = self.tail(res)
        return (x + out).clamp(0.0, 1.0)

    def sobolev_penalty(self) -> torch.Tensor:
        return sum([blk.sobolev_penalty() for blk in self.blocks]) / len(self.blocks)


# --------------------------
#  LightningModule wrapper
# --------------------------
class ResKAN(pl.LightningModule):
    """
    Two modes:
      - universal_unrolled: task-agnostic forward-backward with learned data energy
      - feedforward: task-agnostic deep prior only
    """
    def __init__(
        self,
        mode: str = "universal_unrolled",  # or "feedforward"
        in_ch: int = 3,
        feat_ch: int = 48,
        num_blocks: int = 8,
        kan_K: int = 32,
        T: int = 8,
        tau: float = 1.0,
        gamma: float = 0.0,
        ksize_data: int = 7,
        use_gate: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_sobolev: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        if mode == "universal_unrolled":
            self.net = ResKANUniversal(
                in_ch=in_ch, feat_ch=feat_ch, T=T, kan_K=kan_K,
                tau=tau, gamma=gamma, ksize_data=ksize_data, use_gate=use_gate
            )
        elif mode == "feedforward":
            self.net = ResKANNet(in_ch=in_ch, feat_ch=feat_ch, num_blocks=num_blocks, kan_K=kan_K)
        else:
            raise ValueError("mode must be 'universal_unrolled' or 'feedforward'")

        self.loss_charb = CharbonnierLoss(1e-3)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.lr = lr
        self.wd = weight_decay
        self.lambda_tv = lambda_tv
        self.lambda_sobolev = lambda_sobolev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _sobolev_penalty(self) -> torch.Tensor:
        return self.net.sobolev_penalty()

    def training_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1); gt = gt.clamp(0, 1)
        loss = self.loss_charb(pred, gt)
        if self.lambda_tv > 0: loss = loss + self.lambda_tv * tv_l1(pred)
        if self.lambda_sobolev > 0: loss = loss + self.lambda_sobolev * self._sobolev_penalty()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1); gt = gt.clamp(0, 1)
        psnr = self.val_psnr(pred, gt); ssim = self.val_ssim(pred, gt)
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("val_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("val_loss", self.loss_charb(pred, gt), on_step=False, on_epoch=True, batch_size=lq.size(0))
        return {"val_psnr": psnr, "val_ssim": ssim}

    def test_step(self, batch, batch_idx):
        lq, gt = batch
        pred = self(lq).clamp(0, 1); gt = gt.clamp(0, 1)
        psnr = self.val_psnr(pred, gt); ssim = self.val_ssim(pred, gt)
        self.log("test_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        self.log("test_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, batch_size=lq.size(0), sync_dist=False)
        return {"test_psnr": psnr, "test_ssim": ssim}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=self.lr * 0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
