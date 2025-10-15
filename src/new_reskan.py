# reskan_pyramid.py
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

def tv_l1(x):
    return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()

# --------------------------------------
#  Fixed Haar Wavelet DWT / iDWT blocks
# --------------------------------------
class HaarDWT2D(nn.Module):
    """Depthwise DWT Haar (stride=2). In: (B, F, H, W) -> Out: (B, 4F, H/2, W/2)"""
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
        if (H % 2) != 0 or (W % 2) != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        return self.conv(x)

class HaarIDWT2D(nn.Module):
    """Depthwise iDWT Haar (stride=2). In: (B, 4F, H, W) -> Out: (B, F, 2H, 2W)"""
    def __init__(self, channels: int):
        super().__init__()
        Fch = channels
        h = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        g = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])
        ll = torch.einsum('i,j->ij', h, h)
        lh = torch.einsum('i,j->ij', h, g)
        hl = torch.einsum('i,j->ij', g, h)
        hh = torch.einsum('i,j->ij', g, g)
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).repeat(Fch, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.deconv = nn.ConvTranspose2d(4 * Fch, Fch, kernel_size=2, stride=2, padding=0, groups=Fch, bias=False)
        with torch.no_grad():
            self.deconv.weight.copy_(self.weight)
        for p in self.deconv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.deconv(x)

# ---------------------------------------------------------
#  Group-KAN: shared spline per group + per-channel affine
# ---------------------------------------------------------
class KANGroup1D(nn.Module):
    """
    KAN 1D cubic B-spline, shared across G groups (G << C) + per-channel (a,b,id_gain,bias).
    Forward áp dụng theo từng vị trí pixel/kênh độc lập.

    alpha: (G, K)
    per-channel: a,b,id_gain,bias: (C,)
    group_idx: (C,) mapping kênh -> group
    """
    def __init__(self, channels: int, K: int = 32, groups: int = 32, clamp_val: float = 1.5):
        super().__init__()
        assert K >= 4
        self.C = channels
        self.K = K
        self.G = min(groups, channels)
        self.clamp_val = clamp_val

        # map kênh -> group đều nhất có thể
        gids = torch.arange(channels, dtype=torch.long)
        group_idx = (gids * self.G) // channels  # phân bố đều 0..G-1
        self.register_buffer("group_idx", group_idx, persistent=False)  # (C,)

        # spline weights chia sẻ theo group
        self.alpha = nn.Parameter(torch.zeros(self.G, K))

        # per-channel affine & skip
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.id_gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.register_buffer("low", torch.tensor(-1.0), persistent=False)
        self.register_buffer("high", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _cubic_basis(t: torch.Tensor):
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
        dev = x.device
        # per-channel affine + clamp
        x_aff = x * self.a.view(1, C, 1, 1) + self.b.view(1, C, 1, 1)
        if self.clamp_val is not None:
            x_aff = torch.clamp(x_aff, -self.clamp_val, self.clamp_val)

        # map [-1,1] -> [0, K-1]
        u = (x_aff - self.low) / (self.high - self.low) * (self.K - 1)
        i = torch.floor(u).to(torch.long)
        t = (u - i.to(u.dtype)).clamp(0.0, 1.0)

        i0 = (i - 1).clamp(0, self.K - 1)
        i1 = i.clamp(0, self.K - 1)
        i2 = (i + 1).clamp(0, self.K - 1)
        i3 = (i + 2).clamp(0, self.K - 1)
        b0, b1, b2, b3 = self._cubic_basis(t)

        # Lấy alpha theo group cho từng kênh: (C,K)
        alpha_pc = self.alpha.index_select(0, self.group_idx.to(self.alpha.device))  # (C,K)
        alpha_pc = alpha_pc.view(1, C, 1, 1, self.K).expand(B, C, H, W, self.K)

        a0 = torch.gather(alpha_pc, -1, i0.unsqueeze(-1)).squeeze(-1)
        a1 = torch.gather(alpha_pc, -1, i1.unsqueeze(-1)).squeeze(-1)
        a2 = torch.gather(alpha_pc, -1, i2.unsqueeze(-1)).squeeze(-1)
        a3 = torch.gather(alpha_pc, -1, i3.unsqueeze(-1)).squeeze(-1)

        spline_val = a0*b0 + a1*b1 + a2*b2 + a3*b3
        y = self.id_gain.view(1, C, 1, 1) * x + spline_val + self.bias.view(1, C, 1, 1)
        return y

    def sobolev_penalty(self):
        d1 = self.alpha[:, 1:] - self.alpha[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        return (d2 ** 2).mean()

# -------------------------
#  Blocks: DSConv + KAN + SE
# -------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(ch, ch, kernel_size, padding=pad, groups=ch, bias=True)
        self.pw1 = nn.Conv2d(ch, ch, 1, bias=True)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.act(self.pw1(x))
        x = self.pw2(x)
        return x

class SE(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(ch // r, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class ResKANLiteBlock(nn.Module):
    def __init__(self, ch: int, kan_K: int = 32, kan_groups: int = 32, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ds1 = DepthwiseSeparableConv(ch, 3)
        self.kan = KANGroup1D(ch, K=kan_K, groups=kan_groups)
        self.ds2 = DepthwiseSeparableConv(ch, 3)
        self.se = SE(ch, r=8)

    def _forward_impl(self, x):
        z = self.ds1(x)
        z = self.kan(z)
        z = self.ds2(z)
        z = self.se(z)
        return z

    def forward(self, x):
        if self.use_checkpoint and self.training:
            import torch.utils.checkpoint as cp
            z = cp.checkpoint(self._forward_impl, x)
        else:
            z = self._forward_impl(x)
        return x + z

# -------------------------
#  Pyramid ResKAN Network
# -------------------------
class ResKANPyramid(nn.Module):
    """
    Wavelet Pyramid:
      Level-0 (H,W): shallow conv
      DWT -> Level-1 (H/2): compress -> n1 blocks -> DWT -> Level-2 (H/4): compress -> n2 blocks
           -> expand -> iDWT to Level-1 -> tail blocks -> expand -> iDWT to Level-0 -> recon
    """
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 128,            # base F
        n_blocks_l1: int = 6,
        n_blocks_l2: int = 12,
        compress1: float = 0.5,        # after first DWT (4F -> C1 = 4F*compress1)
        compress2: float = 0.5,        # after second DWT (4*C1 -> C2 = 4*C1*compress2)
        kan_K: int = 32,
        kan_groups: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        F = feat_ch
        self.shallow = nn.Conv2d(in_ch, F, 3, padding=1)

        # Level-1 (H/2)
        self.dwt1 = HaarDWT2D(F)               # F -> 4F
        C1_in = 4 * F
        C1 = int(C1_in * compress1)
        self.l1_down = nn.Conv2d(C1_in, C1, 1)

        self.l1_blocks_head = nn.ModuleList([
            ResKANLiteBlock(C1, kan_K=kan_K, kan_groups=kan_groups, use_checkpoint=use_checkpoint)
            for _ in range(n_blocks_l1 // 2)
        ])

        # Level-2 (H/4)
        self.dwt2 = HaarDWT2D(C1)             # C1 -> 4*C1
        C2_in = 4 * C1
        C2 = int(C2_in * compress2)
        self.l2_down = nn.Conv2d(C2_in, C2, 1)

        self.l2_blocks = nn.ModuleList([
            ResKANLiteBlock(C2, kan_K=kan_K, kan_groups=kan_groups, use_checkpoint=use_checkpoint)
            for _ in range(n_blocks_l2)
        ])

        self.l2_up = nn.Conv2d(C2, C2_in, 1)
        self.idwt2 = HaarIDWT2D(C1)           # 4*C1 -> C1

        # Level-1 tail
        self.l1_blocks_tail = nn.ModuleList([
            ResKANLiteBlock(C1, kan_K=kan_K, kan_groups=kan_groups, use_checkpoint=use_checkpoint)
            for _ in range(n_blocks_l1 - len(self.l1_blocks_head))
        ])

        self.l1_up = nn.Conv2d(C1, C1_in, 1)
        self.idwt1 = HaarIDWT2D(F)            # 4F -> F

        self.recon = nn.Conv2d(F, in_ch, 3, padding=1)

    def forward(self, x):
        inp = x
        f0 = self.shallow(x)                   # (B,F,H,W)

        # Level-1
        z1 = self.dwt1(f0)                     # (B,4F,H/2,W/2)
        z1 = self.l1_down(z1)                  # (B,C1,H/2,W/2)

        for blk in self.l1_blocks_head:
            z1 = blk(z1)

        # Level-2
        z2 = self.dwt2(z1)                     # (B,4*C1,H/4,W/4)
        z2 = self.l2_down(z2)                  # (B,C2,H/4,W/4)
        for blk in self.l2_blocks:
            z2 = blk(z2)
        z2 = self.l2_up(z2)                    # (B,4*C1,H/4,W/4)
        z1 = self.idwt2(z2) + z1               # skip Level-2 -> Level-1

        for blk in self.l1_blocks_tail:
            z1 = blk(z1)

        z1 = self.l1_up(z1)                    # (B,4F,H/2,W/2)
        f0 = self.idwt1(z1) + f0               # back to Level-0

        out = self.recon(f0)
        return torch.clamp(inp + out, 0.0, 1.0)

# ---------------------------------------
#  LightningModule: training + validation
# ---------------------------------------
class ResKANPyrLightning(pl.LightningModule):
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 128,
        n_blocks_l1: int = 6,
        n_blocks_l2: int = 12,
        compress1: float = 0.5,
        compress2: float = 0.5,
        kan_K: int = 32,
        kan_groups: int = 32,
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

        self.net = ResKANPyramid(
            in_ch=in_ch,
            feat_ch=feat_ch,
            n_blocks_l1=n_blocks_l1,
            n_blocks_l2=n_blocks_l2,
            compress1=compress1,
            compress2=compress2,
            kan_K=kan_K,
            kan_groups=kan_groups,
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

    def _kan_sobolev_penalty(self):
        pen = 0.0
        for m in self.net.modules():
            if isinstance(m, KANGroup1D):
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
