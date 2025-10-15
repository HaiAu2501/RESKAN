# reskan.py
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


# ----------------------------
#  Norm & small helper layers
# ----------------------------
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B,C,H,W)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# -------------------------------------------
#  Multi-Axis Gating (MAXIM-inspired, light)
# -------------------------------------------
class MultiAxisGating(nn.Module):
    """
    Tuyến tính theo số điểm ảnh. Nhánh trục-H (kx1 depthwise), trục-W (1xk depthwise),
    và nhánh global-grid (AdaptiveAvgPool -> 1x1 -> upsample). Trộn bằng pointwise conv.
    """
    def __init__(self, channels: int, expand: float = 2.0, kernel_size: int = 7, grid_size: int = 8):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        hidden = max(int(channels * expand), 8)
        pad = kernel_size // 2

        self.pw_in = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)

        # axial depthwise conv
        self.dw_h = nn.Conv2d(hidden, hidden, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              groups=hidden, bias=True)
        self.dw_w = nn.Conv2d(hidden, hidden, kernel_size=(1, kernel_size), padding=(0, pad),
                              groups=hidden, bias=True)

        # global grid pooling branch
        self.grid_size = grid_size
        self.pw_grid = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)

        # fuse
        self.pw_out = nn.Conv2d(hidden * 3, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        y = self.norm(x)
        y = self.pw_in(y)
        y_h = F.gelu(self.dw_h(y))
        y_w = F.gelu(self.dw_w(y))

        # global-grid pooling
        g = F.adaptive_avg_pool2d(y, output_size=self.grid_size)
        g = F.gelu(self.pw_grid(g))
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)

        fused = torch.cat([y_h, y_w, g], dim=1)
        out = self.pw_out(fused)  # (B,C,H,W)
        return out


class CrossGating(nn.Module):
    """
    Cross-gating nhánh chéo: gate(X) = sigma(MAG(Y)), gate(Y) = sigma(MAG(X))
    """
    def __init__(self, channels: int, expand: float = 2.0, kernel_size: int = 7, grid_size: int = 8):
        super().__init__()
        self.gx = MultiAxisGating(channels, expand, kernel_size, grid_size)
        self.gy = MultiAxisGating(channels, expand, kernel_size, grid_size)

    def forward(self, x, y):
        # x,y: (B,C,H,W)
        gate_y = torch.sigmoid(self.gx(y))
        gate_x = torch.sigmoid(self.gy(x))
        x_hat = x * gate_y
        y_hat = y * gate_x
        return x_hat, y_hat


# ------------------------------------------------------
#  Cross-Subband Low-Rank Coupler (within 4-band groups)
# ------------------------------------------------------
class CrossSubbandLowRankCoupler(nn.Module):
    """
    Trộn 4 dải LL/LH/HL/HH theo từng kênh gốc bằng biến đổi (I + U V^T), hạng thấp r.
    In:  (B, 4F, H, W)
    Out: (B, 4F, H, W)
    """
    def __init__(self, feat_ch: int, rank: int = 1):
        super().__init__()
        self.groups = feat_ch   # mỗi group tương ứng 4 subbands
        self.rank = rank
        # U, V: (G, 4, r). Khởi tạo nhỏ để gần identity.
        self.U = nn.Parameter(torch.zeros(self.groups, 4, rank))
        self.V = nn.Parameter(torch.zeros(self.groups, 4, rank))

    def forward(self, z):
        # z: (B, 4F, H, W)
        B, C, H, W = z.shape
        G = self.groups
        assert C == 4 * G, f"Expected channels = 4*feat_ch. Got {C} vs 4*{G}"

        z_rg = z.view(B, G, 4, H, W)  # (B,G,4,H,W)
        # W_g = I + U_g V_g^T, tính bằng einsum
        # UVT: (G,4,4)
        UVT = torch.einsum('gik,gjk->gij', self.U, self.V)
        I = torch.eye(4, device=z.device, dtype=z.dtype).unsqueeze(0).expand(G, 4, 4)
        Wg = I + UVT  # (G,4,4)

        # apply: out[b,g,i,h,w] = sum_j Wg[g,i,j] * z_rg[b,g,j,h,w]
        out = torch.einsum('gij,bgjhw->bgihw', Wg, z_rg)  # (B,G,4,H,W)
        out = out.reshape(B, 4 * G, H, W)
        return out


# -----------------------------------
#  MoE-KAN: mixture of KANCubic1D
# -----------------------------------
class MoEKAN1D(nn.Module):
    """
    M chuyên gia KAN theo kênh. Định tuyến mềm bằng GAP -> 1x1 -> softmax.
    Gọn VRAM: trọng số gate theo (B,M,1,1), không theo (B,M,C,H,W).
    """
    def __init__(self, channels: int, K: int = 32, num_experts: int = 3, gate_hidden: int | None = None):
        super().__init__()
        self.M = num_experts
        self.experts = nn.ModuleList([KANCubic1D(channels, K=K) for _ in range(num_experts)])
        gh = gate_hidden if gate_hidden is not None else max(channels // 4, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gh, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gh, num_experts, kernel_size=1)
        )

    def forward(self, z):
        # z: (B,C,H,W)
        B, C, H, W = z.shape
        logits = self.gate(z)                 # (B,M,1,1)
        weights = torch.softmax(logits, dim=1)  # (B,M,1,1)

        outs = [exp(z) for exp in self.experts]             # list of (B,C,H,W)
        stack = torch.stack(outs, dim=1)                    # (B,M,C,H,W)
        y = (weights.unsqueeze(2) * stack).sum(dim=1)       # (B,C,H,W)
        return y

    def sobolev_penalty(self):
        pen = 0.0
        for exp in self.experts:
            pen = pen + exp.sobolev_penalty()
        return pen / self.M


# -------------------------
#  ResKAN+ Residual Block
# -------------------------
class ResKANBlock(nn.Module):
    """
    ResKAN+ block:
      DWT -> Conv(3x3) -> Norm -> CSLC(4-band low-rank) ->
      Split 1/2 ch -> CrossGating (multi-axis) -> Concat ->
      MoE-KAN -> Conv(3x3) -> SE -> iDWT -> Residual add
    """
    def __init__(
        self,
        feat_ch: int,
        kernel_size: int = 3,
        kan_K: int = 32,
        moe_experts: int = 3,
        gate_expand: float = 2.0,
        gate_kernel: int = 7,
        gate_grid: int = 8,
        cslc_rank: int = 1
    ):
        super().__init__()
        self.dwt = HaarDWT2D(feat_ch)
        self.idwt = HaarIDWT2D(feat_ch)

        in_ch = 4 * feat_ch
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)
        self.norm1 = LayerNorm2d(in_ch)

        # Low-rank subband coupler
        self.cslc = CrossSubbandLowRankCoupler(feat_ch, rank=cslc_rank)

        # Cross-gating operates on half channels each
        assert in_ch % 2 == 0, "in_ch must be even to split"
        self.cross_gate = CrossGating(in_ch // 2, expand=gate_expand, kernel_size=gate_kernel, grid_size=gate_grid)

        # MoE-KAN on full channels after concat
        self.moe_kan = MoEKAN1D(in_ch, K=kan_K, num_experts=moe_experts)

        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=1, bias=True)

        # Squeeze-Excite (task-agnostic channel reweight)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(in_ch // 8, 8), kernel_size=1),
            nn.ReLU(),  # not inplace
            nn.Conv2d(max(in_ch // 8, 8), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B,F,H,W)
        z = self.dwt(x)                        # (B, 4F, H/2, W/2)
        z = self.conv1(z)
        z = F.gelu(self.norm1(z))

        # low-rank subband coupling per 4-group
        z = self.cslc(z)                       # (B, 4F, H/2, W/2)

        # cross-gating between two channel halves
        z1, z2 = torch.chunk(z, chunks=2, dim=1)  # each (B, 2F, H/2, W/2)
        z1g, z2g = self.cross_gate(z1, z2)
        z = torch.cat([z1g, z2g], dim=1)       # (B, 4F, H/2, W/2)

        # MoE-KAN nonlinearity (per-channel)
        z = self.moe_kan(z)
        z = self.conv2(z)

        # SE channel gating
        z = z * self.se(z)

        y = self.idwt(z)                       # (B, F, H, W)
        return x + y


# -------------------------
#  Full ResKAN+ Network
# -------------------------
class ResKANNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        num_blocks: int = 8,
        kan_K: int = 32,
        moe_experts: int = 3,
        gate_expand: float = 2.0,
        gate_kernel: int = 7,
        gate_grid: int = 8,
        cslc_rank: int = 1
    ):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResKANBlock(
                feat_ch=feat_ch,
                kernel_size=3,
                kan_K=kan_K,
                moe_experts=moe_experts,
                gate_expand=gate_expand,
                gate_kernel=gate_kernel,
                gate_grid=gate_grid,
                cslc_rank=cslc_rank,
            )
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
        kan_K: int = 32,
        moe_experts: int = 3,
        gate_expand: float = 2.0,
        gate_kernel: int = 7,
        gate_grid: int = 8,
        cslc_rank: int = 1,
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
            kan_K=kan_K, moe_experts=moe_experts,
            gate_expand=gate_expand, gate_kernel=gate_kernel,
            gate_grid=gate_grid, cslc_rank=cslc_rank
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
            # tính cả KAN đơn lẻ và KAN trong MoE
            if isinstance(m, KANCubic1D):
                pen = pen + m.sobolev_penalty()
            if isinstance(m, MoEKAN1D):
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
