from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class KANSpline1D(nn.Module):
    def __init__(self, channels: int, K: int = 16, init_identity: bool = True):
        super().__init__()
        self.channels = channels
        self.K = K

        centers = torch.linspace(-1.0, 1.0, K)
        delta = 2.0 / (K - 1)
        self.register_buffer("centers", centers[None, None, :, None, None])  # (1,1,K,1,1)
        self.register_buffer("delta", torch.tensor(delta))

        self.alpha = nn.Parameter(torch.zeros(channels, K))  # (C, K)
        self.beta = nn.Parameter(torch.ones(channels, 1))    # linear skip
        self.bias = nn.Parameter(torch.zeros(channels, 1))   # bias

        # Per-channel affine pre-normalization: t = clamp(s*(x - m), [-1.5, 1.5])
        self.shift = nn.Parameter(torch.zeros(channels, 1))
        self.scale = nn.Parameter(torch.ones(channels, 1))

        if init_identity:
            nn.init.zeros_(self.alpha)
            with torch.no_grad():
                self.beta.fill_(1.0)
                self.bias.zero_()
                self.shift.zero_()
                self.scale.fill_(1.0)

    @staticmethod
    def _hat(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - x.abs(), min=0.0)

    def forward(
        self,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected C={self.channels}, got {C}"

        t = (x - self.shift.view(1, C, 1, 1)) * self.scale.view(1, C, 1, 1)
        if film is not None:
            gamma, delta = film
            t = (1.0 + gamma) * t + delta
        t = torch.clamp(t, -1.5, 1.5)

        t_exp = t.unsqueeze(2)                               # (B,C,1,H,W)
        u = (t_exp - self.centers) / self.delta              # (B,C,K,H,W)
        basis = self._hat(u)

        phi = torch.einsum('bckhw,ck->bchw', basis, self.alpha)
        phi = phi + self.beta.view(1, C, 1, 1) * x + self.bias.view(1, C, 1, 1)
        return phi

    def sobolev_smoothness(self) -> torch.Tensor:
        if self.K < 3:
            return self.alpha.new_zeros(())
        d2 = self.alpha[:, :-2] - 2 * self.alpha[:, 1:-1] + self.alpha[:, 2:]
        return (d2.pow(2).mean())

class KANProxBlock(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, K: int = 16, use_cond: bool = False):
        super().__init__()
        self.use_cond = use_cond
        self.analysis = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True)
        self.kan = KANSpline1D(hidden, K=K, init_identity=True)
        self.synthesis = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True)

        self.gate = nn.Parameter(torch.zeros(1))  # scalar gate (sigmoid)

        if use_cond:
            self.cond_proj = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2 * hidden, kernel_size=1, bias=True)
            )
        else:
            self.cond_proj = None

        nn.init.kaiming_uniform_(self.analysis.weight, a=1.0)
        nn.init.kaiming_uniform_(self.synthesis.weight, a=1.0)
        nn.init.zeros_(self.analysis.bias)
        nn.init.zeros_(self.synthesis.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.analysis(x)

        film = None
        if self.use_cond and (cond is not None):
            cd = self.cond_proj(cond)                 # (B, 2*hidden, H, W)
            cd = F.adaptive_avg_pool2d(cd, 1)        # (B, 2*hidden, 1, 1)
            hidden = y.shape[1]
            gamma, delta = cd.split(hidden, dim=1)   # (B, hidden, 1, 1)
            film = (gamma, delta)

        z = self.kan(y, film=film)                   # (B, hidden, H, W)
        o = self.synthesis(z)                        # (B, C, H, W)

        g = torch.sigmoid(self.gate)                 # scalar in (0,1)
        out = x + g * (o - x)
        return out, self.kan.sobolev_smoothness()

class KANRestorationNet(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64, num_blocks: int = 6, K: int = 16, use_cond: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            KANProxBlock(in_channels, hidden=hidden, K=K, use_cond=use_cond) for _ in range(num_blocks)
        ])
        self.use_cond = use_cond
        self._last_spline_penalty: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        penalties = []
        h = x
        for blk in self.blocks:
            h, p = blk(h, cond if self.use_cond else None)
            penalties.append(p)
        self._last_spline_penalty = torch.stack(penalties).mean()
        return h

    def sobolev_regularizer(self, lam: float = 1e-3) -> torch.Tensor:
        p = getattr(self, "_last_spline_penalty", None)
        if p is None:
            ps = [blk.kan.sobolev_smoothness() for blk in self.blocks]
            p = torch.stack(ps).mean()
        return lam * p

class LitKANRestoration(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        num_blocks: int = 6,
        K: int = 16,
        use_cond: bool = False,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        lam_spline: float = 1e-3,
        data_range: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = KANRestorationNet(
            in_channels=in_channels,
            hidden=hidden,
            num_blocks=num_blocks,
            K=K,
            use_cond=use_cond
        )

        # Loss
        if loss_type.lower() == "l1":
            self.crit = nn.L1Loss()
        else:
            self.crit = nn.MSELoss()

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

        self.lam_spline = lam_spline
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cond = use_cond

    # ------------- utils -------------
    @staticmethod
    def _parse_batch(
        batch: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                y, x, cond = batch
            elif len(batch) == 2:
                y, x = batch
                cond = None
            else:
                raise ValueError("Expected batch as (y, x) or (y, x, cond)")
        elif isinstance(batch, dict):
            # Be generous with possible key names
            y = batch.get("y", None) or batch.get("input", None) or batch.get("degraded", None) or batch.get("noisy", None)
            x = batch.get("x", None) or batch.get("target", None) or batch.get("gt", None)
            cond = batch.get("cond", None)
            if y is None or x is None:
                raise ValueError("Dict batch must include y/input/degraded and x/target/gt")
        else:
            raise TypeError("Unsupported batch type")
        return y, x, cond

    # ------------- forward -------------
    def forward(self, y: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(y, cond=cond if self.use_cond else None)

    # ------------- steps -------------
    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        y, x, cond = self._parse_batch(batch)
        x_hat = self(y, cond)
        data_loss = self.crit(x_hat, x)
        reg = self.net.sobolev_regularizer(self.lam_spline)
        loss = data_loss + reg

        # Always log losses
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.size(0))
        self.log(f"{stage}/data_loss", data_loss, on_step=True, on_epoch=True, batch_size=y.size(0))
        self.log(f"{stage}/spline_reg", reg, on_step=True, on_epoch=True, batch_size=y.size(0))

        # Metrics: compute for val/test (train metrics can be noisy/slow)
        if stage in ("val", "test"):
            psnr = self.psnr(x_hat, x)
            ssim = self.ssim(x_hat, x)
            self.log(f"{stage}/psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
            self.log(f"{stage}/ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    # ------------- optim -------------
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Optional: cosine or step scheduler (comment in if desired)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.max_epochs)
        # return {"optimizer": optim, "lr_scheduler": sched}
        return optim

if __name__ == "__main__":
    B, C, H, W = 2, 3, 64, 64
    y = torch.randn(B, C, H, W)        # degraded / noisy
    x = torch.randn_like(y)            # ground-truth
    cond = torch.randn(B, C, H, W)     # optional conditioning (or None)

    model = LitKANRestoration(
        in_channels=C, hidden=64, num_blocks=4, K=16, use_cond=True,
        lr=2e-4, lam_spline=1e-3, data_range=1.0, loss_type="mse"
    )

    with torch.no_grad():
        x_hat = model(y, cond)
        print("Out shape:", x_hat.shape)
