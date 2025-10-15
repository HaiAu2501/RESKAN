import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.loader import Loader

def setup(cfg: DictConfig) -> tuple[Trainer, ModelCheckpoint]:
    wandb_logger = WandbLogger(project="RESKAN")
    ckpt_callback = ModelCheckpoint(monitor="val_psnr", mode="max", save_top_k=3)
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        gradient_clip_val=1.0,
    )
    return trainer, ckpt_callback

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    loader = Loader(dataset_cfg=cfg.dataset, cfg=cfg.loader)
    trainer, ckpt_callback = setup(cfg.trainer)

    model = instantiate(cfg.model)
    if os.getenv("CHECKPOINT_PATH"):
        ckpt_path = os.getenv("CHECKPOINT_PATH")
        model = type(model).load_from_checkpoint(ckpt_path)
        print(f"Fine-tuning from checkpoint: {ckpt_path}")
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_callback.best_model_path
    test_loader = loader.test_dataloader()
    trainer.test(model, test_loader, ckpt_path=best_ckpt)
  
if __name__ == "__main__":
    main()