import hydra
from omegaconf import DictConfig, OmegaConf
from utils.loader import Loader

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    loader = Loader(dataset_cfg=cfg.dataset, cfg=cfg.loader)
    train_loader = loader.train_dataloader()

    for x, y in train_loader:
        print(f"x: {x.shape}, y: {y.shape}")
        break
        
if __name__ == "__main__":
    main()