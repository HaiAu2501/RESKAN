import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train()