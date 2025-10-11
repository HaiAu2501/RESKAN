from torch.utils.data import DataLoader, random_split
from utils.paired_dataset import PairedImageDataset

class Loader:
    def __init__(self, dataset_cfg, cfg):
        self.dataset_cfg = dataset_cfg
        
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.shuffle = cfg.shuffle
        self.val_ratio = cfg.val_ratio

        self.ds_full = PairedImageDataset(self.dataset_cfg, phase='train')
        self.ds_test = PairedImageDataset(self.dataset_cfg, phase='test')

        val_size = int(len(self.ds_full) * self.val_ratio)
        train_size = len(self.ds_full) - val_size
        self.ds_train, self.ds_val = random_split(self.ds_full, [train_size, val_size])
        
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=1, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=1, shuffle=False, num_workers=1)
