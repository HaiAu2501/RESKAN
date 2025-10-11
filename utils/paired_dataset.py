import os, random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from utils.utils import get_image_paths, random_crop

class PairedImageDataset(Dataset):
    def __init__(self, cfg, phase='train', transform=None):
        self.root = cfg.root
        self.gt_dir = cfg.gt_dirname
        self.lq_dir = cfg.lq_dirname
        self.pathch_size = cfg.patch_size
        
        self.phase = phase
        self.transform = transform

        # Load image paths
        self.gt_images, self.lq_images = get_image_paths(self.root, self.phase, self.gt_dir, self.lq_dir)

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        gt_image = Image.open(self.gt_images[idx]).convert('RGB')
        lq_image = Image.open(self.lq_images[idx]).convert('RGB')

        gt = TF.to_tensor(gt_image)
        lq = TF.to_tensor(lq_image)

        if self.phase == 'train':
            gt, lq = random_crop(gt, lq, self.pathch_size)
            if random.random() < 0.5:
                gt = TF.hflip(gt)
                lq = TF.hflip(lq)

        return lq, gt # (x, y)
