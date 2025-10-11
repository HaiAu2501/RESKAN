import os, random

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(root: str, phase: str, gt_dir: str, lq_dir: str):
    lq_path = f"{root}/{phase}/{lq_dir}"
    gt_path = f"{root}/{phase}/{gt_dir}"
    
    lq_images = sorted([os.path.join(lq_path, x) for x in os.listdir(lq_path) if is_image_file(x)])
    gt_images = sorted([os.path.join(gt_path, x) for x in os.listdir(gt_path) if is_image_file(x)])
    
    return gt_images, lq_images

def random_crop(gt, lq, patch_size):
    _, h, w = lq.shape
    if h < patch_size or w < patch_size:
        raise ValueError("Patch size is larger than image dimensions.")

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    
    gt_crop = gt[:, top:top + patch_size, left:left + patch_size]
    lq_crop = lq[:, top:top + patch_size, left:left + patch_size]
    
    return gt_crop, lq_crop