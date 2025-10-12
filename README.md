# RESKAN

KAN Do It Better: Revisting Image Restoration from a Functional View

> Recent advances in image restoration are dominated by architectures that learn feature hierarchies, from convolutional to transformer-based designs. In contrast, we revisit restoration from a functional approximation viewpoint: our goal is to directly learn the restoration operator — a structured, interpretable function that maps degraded observations to clean reconstructions.

## Project Structure

You should structure your dataset folder as follows:

```
<dataset_root>/
    ├── train/
    │   ├── <ground_truth>/
    │   │   ├── img1.<ext>
    │   │   ├── img2.<ext>
    │   │   └── ...
    │   └── <low_quality>/
    │       ├── img1.<ext>
    │       ├── img2.<ext>
    │       └── ...
    └── test/
        ├── <ground_truth>/
        │   ├── img1.<ext>
        │   ├── img2.<ext>
        │   └── ...
        └── <low_quality>/
            ├── img1.<ext>
            ├── img2.<ext>
            └── ...
```

- `<dataset_root>`: The root directory of your dataset.
- `<ground_truth>`: Folder containing the ground truth images.
- `<low_quality>`: Folder containing the low-quality images.
- `<ext>`: The image file extension (e.g., jpg, png).
- Ensure that filenames in `<ground_truth>` and `<low_quality>` match for corresponding image pairs.

If the dataset you want to use is not already included in the `cfg/dataset` folder, create a new YAML configuration file in the `cfg/dataset` directory. Use the naming convention `<your_dataset_name>.yaml`.

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/HaiAu2501/RESKAN.git
cd RESKAN
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset following the structure mentioned above.

4. Run the main script with your dataset configuration:

```bash
python main.py \
       dataset=<your_dataset> \
       dataset.root=<dataset_root> \
       dataset.lq_dirname=<low_quality> \
       dataset.gt_dirname=<ground_truth> \
       dataset.patch_size=<patch_size> \
       loader.batch_size=<batch_size> \
```
