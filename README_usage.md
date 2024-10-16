## Module 1: Squeeze with Class BN ([Google Drive](https://drive.google.com/drive/folders/1a1cT5uq0LZZuf4aGn3AgMR5EMopyvKHj?usp=drive_link))

To obtain network (i.e., ResNet) with class-wise batch normalization statistics, a modified ResNet is being forwarded for one epoch.
- modified ResNet: it is exactly same as ResNet but with `Normal BatchNorm2d` replaced by `ClassAwareBatchNorm2d`. It additionally tracks per-class BN statistics and does not have any influence on the performance.
- modified ResNet is at `models/resnet_class.py`

To obtain model with class BN, run the following scripts or download from [google drive](https://drive.google.com/drive/folders/1a1cT5uq0LZZuf4aGn3AgMR5EMopyvKHj?usp=drive_link):
```sh   
cd recover
bash scripts/imagenet1k_forward.sh
bash scripts/tiny_forward.sh
```
- Training model on ImageNet-21K is based on [this repo](https://github.com/Alibaba-MIIL/ImageNet21K).




## Module 2: Recover with Class BN ([Google Drive](https://drive.google.com/drive/folders/1JELI-Sbmob4WjW8a52xxVuOYtWdDmQaq?usp=drive_link))
To recover LPLD-distilled images, run the following script or download from [google drive](https://drive.google.com/drive/folders/1JELI-Sbmob4WjW8a52xxVuOYtWdDmQaq?usp=drive_link):
```sh
cd recover
bash scripts/imagenet1k_recover.sh
bash scripts/tiny_recover.sh
bash scripts/in21k_recover.sh
```
- This training script requires the class-wise BN stats from [Module 1](#module-1-squeeze-with-class-bn-google-drive).

## Module 3: Relabel, Prune, and Validate ([Google Drive](https://drive.google.com/drive/folders/1LIKrlcydyowSkw2lRjgrzfULHYZWTNh7?usp=drive_link))

### Module 3.1: Relabel and Prune
Basic Usage:
```sh
cd relabel_and_validate
python generate_soft_label_pruning.py \
    --cfg_yaml [config file]  \
    --train_dir [image path] \
    --fkd_path [label path to save] \
    --pruning_ratio [optional: label pruning ratio]
```
- the `[config file]` should contain information about **batch size, augmentation, and etc.**
    - the `val_dir` in `[config file]` should be replaced with the your own path to the dataset test set.
- if `pruning_ratio` is put, full labels will **NOT** be generated, and this **speeds up label generation process**.
    - since we perform random pruning, it is equivalent to only generate a few labels.
- labels are provided in [google drive](#download-labels)

Script files are provided and integrated in [validation](#module-32-validate).

### Module 3.2: Validate

Basic Usage:
```sh
python train_FKD_label_pruning_batch.py \
    --cfg_yaml [config file] \
    --model [model] \
    --prune_ratio [pruning ratio] \
    --granularity [epoch/batch] \
    --prune_metric [order/random] \
    --train_dir [image path] \
    --fkd_path [label path to load]
```
- validation should share the same `[config file], [image path], and [label path]` with [relabel](#module-31-relabel-and-prune).
- `granularity` is where to prune with epochs or batch (improved label pool).
- `prune_metric` is where to prune in a random order.

For convenience, run the following scripts to reproduce results of the main table in the paper:
```sh
cd relabel_and_validate
bash scripts/reproduce/main_table_in1k.sh
bash scripts/reproduce/main_table_tiny.sh
bash scripts/reproduce/main_table_in21k.sh
```

In our paper's implementation, we sample labels from full labels to adpat different pruning ratios.
For a specific pruning ratio, you do not need to generate full labels:
```sh
cd relabel_and_validate
bash scripts/reproduce/less_label_in1k.sh
bash scripts/reproduce/less_label_tiny.sh
bash scripts/reproduce/less_label_in21k.sh
```

