# Soft Label Pruning for Large-scale Dataset Distillation (<ins>LPLD</ins>)
[[`Paper`](https://arxiv.org/abs/2410.15919) | [`BibTex`](#citation) | [`Google Drive`](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing)]

---

Official Implementation for "[Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation?](https://arxiv.org/abs/2410.15919)", published at NeurIPS'24.

[Lingao Xiao](https://scholar.google.com/citations?user=MlNI5YYAAAAJ),&nbsp; [Yang He](https://scholar.google.com/citations?user=vvnFsIIAAAAJ)


> **Abstract**: In ImageNet-condensation, the storage for auxiliary soft labels exceeds that of the condensed dataset by over 30 times.
However, ***are large-scale soft labels necessary for large-scale dataset distillation***?
In this paper, we first discover that the high within-class similarity in condensed datasets necessitates the use of large-scale soft labels.
This high within-class similarity can be attributed to the fact that previous methods use samples from different classes to construct a single batch for batch normalization (BN) matching.
To reduce the within-class similarity, we introduce class-wise supervision during the image synthesizing process by batching the samples within classes, instead of across classes.
As a result, we can increase within-class diversity and reduce the size of required soft labels.
A key benefit of improved image diversity is that soft label compression can be achieved through simple random pruning, eliminating the need for complex rule-based strategies. Experiments validate our discoveries.
For example, when condensing ImageNet-1K to 200 images per class, our approach compresses the required soft labels from 113 GB to 2.8 GB (40x compression) with a 2.6% performance gain.


<div align=left>
<img style="width:100%" src="https://github.com/ArmandXiao/Public-Large-Files/blob/0ae81e632661c8507c8d377c0a14080439a1b25e/NeurIPS24_LPLD_animation.gif">
</div>
> Images from left to right are from IPC20 LPLD datasets: cock (left), bald eagle, volcano, trailer truck (right).

# Installation

Donwload repo:
```sh
git clone https://github.com/he-y/soft-label-pruning-for-dataset-distillation.git LPLD
cd LPLD
```

Create pytorch environment:
```sh
conda env create -f environment.yml
conda activate lpld
```

## Download all datasets and labels
### Method 1: Automatic Downloading
```sh
# sh download.sh [true|false]
sh download.sh false
```
- `true|false` meaning whether to download only 40x compressed labels or all labels. (default: false, download all labels)


### Method 2: Manual Downloading

Download manually from [Google Drive](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing), and place downloaded files in the following structure:
```
.
├── README.md
├── recover
│   └── model_with_class_bn
│       └── [put Models-with-Class-BN here]
│   └── validate_result
│       └── [put Distilled-Datast here]
├── relabel_and_validate
│   └── syn_label_LPLD
│       └── [put Labels here]
```
## You will find following after downloading

#### Model with Class BN 

|    Dataset    | Model with Class BN |                                                Size                                                |
| :-----------: | :-----------------: | :------------------------------------------------------------------------------------------------: |
|  ImageNet-1K  |      ResNet18       | [50.41 MB](https://drive.google.com/file/d/1Vfou8nPp3x7m7YEG0wd7FcuQ_yE9jj34/view?usp=drive_link)  |
| Tiny-ImageNet |      ResNet18       | [81.30 MB](https://drive.google.com/file/d/1sCArvJoHFthbSaBuWoUhDn67tsYtRPTn/view?usp=drive_link)  |
| ImageNet-21K  |      ResNet18       | [445.87 MB](https://drive.google.com/file/d/1BuplTqBhXKzdfJqCKkTBg218Cbezef57/view?usp=drive_link) |

#### Distilled Image Dataset

|    Dataset    |                   Setting                   |                                                                                                                                                                                                                                                   Dataset Size                                                                                                                                                                                                                                                   |
| :-----------: | :-----------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ImageNet-1K  | IPC10<br>IPC20<br>IPC50<br>IPC100<br>IPC200 | [0.15 GB](https://drive.google.com/file/d/1lXN_zi8LRq1pvrVZZgQA6W_ZU83BwvUf/view?usp=drive_link)<br>[0.30 GB](https://drive.google.com/file/d/18q2MZ5sr9AfNYqcd-NAa3-j3m7LRXKwN/view?usp=drive_link)<br>[0.75 GB](https://drive.google.com/file/d/1o081uXC-ebu28S_uuT04liACqAwhjA1O/view?usp=drive_link)<br>[1.49 GB](https://drive.google.com/file/d/18maJqCbuPXKT8zBHTLebbMbUgGwJIn4o/view?usp=drive_link)<br>[2.98 GB](https://drive.google.com/file/d/1-dLbdD3ww5wap4LpSjb1Ees4II4crv7p/view?usp=drive_link) |
| Tiny-ImageNet |               IPC50<br>IPC100               |                                                                                                                                                         [21 MB](https://drive.google.com/file/d/1W0JUOAZBrQwIlquIgOpdbFi5C_s8TNt8/view?usp=drive_link)<br>[40 MB](https://drive.google.com/file/d/1cQDD8OfMfoshsDIaiWOQb95pn2q9veuk/view?usp=drive_link)                                                                                                                                                         |
| ImageNet-21K  |               IPC10<br>IPC20                |                                                                                                                                                          [3 GB](https://drive.google.com/file/d/1DgmZNr1swgJrKZySjk1smgOiGr0mUU2R/view?usp=drive_link)<br>[5 GB](https://drive.google.com/file/d/1rycYU2q6JeUbGUDPBatr_QJQMJQSnGUk/view?usp=drive_link)                                                                                                                                                          |

#### Previous Soft Labels vs Ours

| Dataset       |                  Setting                  |                  Previous<br>Label Size                  |          Previous<br>Model Acc.           |                                                                                                                                                                                                                                                Ours<br>Label Size                                                                                                                                                                                                                                                |            Ours<br>Model Acc.             |
| :------------ | :---------------------------------------: | :------------------------------------------------------: | :---------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------: |
| ImageNet-1K   | IP10<br>IP20<br>IPC50<br>IPC100<br>IPC200 | 5.67 GB<br>11.33 GB<br>28.33 GB<br>56.66 GB<br>113.33 GB | 20.1%<br>33.6%<br>46.8%<br>52.8%<br>57.0% | [0.14 GB (40x)](https://drive.google.com/file/d/1Nf1piVIXIF-_v-jCEmaYGHdWTXsuQIkY/view?usp=drive_link)<br>[0.29 GB (40x)](https://drive.google.com/file/d/1AdP44DJUadFlY1WCrYiE7F6slotk3Vx4/view?usp=drive_link)<br>[0.71 GB (40x)](https://drive.google.com/file/d/1GnCY-Apg-dXgZe8BvDwDKqrQSAz1PAbs/view?usp=drive_link)<br>[1.43 GB (40x)](https://drive.google.com/file/d/12f6qUjsoN6AczK7iJz2ZAT8xNiX0W4bX/view?usp=drive_link)<br>[2.85 GB (40x)](https://drive.google.com/file/d/1mHWwOaB0yG7fP_lbDSZMmIHUrMh_nDWZ/view?usp=drive_link) | 20.2%<br>33.0%<br>46.7%<br>54.0%<br>59.6% |
| Tiny-ImageNet |              IPC50<br>IPC100              |                  449 MB<br>898 MB <br>                   |              41.1%<br>49.7%               |                                                                                                                                                       [11 MB (40x)](https://drive.google.com/file/d/1Yzgu-I96ODg2J8_AhGuNOP2mlUtbCzHU/view?usp=drive_link)<br>[22 MB (40x)](https://drive.google.com/file/d/1oJuUIq36raTtD63sfzT37ZJ3kGqZGqbv/view?usp=drive_link)<br>                                                                                                                                                       |              38.4%<br>46.1%               |
| ImageNet-21K  |              IPC10<br>IPC20               |                  643 GB<br>1286 GB<br>                   |              18.5%<br>20.5%               |                                                                                                                                                         [16 GB (40x)](https://drive.google.com/file/d/1inuNAC7ApJWiuXaCsEwWU9_z7DOpMBzG/view?usp=drive_link)<br>[32 GB (40x)](https://drive.google.com/file/d/1g52Lo2XoKHbJySkiLFo3Gsl6hnjffOEN/view?usp=drive_link)                                                                                                                                                         |              21.3%<br>29.4%               |
- full labels for ImageNet-21K are too large to upload; nevertheless, we provide the 40x pruned labels.
- labels for other compression ratios are provided in [google drive](https://drive.google.com/drive/folders/1LIKrlcydyowSkw2lRjgrzfULHYZWTNh7?usp=drive_link), or refer [README: Usage](./README_usage.md) to generate the labels.

## Necessary Modification for Pytorch
Modify PyTorch source code `torch.utils.data._utils.fetch._MapDatasetFetcher` to support multi-processing loading of soft label data and mix configurations.

```python
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            if hasattr(self.dataset, "G_VBSM") and self.dataset.G_VBSM:
                pass # G_VBSM: uses self-decoding in the training script
            elif hasattr(self.dataset, "use_batch") and self.dataset.use_batch:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config_by_batch_idx(possibly_batched_index[0])
            else:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            # NOTE: mix_index, mix_lam, mix_bbox can be None
            mix_index_cpu = mix_index.cpu() if mix_index is not None else None
            return self.collate_fn(data), mix_index_cpu, mix_lam, mix_bbox, soft_label.cpu()
        else:
            return self.collate_fn(data)
```

# Reproduce Results for 40x compression ratio

To reproduce the [[`Table`](#Previous-Soft-Labels-vs-Ours)] for 40x compression ratio, run the following code:

```sh
cd relabel_and_validate
bash scripts/reproduce/main_table_in1k.sh
bash scripts/reproduce/main_table_tiny.sh
bash scripts/reproduce/main_table_in21k.sh
```

NOTE: validation directory (`val_dir`) in config files (`relabel_and_validate/cfg/reproduce/CONFIG_FILE`) should be changed to correct path on your device.

# To Reproduce Results for other compression ratios

**Please refer to [README: Usage](./README_usage.md) for details, including three modules**.

## Table Results ([Google Drive](https://drive.google.com/drive/folders/1hw62Qi5N2Vuh1NLdXCAM3BXyNBzT0n1u?usp=drive_link))

| No.                                                                                                 | Content                                                                                 | Datasets      |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------- |
| Table 1                                                                                             | Dataset Analysis                                                                        | ImageNet-1K   |
| [Table 2](https://drive.google.com/drive/folders/17GEr8tbvKtNmNvxKZHEkB_KMr0LbK9o6?usp=drive_link)  | (a) SOTA Comparison<br>(b) Large Networks                                               | Tiny ImageNet |
| [Table 3](https://drive.google.com/drive/folders/1JGgDLB8vuNovTJ_fIN4peHvih4_aJ83c?usp=drive_link)  | SOTA Comparison                                                                         | ImageNet-1K   |
| [Table 4](https://drive.google.com/drive/folders/1Q37e8IXV30nISHRFTStfin33TZJ2yVFm?usp=drive_link)  | Ablation Study                                                                          | ImageNet-1K   |
| [Table 5](https://drive.google.com/drive/folders/1R34uovaGjB7vz-VrHcFhTuZs86VYcRJs?usp=drive_link)  | (a) Pruning Metrics<br>(b) Calibration                                                  | ImageNet-1K   |
| [Table 6](https://drive.google.com/drive/folders/1KsxslLvXK5enPAhNpBlIif7EDykH0yUX?usp=drive_link)  | (a) Large Pruning Ratio<br>(b) ResNet-50 Result<br>(c) Cross Architecture Result        | ImageNet-1K   |
| [Table 7](https://drive.google.com/drive/folders/1Cc_hwZYCKN9inzLKO8DsKOdwBGaY3cHu?usp=drive_link)  | SOTA Comparison                                                                         | ImageNet-21K  |
| [Table 8](https://drive.google.com/drive/folders/14zo1eKf_s3d1bwJ3iB5lcSlzEX1LYl5Q?usp=drive_link)  | Adaptation to Optimization-free Method (i.e., [RDED](https://arxiv.org/abs/2312.03526)) | ImageNet-1K   |
| [Table 9](https://drive.google.com/drive/folders/1Ycnk5dqs0P7AgmY1_gBb1E1z2o6b-zGY?usp=drive_link)  | Comparison to [G-VBSM](https://arxiv.org/abs/2311.17950)                                | ImageNet-1K   |
| **Appendix**                                                                                        |                                                                                         |               |
| Table 10-18                                                                                         | Configurations                                                                          | -             |
| [Table 19](https://drive.google.com/drive/folders/1O-3HLxKGOTDPyn-iXaEmQuBQwfWdzh6G?usp=drive_link) | Detailed Ablation                                                                       | ImageNet-1K   |
| [Table 20](https://drive.google.com/drive/folders/15TDuuiScIsjiYnkxAeYuGwssViW-jNWP?usp=drive_link) | Large IPCs (i.e., IPC300 and IPC400)                                                    | ImageNet-1K   |
| [Table 23](https://drive.google.com/drive/folders/1L-75-dVPBS2JD63DncMd3S3WlKqJgBV0?usp=drive_link) | Comparison to [FKD](https://github.com/szq0214/FKD/blob/main/FKD)                       | ImageNet-1K   |

## Related Repos
Our code is mainly related to the following papers and repos:
- [Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective](https://github.com/VILA-Lab/SRe2L)
- [ImageNet-21K Pretraining for the Masses](https://github.com/Alibaba-MIIL/ImageNet21K)


## Citation

```
@inproceedings{xiao2024lpld,
  title={Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation?},
  author={Lingao Xiao and Yang He},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
