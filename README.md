# Screening Foundation Model (SFM) for Cardio-Kidney-Metabolic Health

SFM is a retinal foundation-model pipeline for joint cardio-kidney-metabolic (CKM) screening from routine fundus photographs. The submission package includes the core model components (SwinViT encoder, sparse MoE routing, optional image-text alignment), downstream task heads, source-data preparation utilities, and the decision-analytic cost-effectiveness model used in the manuscript.

## Citation

```bibtex
@article{yang2026sfm_ckm,
  title   = {A Screening Foundation Model for Cardio-Kidney-Metabolic Health using Routine Retinal Photographs},
  author  = {Yang, Longzhen and Liu, Yihang and Zhang, Juzhao and Wen, Ying
             and Yang, Jiaxiong and Yang, Ziteng and Liu, Qi and Lu, Lina
             and Xu, Yi and Shi, Danli and He, Mingguang and Li, Chunhua
             and Zou, Haidong and Cheng, Ching Yu and He, Jide and Lin, Senlin
             and He, Lianghua},
  journal = {},
  year    = {2026},
  doi     = {10.xxxx/xxxxxxxx},
  note    = {Software release DOI to be minted at acceptance: 10.5281/zenodo.TBD}
}
```

## Installation

```bash
conda env create -f environment.yml
conda activate sfm
python -m pip install -e .
```

Validated on Linux with Python 3.10. The manuscript experiments were run on NVIDIA A6000 hardware.

## Data Preparation

Use the directory and label specifications in [docs/data_format.md](docs/data_format.md). Public datasets fall into two roles: pretraining-only (used for SSL and image-text alignment, not for downstream evaluation) and downstream-evaluation benchmarks (the 8 public ophthalmic benchmarks reported in Fig. 2a).

### Pretraining datasets

The pretraining corpus combines 76,355 SEDPTC community photographs (CKM-relevant; access path described in the "CKM Data" section below) with 136,255 public ophthalmic images from the two datasets below. Neither AIROGS nor EyePACS is used as a downstream evaluation benchmark in this work.

#### AIROGS — Artificial Intelligence for Robust Glaucoma Screening

- **Access**: Grand Challenge: <https://airogs.grand-challenge.org/data-and-challenge/>. The training set is downloadable after a free Grand Challenge account registration; the held-out test set is not used in this work.
- **Layout**: `AIROGS/training/<image_id>.jpg` plus `train_labels.csv` mapping each `image_id` to a referable / non-referable glaucoma label.
- **Split used in this work**: the official Grand Challenge training set is consumed as a flat, unlabeled corpus during DINO-style self-supervised pretraining. No train/val/test partition is required because pretraining does not use downstream-style supervision.
- **Citation**: de Vente *et al.*, "AIROGS: artificial intelligence for robust glaucoma screening challenge," *IEEE Trans. Med. Imaging* 2023.

#### EyePACS

- **Access**: Kaggle Diabetic Retinopathy Detection: <https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data>. Free Kaggle account and acceptance of the competition rules required.
- **Layout**: `EyePACS/train/<image>.jpeg` plus `trainLabels.csv` with columns `image, level` (0--4 DR severity).
- **Split used in this work**: the official Kaggle `train` set is consumed as a flat corpus during pretraining. No EyePACS-specific train/val/test split file is shipped because downstream DR evaluation is performed on APTOS-2019, IDRiD and MESSIDOR-2 instead.
- **Citation**: Gulshan *et al.*, "Development and validation of a deep learning algorithm for detection of diabetic retinopathy," *JAMA* 2016 (referenced in this work for the image source).

### Publich downstream evaluation datasets

These eight datasets are used for the ophthalmic-transfer experiments reported in Fig. 2a. Each has its own train/val/test partition shipped in `configs/datasplits/`.

#### APTOS-2019 — Asia Pacific Tele-Ophthalmology Society blindness detection

- **Access**: Kaggle: <https://www.kaggle.com/competitions/aptos2019-blindness-detection/data>. Free Kaggle account required.
- **Layout**: `APTOS2019/train_images/<id>.png` plus `train.csv` with `id_code, diagnosis` (0--4).
- **Split used in this work**: image-id partition shipped at `configs/datasplits/aptos_{train,val,test}.txt` (2,048 / 514 / 1,100 images).
- **Citation**: APTOS 2019 Blindness Detection Challenge (Kaggle), 2019.

#### IDRiD — Indian Diabetic Retinopathy Image Dataset

- **Access**: IEEE DataPort: <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>. Free IEEE DataPort account required.
- **Layout**: `IDRiD/B. Disease Grading/1. Original Images/{a. Training Set,b. Testing Set}/<image>.jpg` and grade CSVs.
- **Split used in this work**: the official IDRiD train/test partition with a 10% held-out validation slice from the training set. Listing: `configs/datasplits/idrid_{train,val,test}.txt` (301 / 82 / 101 images).
- **Citation**: Porwal *et al.*, "Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research," *Data* 2018.

#### MESSIDOR-2

- **Access**: ADCIS distribution: <https://www.adcis.net/en/third-party/messidor2/>. Free academic registration required.
- **Layout**: `Messidor-2/IMAGES/<image>.png` plus the public DR grade CSV.
- **Split used in this work**: image-id partition shipped at `configs/datasplits/messidor2_{train,val,test}.txt` (953 / 242 / 522 images).
- **Citation**: Decencière *et al.*, "Feedback on a publicly distributed image database: the Messidor database," *Image Anal Stereol* 2014.

#### PAPILA — Papilla and optic disc dataset

- **Access**: figshare: <https://figshare.com/articles/dataset/PAPILA/14798004>. Open access; no registration required.
- **Layout**: `PAPILA/FundusImages/<id>.jpg` plus `Diagnosis.xlsx` with glaucoma stage.
- **Split used in this work**: participant-id partition shipped at `configs/datasplits/papila_{train,val,test}.txt` (309 / 75 / 94 images).
- **Citation**: Kovalyk *et al.*, "PAPILA: dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment," *Sci Data* 2022.

#### Glaucoma fundus

- **Access**: Hosted as supplementary data alongside the original publication. Refer to the data-availability statement of Ahn *et al.*, *PLoS One* 2018 (DOI <https://doi.org/10.1371/journal.pone.0207982>) for the canonical download instructions.
- **Layout**: `Glaucoma_fundus/<class>/<image>.jpg` where class is one of normal / early / advanced.
- **Split used in this work**: image-id partition shipped at `configs/datasplits/glaucoma_{train,val,test}.txt` (846 / 218 / 465 images).
- **Citation**: Ahn *et al.*, "A deep learning model for the detection of both advanced and early glaucoma using fundus photography," *PLoS One* 2018.

#### JSIEC — Joint Shantou International Eye Centre 39-class fundus

- **Access**: Kaggle: <https://www.kaggle.com/datasets/linchundan/fundusimage1000>. Free Kaggle account required.
- **Layout**: `JSIEC/<class_name>/<image>.jpg` with 39 native disease classes.
- **Split used in this work**: image-id partition shipped at `configs/datasplits/jsiec_{train,val,test}.txt` (531 / 150 / 301 images). The 39 native disease classes are used as-is for 39-way classification, matching the original JSIEC paper's task framing; `configs/datasplits/jsiec_class_map.csv` documents the canonical native-class to integer-index assignment used by the data loader (no further harmonisation is applied).
- **Citation**: Cen *et al.*, "Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks," *Nat Commun* 2021.

#### ODIR-5K — Ocular Disease Intelligent Recognition

- **Access**: Grand Challenge: <https://odir2019.grand-challenge.org/dataset/>. Free Grand Challenge account.
- **Layout**: `ODIR-5K/Training/Images/<image>.jpg` plus `data.xlsx` listing eight ocular disease tags per sample.
- **Split used in this work**: official ODIR train/val/test partition shipped at `configs/datasplits/odir_{train,val,test}.txt` (4,900 / 692 / 1,408 images); multi-label evaluation with the eight original tags (one head per tag).
- **Citation**: Xu *et al.* and the Peking University ODIR organisers (2019); we cite Herrera *et al.* per the manuscript.

#### Retina (multi-disease retinal benchmark)

- **Access**: RFMiD via the RIADD challenge: <https://riadd.grand-challenge.org/RIADD/>. Free Grand Challenge account required.
- **Layout**: `Retina/<split>/<class>/<image>.png` where `<class>` is one of `anormal` (normal), `bcataract` (cataract), `cglaucoma` (glaucoma), or `ddretina_disease` (other retinal disease).
- **Split used in this work**: a curated 4-class subset of RFMiD covering normal vs three multi-disease buckets (cataract, glaucoma, other retinal disease), with image-id partition shipped at `configs/datasplits/rfmid_{train,val,test}.txt` (336 / 84 / 181 images, 601 total). This 4-class framing matches the manuscript Methods §Ophthalmic disease diagnosis "multi-disease classification" task framing for the Retina benchmark; it is not the official 28-tag RIADD multi-label challenge.
- **Citation**: Pachade *et al.*, "Retinal Fundus Multi-Disease Image Dataset (RFMiD): a dataset for multi-disease detection research," *Data* 2021.

### Notes for all public datasets

- Verify each dataset's terms of use before downloading. Most require an accepted user agreement (Kaggle competition terms, ADCIS academic licence, Mendeley CC licence variants).
- The shipped `configs/datasplits/*.txt` files contain the explicit image identifier listings used in this work for the eight downstream-evaluation benchmarks. Each line carries one identifier with an optional whitespace-separated label. No pixel data is redistributed.
- The data loader expects images at native resolution; downsampling to 1024 × 1024 RGB happens inside the augmentation pipeline.

### CKM Data

The CKM cohorts used in the manuscript are not redistributed with this repository.

1. Refer to the manuscript Data Availability statement for the access procedure.
2. Access to the full SEDPTC community cohort, linked CKM labels, and longitudinal follow-up records requires an approved institutional data-use agreement.
3. UK Biobank retinal photographs and linked health records require an independent UK Biobank application.
4. External validation cohorts were used under separate institutional agreements and require independent application.
5. A public Zenodo demonstration set of **de-identified retinal photographs** is provided for minimum validation; it is not a substitute for the full cohorts.

## Reproducing Figures

See [docs/reproducibility.md](docs/reproducibility.md). Each figure subsection provides a command, expected output artifact, and expected sanity-check value.

## Model Weights

Pretrained SFM weights for non-commercial research are released under a separate model-use agreement. Contact the corresponding author email below for access instructions.

## License

Code is released under Apache-2.0. Model weights are distributed under a separate non-commercial research agreement.

## Acknowledgments

Funding and institutional support follow the manuscript Acknowledgments section.

## Contact

For data, code and model questions: `yanglongzhen@tongji.edu.cn` and `helianghua@tongji.edu.cn.`.
