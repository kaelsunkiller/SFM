# Data Format

## Image Layout

Set `SFM_DATA_ROOT` to a root directory containing per-dataset image folders. Pretraining and downstream evaluation use distinct subsets:

```text
${SFM_DATA_ROOT}/
  # Pretraining-only corpora (no downstream train/val/test split)
  AIROGS/
    training/
      <image_id>.jpg
  EyePACS/
    train/
      <image>.jpeg
  # Downstream evaluation benchmarks (each with explicit splits in
  # configs/datasplits/<dataset>_{train,val,test}.txt)
  APTOS2019/
    train_images/
      <id>.png
  IDRiD/
    ...
  Messidor-2/
    ...
  PAPILA/
    FundusImages/
      <id>.jpg
  Glaucoma_fundus/
    <class>/
      <image>.jpg
  JSIEC/
    <class_name>/
      <image>.jpg
  ODIR-5K/
    Training/Images/
      <image>.jpg
  Retina/
    Train/
      <image>.png
```

## Label CSV Schema

The public loader expects a CSV with at least these columns:

- `image_path`: path relative to `SFM_DATA_ROOT`
- `label` or task-specific label columns

### CKM Comorbidity Example

```csv
image_path,CKD,Diabetes,Hypertension,Stroke,Obesity,Cardiopathy
cohort_a/img_0001.jpg,1,0,1,0,1,0
```

### CKM Staging Example

```csv
image_path,true_stage
staging/img_0001.jpg,2
```

### Biomarker Example

```csv
image_path,eGFR_class,HbA1c_class,TG_class
biomarker/img_0001.jpg,4,2,3
```

## Environment Variables

- `SFM_DATA_ROOT`: required image data root
- `SFM_ANALYSIS_ROOT`: optional manuscript analysis-artifact root used by `analysis/source_data/*` builders (see docs/reproducibility.md)
- `SFM_CKPT_DIR`: optional checkpoint path (default: `checkpoints`)
- `SFM_OUTPUT_DIR`: optional output path (default: `outputs`)
