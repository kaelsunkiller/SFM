# Model Card

## Model Details

SFM is a retinal foundation-model family for CKM screening. This release includes modular code for:

- Visual encoding (Swin backbone wrapper)
- Sparse MoE routing
- Optional image-text alignment
- Downstream heads for comorbidity, staging, biomarkers, progression, and ophthalmic transfer

## Intended Use

- Research on retinal-image-based CKM risk stratification
- External validation on independent cohorts under approved governance
- Method benchmarking against ophthalmic foundation models

## Out-of-Scope Use

- Standalone diagnostic decisions without clinical oversight
- Use on unsupported imaging protocols without recalibration
- Use as a substitute for regulated clinical workflows

## Training Data Summary

Per manuscript, pretraining combined public ophthalmic datasets and controlled-access community retinal cohorts. This repository does not include controlled-access cohort data.

## Limitations

- Performance may degrade under strong domain shift.
- Calibration requires cohort-specific checks before deployment.
- Decision-thresholds are context-sensitive and not universal.

## Fairness Considerations

- Evaluate subgroup performance before deployment.
- Report confidence intervals for stratified analyses.
- Confirm that screening benefits are consistent across demographics and acquisition sites.
