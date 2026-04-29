# Reproducibility Guide

This document describes how to regenerate the manuscript's Source Data
workbooks from the per-sample analysis artifacts described in Methods.

## Prerequisites

The per-sample artifacts (per-sample model probabilities, per-seed
metric tables, affinity matrices, Plot_v2 spreadsheets, revision-record
CSVs) are not redistributed with this repository. They are released to
qualified researchers under the SHEDPTC institutional data-use
agreement; see the README "CKM Data" section for the application path.

Once obtained, set the `SFM_ANALYSIS_ROOT` environment variable to the
directory that contains the artifact tree (the parent of
`data_for_figures/`, `Plot_v2/`, and `revision_records/`). The expected
sub-tree is documented in the module docstring of
`analysis/source_data/generate_source_data.py`.

```bash
export SFM_ANALYSIS_ROOT=/path/to/manuscript_artifacts
export SFM_OUTPUT_DIR=$(pwd)/Source_Data
```

## Reproduce all figures + run the validation pass

A single command rebuilds every Source Data workbook (12 main + ED) and
prints a `[PASS] / [FAIL]` table against the manuscript's headline
values:

```bash
python -m analysis.source_data.generate_source_data
```

Behaviour:

- Writes `Source_Data_Fig1.xlsx` ... `Source_Data_Fig6.xlsx` and
  `Source_Data_ED_Fig3.xlsx` ... `Source_Data_ED_Fig8.xlsx` into
  `$SFM_OUTPUT_DIR`.
- Validates every figure's headline value against the manuscript;
  any failed workbook is moved to `Source_Data/_invalid/`.
- Writes `Source_Data/RUN_REPORT.md` with timestamp, git hash, and
  per-sheet validation outcome.
- ED Fig 1 / 2 are pure schematics with no plotted data and therefore
  do not have Source Data files.

## Headline values validated by the run

| Sheet | Headline value | Tolerance |
|---|---|---|
| `Source_Data_Fig2.xlsx::b` | SFM-M2 macro AUROC = 0.822 | ±0.001 |
| `Source_Data_Fig3.xlsx::b` | β̄(1→2) = +15.1% | ±0.5 |
| `Source_Data_Fig3.xlsx::c` | SFM-M2 ARI = 0.954, NMI = 0.965 | ±0.001 |
| `Source_Data_Fig4.xlsx::b` | SFM-M2 internal macro AUROC = 0.883 | ±0.001 |
| `Source_Data_Fig4.xlsx::d` | SFM-M2 external Overall AUROC = 0.734 | ±0.001 |
| `Source_Data_Fig4.xlsx::e` | SFM-M2 DCA useful range [0.34, 0.68] | ±0.005 |
| `Source_Data_Fig5.xlsx::e` | SFM-M2 prognostic AUROC = 0.962 | ±0.001 |
| `Source_Data_Fig5.xlsx::e` | SFM-M2 image-only AUROC = 0.880 | ±0.001 |
| `Source_Data_Fig5.xlsx::g` | SFM-M2 C-index = 0.822 | ±0.001 |
| `Source_Data_Fig5.xlsx::h` | SFM-M2 base-case cost = $2,171 | exact |
| `Source_Data_Fig5.xlsx::h` | SFM-M2 5-year QALYs = 4.153 | exact |
| `Source_Data_Fig5.xlsx::i` | P(SFM-M2 dominates Trad.) = 90.7% | ±0.1 |

## Customise the run

Override path and output via environment variables (preferred) or via CLI flags:

```bash
# Env-var form
export SFM_ANALYSIS_ROOT=/different/artifact/path
export SFM_OUTPUT_DIR=/where/to/write
python -m analysis.source_data.generate_source_data

# CLI form
python -m analysis.source_data.generate_source_data \
  --data-root /different/artifact/path \
  --output-dir /where/to/write
```

To skip the headline-value validation pass (e.g., for partial re-runs
on synthetic data):

```bash
python -m analysis.source_data.generate_source_data --skip-validation
```

## DOI Placeholders

Release DOI placeholders appear in:

- `README.md`
- `CITATION.cff`

Replace placeholders after Zenodo mints the GitHub-release DOI.
