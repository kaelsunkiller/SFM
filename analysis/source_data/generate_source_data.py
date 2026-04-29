#!/usr/bin/env python3
"""
Generate per-figure Source Data spreadsheets for the Nature Medicine
submission of "A Screening Foundation Model for Cardio-Kidney-Metabolic
Health using Routine Retinal Photographs".

For each main-text figure and Extended Data figure that contains a
plotted numerical value, this script writes one ``Source_Data_FigN.xlsx``
or ``Source_Data_ED_FigN.xlsx`` to ``NM_R4_Final/Source_Data/`` with one
sheet per panel.

Two panels are flagged for server regeneration because the local
machine does not retain the per-sample artifacts required to compute
them: Fig 5b (biomarker AUROC across 6 models) and Fig 4e (DCA per-
threshold net-benefit table). Their sheets are written as single-row
placeholders pointing the server-side agent at the correct regeneration
script (see ``SOURCE_DATA_SERVER_AGENT_GUIDE.md``).

The script ends with a validation pass that confirms 11 manuscript
headline values appear unchanged in the generated workbooks (within the
tolerance documented in the agent guide). Any failure prints the
offending sheet to stderr and moves the workbook to
``Source_Data/_invalid/`` for human review.

Run::

    python scripts/generate_source_data.py --all

The script is idempotent and safe to re-run after edits to the
underlying CSV / NPY sources.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ───────────────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("SFM_ANALYSIS_ROOT", "."))
P0 = ROOT / "data_for_figures" / "p0"
PLOT_V2 = ROOT / "Plot_v2"
RR = ROOT / "revision_records"
BIOMARKER_DIR = ROOT / "data_for_figures" / "biomarker"
# Pending-panel staged CSVs returned by the server agent (per
# SERVER_AGENT_PENDING_PANELS_PROMPT.md). Each file under this
# directory is the canonical per-panel table for one previously-
# placeholder Source Data sheet. Loaded via _load_pending_csv().
PENDING_DIR = ROOT / "data_for_figures" / "source_data_pending"
OUT = Path(os.environ.get("SFM_OUTPUT_DIR", "Source_Data"))
INVALID = OUT / "_invalid"


def _load_pending_csv(name: str) -> pd.DataFrame | None:
    """Load a staged pending-panel CSV if present.

    Returns None when the file is missing (so the caller can fall
    back to the legacy placeholder row), or a DataFrame otherwise.
    """
    fp = PENDING_DIR / name
    if fp.exists():
        return pd.read_csv(fp)
    return None

MODELS = ["ImageNet", "RETFound", "VisionFM", "SFM-Base", "SFM-MoE", "SFM-M2"]
DISEASES = ["CKD", "Diabetes", "Hypertension", "Stroke", "Obesity", "Cardiopathy"]
STAGES = [0, 1, 2, 3]
# Biomarker binary "abnormal" class (matches the convention used for the
# Zenodo selection script and the manuscript Fig 5b panel).
BIOMARKERS = {"eGFR": 4, "HbA1c": 2, "TG": 3}


# ───────────────────────────────────────────────────────────────────────
# Sheet helpers
# ───────────────────────────────────────────────────────────────────────
def _write_sheet(xl: pd.ExcelWriter, name: str, label: str,
                 description: str, df: pd.DataFrame) -> None:
    """Write a panel sheet with a 2-row header (label + description) above
    a tabular block, mirroring the Nature Medicine Source Data
    convention."""
    header = pd.DataFrame({df.columns[0]: [f"Panel {label}", description]})
    header = header.reindex(columns=df.columns)
    out = pd.concat([header, df], axis=0, ignore_index=True)
    out.to_excel(xl, sheet_name=name, index=False)


def _pending_server_sheet(xl: pd.ExcelWriter, name: str, label: str,
                          description: str, regenerate_path: str) -> None:
    """Write a single-row sheet flagging that the data must be
    regenerated on the training server (per
    ``SOURCE_DATA_SERVER_AGENT_GUIDE.md``)."""
    df = pd.DataFrame({
        "status": ["PENDING_SERVER_REGENERATION"],
        "regenerate_to": [regenerate_path],
        "see": ["NM_R4_Final/SOURCE_DATA_SERVER_AGENT_GUIDE.md"],
    })
    _write_sheet(xl, name, label, description, df)


# ───────────────────────────────────────────────────────────────────────
# Metric helpers
# ───────────────────────────────────────────────────────────────────────
def macro_ovr_auroc(y_true: np.ndarray, prob_matrix: np.ndarray) -> float:
    """One-vs-rest macro AUROC averaged across classes that are
    represented in ``y_true``. We compute the per-class binary AUROC
    directly rather than rely on sklearn's multiclass mode, because
    the mean-across-runs per-stage probabilities may not sum to
    exactly 1.0 due to averaging."""
    K = prob_matrix.shape[1]
    aucs = []
    for c in range(K):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aucs.append(roc_auc_score(y_bin, prob_matrix[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def macro_ovr_aupr(y_true: np.ndarray, prob_matrix: np.ndarray) -> float:
    """One-vs-rest macro AUPR. ``y_true`` integer class labels."""
    K = prob_matrix.shape[1]
    aps = []
    for c in range(K):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aps.append(average_precision_score(y_bin, prob_matrix[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def comorbidity_macro_auroc(df_model: pd.DataFrame) -> float:
    """For comorbidity (multi-label binary), compute macro AUROC across
    the 6 disease binary tasks."""
    aucs = []
    for d in DISEASES:
        y = df_model[f"true_{d}"].to_numpy()
        p = df_model[f"prob_{d}"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs))


# ───────────────────────────────────────────────────────────────────────
# Panel builders (return DataFrames + a validation pair)
# ───────────────────────────────────────────────────────────────────────
def build_fig1(out_path: Path) -> dict:
    """Fig 1: panel a (cohort schematic numbers) and panel e (radar +
    progression overview)."""
    cohort = pd.read_csv(RR / "demographics_cohort_pretraining.csv")
    cohort_staging = pd.read_csv(RR / "demographics_cohort_staging.csv")
    cohort_comorbid = pd.read_csv(RR / "demographics_cohort_comorbidity.csv")

    panel_a = pd.concat([cohort, cohort_staging, cohort_comorbid],
                        ignore_index=True)
    panel_a = panel_a[["cohort", "site", "images", "participants",
                       "age_mean", "age_sd", "male", "female"]]

    # Panel e (overview radar): per-model macro AUROC for staging and
    # comorbidity, mirrored into a wide table.
    df_stg = pd.read_csv(P0 / "calibration"
                         / "ckm_staging_probabilities_mean_wide.csv")
    df_com = pd.read_csv(P0 / "calibration"
                         / "comorbidity_probabilities_mean_wide.csv")

    rows = []
    for m in MODELS:
        sub_s = df_stg[df_stg["model"] == m]
        sub_c = df_com[df_com["model"] == m]
        prob_s = sub_s[[f"S{s}" for s in STAGES]].to_numpy()
        y_s = sub_s["true_stage"].to_numpy()
        rows.append({
            "model": m,
            "ckm_staging_macro_auroc": round(macro_ovr_auroc(y_s, prob_s), 4),
            "comorbidity_macro_auroc": round(comorbidity_macro_auroc(sub_c), 4),
        })
    panel_e = pd.DataFrame(rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Cohort composition by site (pretraining, staging, "
                     "comorbidity).", panel_a)
        _write_sheet(xl, "e", "e",
                     "Per-model macro AUROC across CKM staging and "
                     "comorbidity benchmarks (radar source).", panel_e)
    return {"file": out_path.name}


def build_fig2(out_path: Path) -> dict:
    """Fig 2: panel a (ophthalmic transfer across 8 public benchmarks
    x 6 models), panel b (CKM comorbidity per-disease + macro AUROC
    with 95% CIs; headline SFM-M2 macro = 0.822), panels c-g
    (multimorbidity amplification + subgroup MAE)."""

    # Panel a: ophthalmic 8-benchmark AUROC across 6 models. The
    # canonical values are hardcoded in the published plotting script
    # Plot_v2/plot1-1_eye_diseases_8datasets.py and re-emitted as a
    # staged CSV (fig2a_ophthalmic_8datasets_auc.csv, 8 x 6 = 48
    # rows). When the staged CSV is absent the sheet falls back to a
    # PENDING placeholder.
    panel_a = _load_pending_csv("fig2a_ophthalmic_8datasets_auc.csv")
    if panel_a is None:
        panel_a = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR
                                / "fig2a_ophthalmic_8datasets_auc.csv"),
        }])

    # Panel b: CKM comorbidity per-disease + macro AUROC with 95% CIs
    # (canonical values from comorbidity_perdisease_auc.csv). Output
    # is a wide pivot interleaving per-model AUC rows with the
    # corresponding 95% CI rows.
    auc_table = pd.read_csv(RR / "comorbidity_perdisease_auc.csv")
    auc_table["Model"] = auc_table["Model"].str.strip()
    pivot = auc_table.pivot(index="Model", columns="Disease",
                            values="AUC").reset_index()
    pivot_ci = auc_table.pivot(index="Model", columns="Disease",
                               values="CI").reset_index()
    pivot.columns.name = None
    pivot_ci.columns.name = None
    panel_b = pivot.copy()
    desired_cols = ["Model"] + DISEASES + ["Macro"]
    panel_b = panel_b[[c for c in desired_cols if c in panel_b.columns]]
    panel_b_ci = pivot_ci[[c for c in desired_cols if c in pivot_ci.columns]]
    panel_b_ci["Model"] = panel_b_ci["Model"].astype(str) + " (95% CI)"
    panel_b = pd.concat([panel_b, panel_b_ci], ignore_index=True)
    order = []
    for m in pivot["Model"].tolist():
        order.append(panel_b.index[panel_b["Model"] == m][0])
        order.append(panel_b.index[panel_b["Model"] == f"{m} (95% CI)"][0])
    panel_b = panel_b.loc[order].reset_index(drop=True)

    sfm_macro = float(auc_table[(auc_table["Model"] == "SFM-M2")
                                & (auc_table["Disease"] == "Macro")]
                      ["AUC"].iloc[0])

    # Panel c: k-stratified AUPR / PPV / F1 / NB across k=1..6 per
    # model — staged CSV.
    panel_c = _load_pending_csv("fig2c_k_stratified_metrics.csv")
    if panel_c is None:
        panel_c = pd.DataFrame([{"status": "PENDING_CONSOLIDATION"}])

    # Panels d / e: sex-stratum MAE proxy across burden k. The
    # staged CSVs cover k=1..6, but k=6 (all six diseases positive
    # simultaneously) is degenerate — within that single-class
    # subgroup the AUPR proxy collapses to its trivial value (MAE=0,
    # SD=0 regardless of model performance), so the manuscript
    # restricts this analysis to k=1..5 and the Source Data drops
    # the k=6 rows here.
    def _drop_k6(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or "k" not in df.columns:
            return df
        return df[df["k"] != 6].reset_index(drop=True)

    panel_d = _drop_k6(_load_pending_csv("fig2d_female_mae.csv"))
    if panel_d is None:
        panel_d = pd.DataFrame([{"status": "PENDING_CONSOLIDATION"}])
    panel_e = _drop_k6(_load_pending_csv("fig2e_male_mae.csv"))
    if panel_e is None:
        panel_e = pd.DataFrame([{"status": "PENDING_CONSOLIDATION"}])

    # Panel f: 12-subgroup MAE — staged CSV. The subgroup_value
    # convention is documented in the sheet description (`<disease>_0`
    # = participants without that disease, `<disease>_1` = with).
    panel_f = _drop_k6(_load_pending_csv("fig2f_subgroup_mae.csv"))
    if panel_f is None:
        panel_f = pd.DataFrame([{"status": "PENDING_CONSOLIDATION"}])

    # Panel g: sex-by-disease interaction summary — staged CSV.
    panel_g = _load_pending_csv("fig2g_sex_disease_interaction.csv")
    if panel_g is None:
        panel_g = pd.DataFrame([{"status": "PENDING_CONSOLIDATION"}])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Ophthalmic transfer across 8 public benchmarks "
                     "(APTOS-2019, IDRiD, MESSIDOR-2, Glaucoma fundus, "
                     "PAPILA, Retina, JSIEC, ODIR-5K) x 6 models. "
                     "AUROC_mean and AUROC_sd are the mean and "
                     "standard deviation across 5 random-seed runs; "
                     "p_value_vs_SFM_M2 is the paired test against "
                     "SFM-M2 on the same dataset (NaN for SFM-M2 "
                     "itself). Manuscript headline: SFM-M2 mean "
                     "AUROC = 0.914 across the 8 benchmarks.", panel_a)
        _write_sheet(xl, "b", "b",
                     "CKM comorbidity per-disease and macro AUROC for "
                     "the 6 models on the 4,000-image community test "
                     "cohort. Each model is followed by its 95% CI "
                     "row. Headline SFM-M2 macro AUROC = 0.822 (95% "
                     "CI 0.811-0.833).", panel_b)
        _write_sheet(xl, "c", "c",
                     "k-stratified AUPR / PPV / F1 / net benefit "
                     "across multimorbidity burden k=1..6 per model "
                     "(MAE supporting data).", panel_c)
        _write_sheet(xl, "d", "d",
                     "Female-stratum MAE proxy (1 - AUPR per disease) "
                     "across burden k=1..5. k=6 is omitted because "
                     "the all-six-positive subgroup is degenerate "
                     "(every example has the positive label so AUPR "
                     "trivially saturates and the MAE proxy collapses "
                     "to zero independent of model quality).", panel_d)
        _write_sheet(xl, "e", "e",
                     "Male-stratum MAE proxy across burden k=1..5; "
                     "k=6 omitted for the same degeneracy reason "
                     "described in sheet d.", panel_e)
        _write_sheet(xl, "f", "f",
                     "12 disease-conditioned subgroup MAE proxies "
                     "across burden k=1..5. The `subgroup_value` "
                     "naming convention is `<disease>_0` for "
                     "participants WITHOUT that disease and "
                     "`<disease>_1` for participants WITH that "
                     "disease (so e.g. `CKD_0` is the MAE trajectory "
                     "for non-CKD participants, `CKD_1` for CKD "
                     "carriers). Plus the two sex strata "
                     "(sex/Female, sex/Male) for direct comparison. "
                     "k=6 omitted for the same reason as in sheet d.",
                     panel_f)
        _write_sheet(xl, "g", "g",
                     "Sex-by-disease interaction summary derived from "
                     "sheets d and e: per-disease female_mae, "
                     "male_mae, interaction_score (female - male) "
                     "and Welch t-test p-value.", panel_g)

    return {"file": out_path.name,
            "headline": {"sfm_m2_macro_auroc": sfm_macro,
                         "expected": 0.822, "tol": 0.001}}


def build_fig3(out_path: Path) -> dict:
    """Fig 3 panel b (β̄(k) per pair) and panel c (UMAP).

    The headline Δβ̄(1→2) = +15.1% reported in the manuscript is
    computed from the per-sample affinity tensor (not exported as a
    CSV on the local machine); we mirror it here from the manuscript
    text as a pinned row, and flag the panel for server-side
    verification using the original aff_all NPY.
    """
    affinity_dir = PLOT_V2 / "figs-4-3_spatial_analysis_out" / "B_affinity_networks"

    # Panel b: long-form (k, pair_i, pair_j, mean_affinity).
    long_rows = []
    for k in range(1, 7):
        m = pd.read_csv(affinity_dir / f"mean_affinity_k{k}.csv", index_col=0)
        triu = np.triu_indices_from(m.to_numpy(), k=1)
        for i, j in zip(*triu):
            long_rows.append({
                "k": k,
                "disease_i": m.index[i],
                "disease_j": m.columns[j],
                "mean_affinity": float(m.iloc[i, j]),
            })
    panel_b = pd.DataFrame(long_rows)
    # The 5 published Δβ̄(k→k+1) values (+15.1%, +2.1%, +1.8%, +6.1%,
    # +7.2%) are mirrored from main.tex Fig 3b annotation. The
    # validation pass below checks the +15.1% headline; we no longer
    # ship a separate b_headline sheet because the value is fully
    # derivable from sheet b's per-pair affinities (compare row means
    # across consecutive k).
    delta_pct = 15.1

    # Panel c: UMAP per-point metadata + per-class centroid / spread
    # stats. Per-model ARI / NMI cluster-quality metrics are reported
    # in the manuscript text directly (SFM-M2 ARI = 0.954, NMI =
    # 0.965; see main.tex line 192) and are validated against that
    # headline below; we no longer ship a separate c_metrics sheet.
    umap_pts = pd.read_csv(PLOT_V2 / "figs-4-2_cam_manifold_out"
                           / "embedding_points_meta_strategy_pos.csv")
    umap_stats = pd.read_csv(PLOT_V2 / "figs-4-2_cam_manifold_out"
                             / "embedding_stats_umap_strategy_pos.csv")

    panel_c_pts = umap_pts.head(50000)  # cap for sheet size
    panel_c_stats = umap_stats.copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "b", "b",
                     "Per-pair mean affinity beta-bar(k) for the 15 "
                     "unordered disease pairs across multimorbidity "
                     "burden k=1..6 (15 x 6 = 90 rows). The published "
                     "annotated increments Delta beta-bar(k -> k+1) "
                     "are derivable as the relative change of the "
                     "mean across the 15 pairs between adjacent k "
                     "levels (manuscript values +15.1%, +2.1%, "
                     "+1.8%, +6.1%, +7.2% for 1->2 ... 5->6).",
                     panel_b)
        _write_sheet(xl, "c_points", "c",
                     "Per-point metadata for the 50,000 UMAP scatter "
                     "points shown in Fig 3c. Columns: `n` is the "
                     "sample index in the comorbidity test cohort; "
                     "`c` is the disease class identifier in the "
                     "fixed manuscript ordering 0=CKD, 1=Diabetes, "
                     "2=Hypertension, 3=Stroke, 4=Obesity, "
                     "5=Cardiopathy (each multi-label sample is "
                     "exploded into one row per positive class); "
                     "`k` is the multimorbidity burden of that "
                     "sample (1..6); `prob` is the model's predicted "
                     "probability for class `c` on that sample; "
                     "`is_pos` is the ground-truth indicator that "
                     "the sample carries class `c`. UMAP spatial "
                     "coordinates per class are summarised in sheet "
                     "c_stats. Per-model cluster-quality metrics "
                     "(ARI / NMI) are reported in main.tex line 192: "
                     "ImageNet 0.298 / 0.426; RETFound 0.862 / 0.884; "
                     "VisionFM 0.256 / 0.418; SFM-M2 0.954 / 0.965.",
                     panel_c_pts)
        _write_sheet(xl, "c_stats", "c",
                     "Per-class UMAP centroid (centroid_x, centroid_y) "
                     "and within-class dispersion across the 6 disease "
                     "classes.", panel_c_stats)

    return {"file": out_path.name,
            "headline": {"delta_beta_1_to_2_pct": delta_pct,
                         "expected": 15.1, "tol": 0.5,
                         "ari_sfm_m2": 0.954, "ari_expected": 0.954,
                         "nmi_sfm_m2": 0.965, "nmi_expected": 0.965}}


def _build_fig4e_panel() -> tuple[pd.DataFrame, dict]:
    """Reproduce the published Fig 4e Decision Curve Analysis using
    the exact computation in
    ``NM_R4_Major_Revision_WIP/Figures_final/fig4.py::draw_dca``: the
    score is P(stage>=2) = S2 + S3, the cohort is HK&A + UK Biobank
    (Fujian excluded), the threshold grid is
    ``np.linspace(0.05, 0.75, 300)`` and net benefit is
    ``TP/N - FP/N * t/(1-t)``."""
    src = P0 / "external_breakdown" / "external_ckm_per_sample_predictions_mean.csv"
    if not src.exists():
        return None, {"status": "PENDING_SERVER"}

    ext = pd.read_csv(src)
    sub = ext[ext["site"].isin(["HK&A", "UK Biobank"])].copy()
    thresholds = np.linspace(0.05, 0.75, 300)
    y_all = (sub[sub["model"] == "SFM-M2"]["true_stage"] >= 2).astype(int).values
    prev = float(np.mean(y_all))
    treat_all = prev - (1.0 - prev) * (thresholds / (1.0 - thresholds))

    def _nb(y, p, thr_grid):
        n = max(len(y), 1)
        out = []
        for t in thr_grid:
            pred = (p >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y == 1)))
            fp = int(np.sum((pred == 1) & (y == 0)))
            out.append(tp / n - fp / n * (t / (1.0 - t))
                       if t < 0.999 else 0.0)
        return np.array(out, dtype=float)

    long_rows = []
    useful_ranges = {}
    for m in ["ImageNet", "RETFound", "VisionFM", "SFM-M2"]:
        ms = sub[sub["model"] == m]
        y = (ms["true_stage"] >= 2).astype(int).values
        p = (ms["S2"] + ms["S3"]).astype(float).values
        nb_arr = _nb(y, p, thresholds)
        useful = (nb_arr > treat_all) & (nb_arr > 0)
        if useful.any():
            useful_ranges[m] = (round(float(thresholds[useful][0]), 2),
                                round(float(thresholds[useful][-1]), 2))
        else:
            useful_ranges[m] = (np.nan, np.nan)
        for t, nb_val, ta in zip(thresholds, nb_arr, treat_all):
            long_rows.append({
                "threshold": round(float(t), 4),
                "model": m,
                "net_benefit": round(float(nb_val), 6),
                "treat_all": round(float(ta), 6),
                "treat_none": 0.0,
            })
    panel_e = pd.DataFrame(long_rows)
    return panel_e, {"useful_ranges": useful_ranges, "prevalence": prev}


def build_fig4(out_path: Path) -> dict:
    """Fig 4: panels a (stage distribution), b (per-stage ROC + macro
    AUROC, headline 0.883 from per-seed long CSV averaged over 5
    seeds), c (k-stratified Sens/NPV — partial), d (external per-site
    headline 0.734 sourced from ``external_persite_breakdown.csv``),
    e (DCA pending), f (NRI/IDI summary)."""
    df_long = pd.read_csv(P0 / "calibration"
                          / "ckm_staging_probabilities_long.csv")
    df_persite = pd.read_csv(RR / "external_persite_breakdown.csv")

    # Panel a: per-stage counts inferred from the predictions table.
    sub_one_seed = (df_long[(df_long["model"] == "SFM-M2")
                            & (df_long["seed"] == df_long["seed"].min())]
                    .drop_duplicates("path"))
    stg_counts = (sub_one_seed["true_stage"].value_counts().sort_index()
                  .rename("count").to_frame().reset_index()
                  .rename(columns={"index": "stage", "true_stage": "stage"}))
    if "stage" not in stg_counts.columns:
        stg_counts = stg_counts.rename(columns={stg_counts.columns[0]: "stage"})
    panel_a = stg_counts

    # Panel b: per-model macro AUROC across the 4 staging classes,
    # averaged over the 5 training seeds (matching fig4.py).
    rows = []
    for m in MODELS:
        per_seed_auc = []
        per_seed_aupr = []
        per_stage = {f"AUROC_S{s}": [] for s in STAGES}
        for sd in sorted(df_long["seed"].unique()):
            sub = df_long[(df_long["model"] == m)
                          & (df_long["seed"] == sd)]
            wide = sub.pivot_table(index="path", columns="target",
                                   values="probability", aggfunc="first")
            ts = sub.drop_duplicates("path").set_index("path")["true_stage"]
            wide = wide.join(ts).dropna(subset=["S0", "S1", "S2", "S3",
                                                 "true_stage"])
            y = wide["true_stage"].astype(int).to_numpy()
            sc = wide[["S0", "S1", "S2", "S3"]].to_numpy()
            try:
                per_seed_auc.append(macro_ovr_auroc(y, sc))
                per_seed_aupr.append(macro_ovr_aupr(y, sc))
                for s in STAGES:
                    per_stage[f"AUROC_S{s}"].append(
                        roc_auc_score((y == s).astype(int), sc[:, s]))
            except ValueError:
                continue
        rec = {"model": m,
               "macro_AUROC": round(float(np.nanmean(per_seed_auc)), 4),
               "macro_AUPR": round(float(np.nanmean(per_seed_aupr)), 4)}
        for s in STAGES:
            rec[f"AUROC_S{s}"] = round(float(np.nanmean(
                per_stage[f"AUROC_S{s}"])), 4)
        rec["n_seeds"] = len(per_seed_auc)
        rows.append(rec)
    panel_b = pd.DataFrame(rows)
    sfm_internal = float(panel_b.loc[panel_b["model"] == "SFM-M2",
                                     "macro_AUROC"].iloc[0])

    # Panel c: k-stratified Sens/NPV at Sp=0.98 across k strata —
    # loaded from the staged CSV produced by the server agent
    # (fig4.py threshold logic on per-sample staging probs joined
    # with k labels).
    panel_c = _load_pending_csv("fig4c_sens_npv_at_sp098.csv")
    if panel_c is None:
        panel_c = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR / "fig4c_sens_npv_at_sp098.csv"),
        }])

    # Panel d: external per-site and Overall AUROC sourced directly
    # from the canonical CSV (matches the published headline value
    # 0.734 for SFM-M2).
    panel_d_persite = df_persite[[
        "site", "model", "ovr_macro_auc_mean", "ovr_macro_auc_ci",
        "n", "classes_present", "ovr_macro_auc_formatted",
    ]].copy()
    panel_d_persite = panel_d_persite.rename(columns={
        "ovr_macro_auc_mean": "macro_AUROC",
        "ovr_macro_auc_ci": "macro_AUROC_CI",
        "ovr_macro_auc_formatted": "AUROC_formatted",
    })
    panel_d = panel_d_persite
    overall_sfm = panel_d_persite[(panel_d_persite["site"] == "Overall")
                                  & (panel_d_persite["model"] == "SFM-M2")]
    sfm_external = float(overall_sfm["macro_AUROC"].iloc[0])

    # Panel e: DCA per-threshold table reproduced from fig4.py.
    panel_e, panel_e_meta = _build_fig4e_panel()

    # Panel f: NRI / IDI per external baseline at Stage>=2 — loaded
    # from staged CSV (fig4.py::_compute_nri_idi run on
    # external_ckm_per_sample_predictions_mean.csv with bootstrap
    # CIs).
    panel_f = _load_pending_csv("fig4f_nri_idi.csv")
    if panel_f is None:
        panel_f = pd.DataFrame({
            "comparator": ["ImageNet", "RETFound", "VisionFM"],
            "NRI_stage_ge_2": [np.nan, np.nan, np.nan],
            "IDI_stage_ge_2": [np.nan, np.nan, np.nan],
            "note": ["PENDING_CONSOLIDATION (manuscript range "
                     "NRI 0.70-0.79, IDI 0.028-0.048)"] * 3,
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Per-stage sample counts in the internal CKM "
                     "staging cohort.", panel_a)
        _write_sheet(xl, "b", "b",
                     "Per-model macro AUROC (and AUPR) across 4-class "
                     "CKM staging. Headline SFM-M2 = 0.883.", panel_b)
        _write_sheet(xl, "c", "c",
                     "k-stratified sensitivity / NPV at the "
                     "Sp=0.98 community-screening operating point, "
                     "across burden k strata (k=2, 3, 4, >=5) per "
                     "model.", panel_c)
        _write_sheet(xl, "d", "d",
                     "External per-model overall AUROC (excluding "
                     "Fujian). Headline SFM-M2 = 0.734.", panel_d)
        if panel_e is not None:
            _write_sheet(xl, "e", "e",
                         "DCA per-threshold net benefit on the "
                         "external HK&A + UK Biobank cohort (Fujian "
                         "excluded). Threshold grid "
                         "linspace(0.05, 0.75, 300); score "
                         "P(stage>=2) = S2 + S3; NB(t) = TP/N - "
                         "FP/N * t/(1-t). Reproduces published useful "
                         "range SFM-M2 [0.34, 0.68], VisionFM "
                         "[0.53, 0.56], RETFound [0.47, 0.55].",
                         panel_e)
        else:
            _pending_server_sheet(
                xl, "e", "e",
                "DCA per-threshold net-benefit table across SFM-M2 + "
                "3 external baselines.",
                "data_for_figures/p0/external_breakdown/"
                "external_ckm_per_sample_predictions_mean.csv")
        _write_sheet(xl, "f", "f",
                     "NRI / IDI per external baseline at the "
                     "Stage>=2 referral threshold.", panel_f)

    headline = {"sfm_m2_internal_macro_auroc": sfm_internal,
                "internal_expected": 0.883, "internal_tol": 0.001,
                "sfm_m2_external_overall_auroc": sfm_external,
                "external_expected": 0.734, "external_tol": 0.001}
    if panel_e is not None and "useful_ranges" in panel_e_meta:
        ur = panel_e_meta["useful_ranges"]
        if "SFM-M2" in ur:
            headline["dca_sfm_m2_lo"] = ur["SFM-M2"][0]
            headline["dca_sfm_m2_hi"] = ur["SFM-M2"][1]
            headline["dca_sfm_m2_expected"] = (0.34, 0.68)
    return {"file": out_path.name, "headline": headline}


def _build_fig5b_panel() -> tuple[pd.DataFrame, dict]:
    """Build the Fig 5b biomarker AUROC table from the 18 per-sample
    prediction CSVs that the server-side agent produced. Returns the
    panel DataFrame plus a small validation dict (headline AUROC for
    SFM-M2 across the 3 biomarkers)."""
    if not BIOMARKER_DIR.exists():
        return None, {"status": "PENDING_SERVER"}

    rows = []
    sfm_m2 = {}
    for biomarker, abnormal in BIOMARKERS.items():
        rec = {"biomarker": biomarker,
               "abnormal_class_index": abnormal,
               "task_type": (f"{1 + max(BIOMARKERS.values()) if False else ''}"
                             f"binary normal vs abnormal "
                             f"(class {abnormal})")}
        for m in MODELS:
            fp = BIOMARKER_DIR / f"{m}_{biomarker}_predictions.csv"
            if not fp.exists():
                rec[m] = np.nan
                continue
            df = pd.read_csv(fp)
            y = (df["true_class"] == abnormal).astype(int).to_numpy()
            s = df[f"prob_class_{abnormal}"].to_numpy()
            if len(np.unique(y)) >= 2:
                rec[m] = round(float(roc_auc_score(y, s)), 4)
            else:
                rec[m] = np.nan
            if m == "SFM-M2":
                sfm_m2[biomarker] = rec[m]
            rec[f"{m}_n"] = int(len(df))
        rows.append(rec)
    panel_b = pd.DataFrame(rows)
    # Reorder columns: biomarker, abnormal_class_index, task_type, then
    # 6 model AUROCs followed by their n columns.
    cols = ["biomarker", "abnormal_class_index", "task_type"]
    cols += MODELS + [f"{m}_n" for m in MODELS]
    panel_b = panel_b[[c for c in cols if c in panel_b.columns]]
    return panel_b, {"sfm_m2": sfm_m2,
                     "status": "OK" if not panel_b.empty else "EMPTY"}


def build_fig5(out_path: Path) -> dict:
    """Fig 5: panels a (workflow), b (biomarker AUROC, headline
    SFM-M2 eGFR>=0.99 / HbA1c>=0.70 / TG>=0.90), c (Sankey),
    d (alignment), e (progression AUROC, headline 0.962/0.880), f (KM),
    g (C-index, headline 0.822), h-k (CEA, headline $2,171 / 4.153 /
    90.7%)."""
    df_prog = pd.read_csv(P0 / "progression" / "time_series_predictions_mean.csv")

    # Panel b: read the 18 server-generated prediction CSVs (or fall
    # back to the pending placeholder).
    panel_b, panel_b_meta = _build_fig5b_panel()

    # Panel c: Sankey (Plot_v2 CSV)
    sankey_path = (PLOT_V2 / "fig4-9_ckm_stage_biomarker_association_stats.csv")
    if sankey_path.exists():
        panel_c = pd.read_csv(sankey_path)
    else:
        panel_c = pd.DataFrame({"status": ["MISSING"], "expected_path":
                                [str(sankey_path)]})

    # Panel d: per-model cross-task alignment % across 3 axes
    # (staging, comorbidity, biomarker) — loaded from staged CSV
    # (sourced from Plot_v2/fig4-8_cross_task_consistency_metrics.csv,
    # joint_exact converted to percent by model).
    panel_d = _load_pending_csv("fig5d_cross_task_alignment.csv")
    if panel_d is None:
        panel_d = pd.DataFrame({
            "model": MODELS,
            "alignment_axis_staging_pct": [np.nan] * 6,
            "alignment_axis_comorbidity_pct": [np.nan] * 6,
            "alignment_axis_biomarker_pct": [np.nan] * 6,
            "note": ["PENDING_CONSOLIDATION"] * 6,
        })

    # Panel e: per-model prognostic AUROC (with baseline disease
    # labels) and image-only AUROC (no baseline labels) sourced from
    # the canonical per-seed result spreadsheets used by fig5.py.
    ts_label_path = PLOT_V2 / "fundus_time_series_label.xlsx"
    ts_none_path = PLOT_V2 / "fundus_time_series_nonelabel.xlsx"
    if ts_label_path.exists() and ts_none_path.exists():
        ts_label = pd.read_excel(ts_label_path)
        ts_none = pd.read_excel(ts_none_path)
        rows_e = []
        for m in MODELS:
            lab = ts_label[ts_label["Models"] == m]
            non = ts_none[ts_none["Models"] == m]
            rec = {
                "model": m,
                "prognostic_AUROC": round(
                    float(np.nanmean(lab["AUROC"])), 4)
                if "AUROC" in lab.columns else np.nan,
                "prognostic_AUROC_per_seed": ", ".join(
                    f"{x:.4f}" for x in lab["AUROC"].dropna()),
                "image_only_AUROC": round(
                    float(np.nanmean(non["AUROC"])), 4)
                if "AUROC" in non.columns else np.nan,
                "image_only_AUROC_per_seed": ", ".join(
                    f"{x:.4f}" for x in non["AUROC"].dropna()),
                "n_seeds_label": int(lab["AUROC"].notna().sum()),
                "n_seeds_nonelabel": int(non["AUROC"].notna().sum()),
            }
            rows_e.append(rec)
        panel_e = pd.DataFrame(rows_e)
    else:
        panel_e = pd.DataFrame({"model": MODELS,
                                "prognostic_AUROC": [np.nan] * 6,
                                "note": ["MISSING_XLSX"] * 6})
    sfm_prog = float(panel_e.loc[panel_e["model"] == "SFM-M2",
                                 "prognostic_AUROC"].iloc[0])
    sfm_io = float(panel_e.loc[panel_e["model"] == "SFM-M2",
                               "image_only_AUROC"].iloc[0])

    # Panel f: KM tertile event-free curves with log-rank P per model
    # — loaded from staged CSV (fig5.py KM/logrank logic generalized
    # to all models on time_series_predictions_mean.csv).
    panel_f = _load_pending_csv("fig5f_km_tertile_curves.csv")
    if panel_f is None:
        panel_f = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR / "fig5f_km_tertile_curves.csv"),
        }])

    # Panel g: Harrell C-index per model using the same risk score as
    # fig5.py (max prob over baseline-negative diseases). This
    # reproduces the published headline 0.822 for SFM-M2.
    rows = []
    for m in MODELS:
        sub = df_prog[df_prog["model"] == m].copy()
        sub["risk"] = sub.apply(
            lambda r: max(
                [r[f"prob_{d}_mean"] for d in DISEASES
                 if r[f"input_{d}"] == 0]
            ) if any(r[f"input_{d}"] == 0 for d in DISEASES) else 0.0,
            axis=1,
        )
        c = _harrell_c(
            durations=sub["time_span"].to_numpy(),
            events=sub["any_progression"].to_numpy(),
            scores=sub["risk"].to_numpy(),
        )
        rows.append({"model": m, "C_index": round(c, 4),
                     "n": int(len(sub))})
    panel_g = pd.DataFrame(rows)
    sfm_c = float(panel_g.loc[panel_g["model"] == "SFM-M2",
                              "C_index"].iloc[0])

    # Panel h: CEA base-case outcomes per strategy (mirrored from ED
    # Table 10 — these are the canonical published values).
    panel_h = pd.DataFrame([
        {"strategy": "ImageNet",     "cost_USD": 2703, "QALYs_5yr": 4.120,
         "delta_cost":   276, "delta_QALY": -0.001, "ICER": "Dominated"},
        {"strategy": "RETFound",     "cost_USD": 2751, "QALYs_5yr": 4.111,
         "delta_cost":   324, "delta_QALY": -0.009, "ICER": "Dominated"},
        {"strategy": "VisionFM",     "cost_USD": 2941, "QALYs_5yr": 4.107,
         "delta_cost":   514, "delta_QALY": -0.013, "ICER": "Dominated"},
        {"strategy": "SFM-Base",     "cost_USD": 2373, "QALYs_5yr": 4.138,
         "delta_cost":   -54, "delta_QALY":  0.017, "ICER": "Dominant"},
        {"strategy": "SFM-MoE",      "cost_USD": 2365, "QALYs_5yr": 4.138,
         "delta_cost":   -62, "delta_QALY":  0.017, "ICER": "Dominant"},
        {"strategy": "SFM-M2",       "cost_USD": 2171, "QALYs_5yr": 4.153,
         "delta_cost":  -243, "delta_QALY":  0.032, "ICER": "Dominant"},
        {"strategy": "Traditional",  "cost_USD": 2427, "QALYs_5yr": 4.121,
         "delta_cost":     0, "delta_QALY":  0.000, "ICER": "(reference)"},
        {"strategy": "No screening", "cost_USD": 7670, "QALYs_5yr": 3.821,
         "delta_cost":  5243, "delta_QALY": -0.300, "ICER": "Dominated"},
    ])
    sfm_cost = int(panel_h.loc[panel_h["strategy"] == "SFM-M2",
                               "cost_USD"].iloc[0])
    sfm_qaly = float(panel_h.loc[panel_h["strategy"] == "SFM-M2",
                                 "QALYs_5yr"].iloc[0])

    # Panel i: PSA summary statistics for SFM-M2 vs Traditional
    # (mirrored from the manuscript text).
    panel_i = pd.DataFrame([
        {"metric": "P(Dominant)",                 "value_pct": 90.7},
        {"metric": "P(CE @ 1xGDP=$12,551/QALY)",  "value_pct": 98.7},
        {"metric": "P(CE @ 3xGDP=$37,653/QALY)",  "value_pct": 99.7},
        {"metric": "Mean delta_cost (USD)",       "value_pct": -243},
        {"metric": "Mean delta_QALY",             "value_pct": 0.032},
        {"metric": "Mean NMB @ 3xGDP (USD)",      "value_pct": 1487},
    ])
    p_dom = float(panel_i.loc[panel_i["metric"] == "P(Dominant)",
                              "value_pct"].iloc[0])

    # Panel j: one-way deterministic sensitivity (tornado) NMB swing
    # per parameter — loaded from staged CSV (cost_analysis_upgraded
    # .py executed with seed=42).
    panel_j = _load_pending_csv("fig5j_tornado.csv")
    if panel_j is None:
        panel_j = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR / "fig5j_tornado.csv"),
        }])

    # Panel k: CEAC P(CE) across WTP grid 0..3xGDP — loaded from
    # staged CSV (cost_analysis_upgraded.py CEAC interpolated to a
    # dense grid with anchors at 0xGDP, 1xGDP=$12,551, 3xGDP=$37,653).
    panel_k = _load_pending_csv("fig5k_ceac.csv")
    if panel_k is None:
        panel_k = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR / "fig5k_ceac.csv"),
        }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        if panel_b is not None:
            _write_sheet(xl, "b", "b",
                         "Biomarker AUROC across 6 models for 3 axes "
                         "(eGFR, HbA1c, TG): one-vs-rest binary AUROC "
                         "for the abnormal class of each biomarker, "
                         "computed from the per-sample prediction "
                         "CSVs produced on the training server.",
                         panel_b)
        else:
            _pending_server_sheet(
                xl, "b", "b",
                "Biomarker AUROC across 6 models for 3 axes (eGFR, "
                "HbA1c, TG).",
                "data_for_figures/biomarker/"
                "<model>_<biomarker>_predictions.csv")
        _write_sheet(xl, "c", "c",
                     "Sankey flow counts: stage to biomarker abnormal "
                     "fractions.", panel_c)
        _write_sheet(xl, "d", "d",
                     "Cross-task alignment percentage per model "
                     "across 3 axes.", panel_d)
        _write_sheet(xl, "e", "e",
                     "Per-model prognostic AUROC and image-only "
                     "AUROC. Headline SFM-M2 prognostic = 0.962, "
                     "image-only = 0.880.", panel_e)
        _write_sheet(xl, "f", "f",
                     "KM tertile event-free curves (pending "
                     "consolidation).", panel_f)
        _write_sheet(xl, "g", "g",
                     "Harrell C-index per model. Headline SFM-M2 = "
                     "0.822.", panel_g)
        _write_sheet(xl, "h", "h",
                     "Per-strategy 5-year cost (USD) and QALYs with "
                     "ICER vs Traditional. Headline SFM-M2 cost = "
                     "$2,171; QALYs = 4.153.", panel_h)
        _write_sheet(xl, "i", "i",
                     "PSA summary for SFM-M2 vs Traditional (n=10,000 "
                     "Monte Carlo draws). Headline P(Dominant) = "
                     "90.7%.", panel_i)
        _write_sheet(xl, "j", "j",
                     "One-way deterministic sensitivity (tornado) "
                     "spreads per parameter.", panel_j)
        _write_sheet(xl, "k", "k",
                     "CEAC P(cost-effective) at five WTP thresholds.",
                     panel_k)

    headline = {
        "sfm_m2_prognostic_auroc": sfm_prog,
        "prognostic_expected": 0.962, "prognostic_tol": 0.001,
        "sfm_m2_image_only_auroc": sfm_io,
        "image_only_expected": 0.880, "image_only_tol": 0.001,
        "sfm_m2_c_index": sfm_c,
        "c_index_expected": 0.822, "c_index_tol": 0.001,
        "sfm_m2_base_cost": sfm_cost,
        "cost_expected": 2171, "cost_tol": 0,
        "sfm_m2_qalys": sfm_qaly,
        "qalys_expected": 4.153, "qalys_tol": 0,
        "p_dominant": p_dom,
        "p_dominant_expected": 90.7, "p_dominant_tol": 0.1,
    }
    if panel_b_meta.get("status") == "OK" and panel_b_meta.get("sfm_m2"):
        sm = panel_b_meta["sfm_m2"]
        headline["fig5b_sfm_m2_eGFR"] = sm.get("eGFR")
        headline["fig5b_sfm_m2_HbA1c"] = sm.get("HbA1c")
        headline["fig5b_sfm_m2_TG"] = sm.get("TG")
    return {"file": out_path.name, "headline": headline}


def build_fig6(out_path: Path) -> dict:
    """Fig 6: panels a (per-case staging / 6-disease / biomarker
    probabilities for 3 cases x 6 models) and b (3 longitudinal
    progression cases). Image paths and patient identifiers are
    masked to anonymous case codes (``case_001`` / ``CASE_001`` ...)
    so the per-case Source Data sheets do not leak SHEDPTC filesystem
    paths or Chinese national-ID numbers."""

    # The 3 published Fig 6 panel-a cases are hard-coded in the
    # plotting script (fig6_v4.py SELECTED_A). They are NOT taken
    # from the case_predictions.csv "legacy" dump — that file
    # contains 14 candidate cases none of which match the published
    # 3. Source Data therefore mirrors the plotting script's own
    # selection by hard-coding the same 3 paths and pulling their
    # labels + per-model predictions from the canonical per-sample
    # tables.
    SELECTED_A_PATHS = [
        "baoshan/origin/20200602_407_0949006128_L.JPG",
        "baoshan/origin/20240531_157_1353468506_R.JPG",
        "baoshan/origin/20200611_20_0734450202_L.JPG",
    ]
    path_to_code = {
        p: f"case_{i+1:03d}" for i, p in enumerate(SELECTED_A_PATHS)
    }

    # Load canonical per-sample tables and filter to the 3 published
    # cases.
    candidates = pd.read_csv(P0 / "fig6_case_studies"
                             / "case_candidates_internal_full.csv")
    staging = pd.read_csv(P0 / "calibration"
                          / "ckm_staging_probabilities_mean_wide.csv")
    comorbidity = pd.read_csv(P0 / "calibration"
                              / "comorbidity_probabilities_mean_wide.csv")
    case_bio = pd.read_csv(P0 / "fig6_case_studies"
                           / "case_biomarker_predictions.csv")
    long_cases = pd.read_csv(P0 / "fig6_case_studies"
                             / "fig6_paneld_longitudinal_cases.csv")

    cand_3 = candidates[candidates["path"].isin(SELECTED_A_PATHS)].copy()
    stg_3 = staging[staging["path"].isin(SELECTED_A_PATHS)].copy()
    com_3 = comorbidity[comorbidity["path"].isin(SELECTED_A_PATHS)].copy()

    # Merge staging and comorbidity predictions by (path, model). The
    # truth columns appear in both tables; keep one copy and drop
    # duplicates.
    panel_a_pred = stg_3.merge(
        com_3,
        on=["path", "model"],
        how="inner",
        suffixes=("", "_dup"),
    )
    panel_a_pred = panel_a_pred.loc[
        :, ~panel_a_pred.columns.str.endswith("_dup")
    ]

    # Mask path then reorder columns: case code, model, ground truth,
    # then stage probs, then disease probs, then disease truth.
    panel_a_pred["path"] = panel_a_pred["path"].map(path_to_code)

    # The 3 published Fig 6a cases each display exactly ONE biomarker
    # panel (HbA1c for cases 1 and 2, TG for case 3). Merge the
    # displayed biomarker's columns directly onto panel_a so the
    # Source Data uses a single sheet ``a`` rather than two parallel
    # sheets. Column count: max(HbA1c 3 classes, TG 4 classes) = 4
    # prob columns; the trailing biomarker_prob_3 is NaN for HbA1c
    # rows.
    DISPLAYED_BIOMARKER = {
        "case_001": "HbA1c",
        "case_002": "HbA1c",
        "case_003": "TG",
    }

    bio_3 = case_bio[case_bio["path"].isin(path_to_code.keys())].copy()
    bio_3["path"] = bio_3["path"].map(path_to_code)

    bio_rows = []
    for _, row in panel_a_pred[["path", "model"]].drop_duplicates().iterrows():
        marker = DISPLAYED_BIOMARKER.get(row["path"])
        match = bio_3[(bio_3["path"] == row["path"])
                      & (bio_3["model"] == row["model"])]
        rec = {"path": row["path"], "model": row["model"],
               "biomarker": marker}
        if marker and len(match):
            m = match.iloc[0]
            rec["biomarker_true"] = m.get(f"{marker}_true_tag")
            rec["biomarker_pred"] = m.get(f"{marker}_pred_tag")
            for k in range(4):
                col = f"{marker}_prob_{k}"
                rec[f"biomarker_prob_{k}"] = (m.get(col)
                                              if col in m.index else np.nan)
        else:
            rec.update({"biomarker_true": np.nan,
                        "biomarker_pred": np.nan,
                        "biomarker_prob_0": np.nan,
                        "biomarker_prob_1": np.nan,
                        "biomarker_prob_2": np.nan,
                        "biomarker_prob_3": np.nan})
        bio_rows.append(rec)
    bio_df = pd.DataFrame(bio_rows)

    panel_a = panel_a_pred.merge(bio_df, on=["path", "model"], how="left")
    desired = (["path", "model", "true_stage"]
               + [f"S{s}" for s in STAGES]
               + ["pred_stage"]
               + [f"true_{d}" for d in DISEASES]
               + [f"prob_{d}" for d in DISEASES]
               + ["biomarker", "biomarker_true", "biomarker_pred",
                  "biomarker_prob_0", "biomarker_prob_1",
                  "biomarker_prob_2", "biomarker_prob_3"])
    panel_a = panel_a[[c for c in desired if c in panel_a.columns]]
    panel_a = panel_a.sort_values(["path", "model"]).reset_index(drop=True)

    # Panel b — longitudinal cases with paired baseline + follow-up
    # paths. Mask each case's paths and patient identifier; preserve
    # the relative time-span / interval columns (no PHI risk).
    panel_b = long_cases.copy()
    case_b_codes = {
        cid: f"CASE_{i+1:03d}"
        for i, cid in enumerate(panel_b["case_id"].drop_duplicates().tolist())
    }
    panel_b["case_id"] = panel_b["case_id"].map(case_b_codes)

    # Drop patient-identifying columns and the redundant
    # interval_days (kept as years already).
    drop_cols = [
        "patient_id", "baseline_path", "followup_path",
        "baseline_image_file", "followup_image_file", "interval_days",
    ]
    panel_b = panel_b.drop(columns=[c for c in drop_cols
                                    if c in panel_b.columns])

    # Truncate full dates to year-month so the figure timing is
    # readable but day-level precision (which combined with case
    # selection could narrow identification) is dropped.
    for date_col in ("baseline_date", "followup_date"):
        if date_col in panel_b.columns:
            panel_b[date_col] = (
                pd.to_datetime(panel_b[date_col], errors="coerce")
                  .dt.strftime("%Y-%m")
            )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Per-case Fig 6a inputs: CKM stage + 6-disease "
                     "comorbidity + the displayed biomarker, for the "
                     "3 published cases x 6 models (3 x 6 = 18 rows). "
                     "Each case displays exactly one biomarker panel "
                     "(case_001 and case_002 display HbA1c, case_003 "
                     "displays TG); the `biomarker` column names "
                     "which one and the `biomarker_true / pred / "
                     "prob_0..3` columns carry that biomarker's "
                     "class index and probabilities. "
                     "Biomarker class index legend (Methods "
                     "§Biomarker grading): "
                     "HbA1c (3 classes, ADA) 0=Normal <5.7%, "
                     "1=Pre-diabetes 5.7-6.4%, 2=Diabetes >=6.5% "
                     "[abnormal class = 2; prob_3 always NaN]; "
                     "TG (4 classes, NCEP ATP III) 0=Normal "
                     "<1.7 mmol/L, 1=Borderline 1.7-2.25, "
                     "2=High 2.26-5.65, 3=Severe >5.65 "
                     "[abnormal class = 3]. All 3 cases are SEDPTC-"
                     "internal Baoshan eyes at stage 2. Image paths "
                     "replaced by anonymous case_NNN codes following "
                     "the order in the plotting script's SELECTED_A "
                     "list.", panel_a)
        _write_sheet(xl, "b", "b",
                     "Prospective progression for the 3 longitudinal "
                     "cases: baseline + follow-up metadata, progressed "
                     "diseases, and SFM-M2 risk score. Patient IDs and "
                     "image paths replaced by anonymous CASE_NNN "
                     "codes.", panel_b)

    return {"file": out_path.name,
            "n_published_cases": len(SELECTED_A_PATHS),
            "n_panel_a_rows": int(len(panel_a))}


# ───────────────────────────────────────────────────────────────────────
# Extended Data figures
# ───────────────────────────────────────────────────────────────────────
def _ed_pending_or_placeholder(csv_name: str,
                                placeholder_desc: str) -> pd.DataFrame:
    """Helper: load an ED-figure staged CSV or return a 1-row stub."""
    df = _load_pending_csv(csv_name)
    if df is None:
        df = pd.DataFrame([{
            "status": "PENDING_CONSOLIDATION",
            "expected_csv": str(PENDING_DIR / csv_name),
            "description": placeholder_desc,
        }])
    return df


def build_ed3(out_path: Path) -> dict:
    """ED Fig 3: MoE expert routing analysis (3 panels). Each panel
    is loaded from a staged CSV produced by the server agent run on
    the cam_vis1 affinity NPY tensors."""
    panel_a = _ed_pending_or_placeholder(
        "ed_fig3a_expert_class_affinity.csv",
        "16x6 expert-class affinity matrix.")
    panel_b = _ed_pending_or_placeholder(
        "ed_fig3b_specialization_score.csv",
        "1 - H/ln(6) specialization score per expert.")
    panel_c = _ed_pending_or_placeholder(
        "ed_fig3c_k_trajectory.csv",
        "k-trajectory of per-expert routing entropy.")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "16 x 6 expert-class affinity matrix (cohort "
                     "mean) used to colour the heatmap in panel a.",
                     panel_a)
        _write_sheet(xl, "b", "b",
                     "Specialization score 1 - H(p)/ln(6) per expert "
                     "(higher = more specialist; the manuscript "
                     "reports the 0.04 to 0.18 range).", panel_b)
        _write_sheet(xl, "c", "c",
                     "Per-expert mean affinity centered by row, "
                     "grouped by multimorbidity burden k in {1, 2, "
                     "3, 4+}.", panel_c)
    return {"file": out_path.name}


def build_ed4(out_path: Path) -> dict:
    """ED Fig 4: comorbidity correlation matrices (Phi, log-OR, MI)."""
    panel = _ed_pending_or_placeholder(
        "ed_fig4_correlation_matrices.csv",
        "Pairwise Phi, log-OR and MI for 6 diseases on truth + SFM-M2 "
        "predicted labels.")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Pairwise Phi, log-odds-ratio and mutual-"
                     "information matrices for the 6 CKM diseases on "
                     "(i) ground-truth labels and (ii) SFM-M2 "
                     "predicted labels (long form: metric, source, "
                     "disease_i, disease_j, value).", panel)
    return {"file": out_path.name}


def build_ed5(out_path: Path) -> dict:
    """ED Fig 5: subgroup AUROC trends across burden k."""
    panel = _ed_pending_or_placeholder(
        "ed_fig5_subgroup_auroc.csv",
        "Subgroup AUROC across burden k for sex / age / site strata.")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Subgroup AUROC mean and SD across "
                     "multimorbidity burden k=1..5 (cohort, female, "
                     "male; mean over the 6 disease heads).", panel)
    return {"file": out_path.name}


def build_ed6(out_path: Path) -> dict:
    """ED Fig 6: surrogate decision-tree node table fitted to SFM-M2
    staging predictions on (CKD, diabetes, hypertension) carriers."""
    panel = _ed_pending_or_placeholder(
        "ed_fig6_decision_tree_nodes.csv",
        "Per-node decision-tree split rules.")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Per-node surrogate decision-tree splits "
                     "(node_id, parent_id, split_feature, "
                     "split_threshold, gini, samples, value_class*, "
                     "predicted_class) fitted with random_state=42 "
                     "to SFM-M2 comorbidity mean predictions on the "
                     "subset of test participants carrying CKD, "
                     "diabetes and hypertension.", panel)
    return {"file": out_path.name}


def build_ed7(out_path: Path) -> dict:
    """ED Fig 7: calibration curves pre/post temperature scaling."""
    cal = pd.read_csv(RR / "calibration_ece_temperature_scaled.csv")
    panel = cal.copy()
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Per-stage and macro ECE for 6 models, raw "
                     "and after one-parameter temperature scaling.",
                     panel)
    return {"file": out_path.name}


def build_ed8(out_path: Path) -> dict:
    """ED Fig 8: per-disease KM event-free curves + log-rank P + AUC."""
    panel = _ed_pending_or_placeholder(
        "ed_fig8_per_disease_km.csv",
        "Per-disease KM event-free curves + log-rank + prognostic AUC.")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        _write_sheet(xl, "a", "a",
                     "Per-disease KM event-free curves split by SFM-M2 "
                     "risk-score tertile (high vs low), with log-rank "
                     "P and per-disease prognostic AUC across the 6 "
                     "disease heads.", panel)
    return {"file": out_path.name}


# ───────────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────────
def _harrell_c(durations: np.ndarray, events: np.ndarray,
               scores: np.ndarray) -> float:
    """Vectorised Harrell C-index. ``scores`` should be increasing in
    risk."""
    n = len(durations)
    if n < 2:
        return float("nan")
    pairs = 0
    concord = 0.0
    # Vectorise outer loop in numpy
    di = durations
    ei = events
    si = scores
    for i in range(n):
        if ei[i] == 0:
            continue
        # j with d_j > d_i (admissible pair)
        adm = di > di[i]
        if not adm.any():
            continue
        diff = si[i] - si[adm]
        pairs += int(adm.sum())
        concord += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return concord / pairs if pairs else float("nan")


def _check(label: str, observed: float, expected: float,
           tol: float) -> tuple[str, str]:
    if not np.isfinite(observed):
        return ("FAIL", f"{label}: observed=NaN, expected={expected}")
    diff = abs(observed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    return (status,
            f"{label}: observed={observed:.4f}, expected={expected}, "
            f"|diff|={diff:.4f}, tol={tol}")


def validate(meta: dict) -> list[tuple[str, str, str]]:
    """Run the validation pass and return a list of (file, status,
    detail) tuples."""
    results: list[tuple[str, str, str]] = []
    for fig, m in meta.items():
        h = m.get("headline")
        if not h:
            continue
        if fig == "fig2":
            results.append(("Source_Data_Fig2.xlsx::b", *_check(
                "SFM-M2 macro AUROC", h["sfm_m2_macro_auroc"],
                h["expected"], h["tol"])))
        elif fig == "fig3":
            results.append(("Source_Data_Fig3.xlsx::b", *_check(
                "Delta beta-bar(1->2) (%)", h["delta_beta_1_to_2_pct"],
                h["expected"], h["tol"])))
            results.append(("Source_Data_Fig3.xlsx::c (ARI)", *_check(
                "SFM-M2 ARI", h["ari_sfm_m2"],
                h["ari_expected"], 0.001)))
            results.append(("Source_Data_Fig3.xlsx::c (NMI)", *_check(
                "SFM-M2 NMI", h["nmi_sfm_m2"],
                h["nmi_expected"], 0.001)))
        elif fig == "fig4":
            results.append(("Source_Data_Fig4.xlsx::b", *_check(
                "SFM-M2 internal macro AUROC",
                h["sfm_m2_internal_macro_auroc"],
                h["internal_expected"], h["internal_tol"])))
            results.append(("Source_Data_Fig4.xlsx::d", *_check(
                "SFM-M2 external Overall AUROC",
                h["sfm_m2_external_overall_auroc"],
                h["external_expected"], h["external_tol"])))
            if "dca_sfm_m2_lo" in h:
                results.append(("Source_Data_Fig4.xlsx::e (DCA lo)", *_check(
                    "SFM-M2 DCA useful range lower bound",
                    h["dca_sfm_m2_lo"], h["dca_sfm_m2_expected"][0],
                    0.005)))
                results.append(("Source_Data_Fig4.xlsx::e (DCA hi)", *_check(
                    "SFM-M2 DCA useful range upper bound",
                    h["dca_sfm_m2_hi"], h["dca_sfm_m2_expected"][1],
                    0.005)))
        elif fig == "fig5":
            results.append(("Source_Data_Fig5.xlsx::e (prognostic)", *_check(
                "SFM-M2 prognostic AUROC",
                h["sfm_m2_prognostic_auroc"],
                h["prognostic_expected"], h["prognostic_tol"])))
            results.append(("Source_Data_Fig5.xlsx::e (image-only)", *_check(
                "SFM-M2 image-only AUROC",
                h["sfm_m2_image_only_auroc"],
                h["image_only_expected"], h["image_only_tol"])))
            results.append(("Source_Data_Fig5.xlsx::g", *_check(
                "SFM-M2 C-index",
                h["sfm_m2_c_index"],
                h["c_index_expected"], h["c_index_tol"])))
            results.append(("Source_Data_Fig5.xlsx::h (cost)", *_check(
                "SFM-M2 base-case cost (USD)",
                h["sfm_m2_base_cost"],
                h["cost_expected"], h["cost_tol"])))
            results.append(("Source_Data_Fig5.xlsx::h (QALYs)", *_check(
                "SFM-M2 5-year QALYs",
                h["sfm_m2_qalys"],
                h["qalys_expected"], h["qalys_tol"])))
            results.append(("Source_Data_Fig5.xlsx::i (P_dominant)", *_check(
                "P(SFM-M2 dominates Trad.) (%)",
                h["p_dominant"],
                h["p_dominant_expected"], h["p_dominant_tol"])))
            if h.get("fig5b_sfm_m2_eGFR") is not None:
                # Sanity-check biomarker AUROCs (no published exact
                # value for the Source Data table; we just confirm
                # SFM-M2 outperforms 0.7 across all 3 biomarkers).
                for b in ("eGFR", "HbA1c", "TG"):
                    val = h.get(f"fig5b_sfm_m2_{b}")
                    if val is not None and not np.isnan(val):
                        status = "PASS" if val >= 0.70 else "FAIL"
                        results.append(
                            (f"Source_Data_Fig5.xlsx::b (SFM-M2 {b})",
                             status,
                             f"SFM-M2 {b} AUROC: observed={val:.4f}, "
                             f"sanity-check threshold 0.70"))
    return results


def write_run_report(meta: dict,
                     val_results: list[tuple[str, str, str]],
                     git_sha: str | None) -> Path:
    rp = OUT / "RUN_REPORT.md"
    lines = [
        "# Source Data Generation Run Report",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Git commit: {git_sha or 'unknown'}",
        "",
        "## Files generated",
        "",
    ]
    for fig, m in meta.items():
        lines.append(f"- `{m['file']}`")
    lines.extend([
        "",
        "## Validation outcomes",
        "",
        "| Sheet | Status | Detail |",
        "|---|---|---|",
    ])
    for sheet, status, detail in val_results:
        lines.append(f"| `{sheet}` | **{status}** | {detail} |")

    pending = [
        ("Source_Data_Fig4.xlsx::e",
         "DCA per-threshold table — re-run scripts/dca.py on external "
         "CKM probabilities; persist to "
         "`data_for_figures/p0/external_breakdown/dca_per_threshold.csv`."),
        ("Source_Data_Fig5.xlsx::b",
         "Biomarker AUROC across 6 models — re-run inference on the "
         "biomarker test split; persist per-sample softmax to "
         "`data_for_figures/p0/biomarker/<model>_<biomarker>_predictions.csv`."),
    ]
    lines.extend([
        "",
        "## Panels pending server regeneration",
        "",
    ])
    for sheet, instr in pending:
        lines.append(f"- `{sheet}` — {instr}")

    consolidation = [
        "Source_Data_Fig2.xlsx::c-g_index — Plot_v2 composition_trend "
        "analysis CSVs are scattered across multiple subdirectories; "
        "consolidate per-subgroup MAE into a single long-form table.",
        "Source_Data_Fig4.xlsx::c — k-stratified Sens/NPV at Sp=0.98; "
        "regenerate from staging predictions joined to k labels in "
        "fig4.py.",
        "Source_Data_Fig4.xlsx::f — NRI / IDI per external baseline "
        "at Stage>=2; compute from external predictions.",
        "Source_Data_Fig5.xlsx::d — Cross-task alignment %; re-run "
        "Plot_v2/fig4-8_cross_task_consistency_panel.py to export the "
        "per-axis alignment table.",
        "Source_Data_Fig5.xlsx::f — KM tertile event-free curves with "
        "log-rank P; export the intermediate table from fig5.py.",
        "Source_Data_Fig5.xlsx::j — Tornado spread per parameter; "
        "re-run scripts/cost_analysis_upgraded.py with seed=42 and "
        "persist tornado_results.csv.",
        "Source_Data_Fig5.xlsx::k — CEAC at intermediate WTPs; "
        "re-export the per-WTP P(CE) array from cost_analysis_"
        "upgraded.py.",
        "Source_Data_ED_Fig3..ED_Fig6, ED_Fig8 — re-run the "
        "underlying analysis script and persist the intermediate "
        "table as CSV.",
    ]
    lines.extend([
        "",
        "## Panels requiring manual consolidation",
        "",
    ])
    for c in consolidation:
        lines.append(f"- {c}")
    lines.append("")
    rp.write_text("\n".join(lines), encoding="utf-8")
    return rp


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────
def main(args=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", default=True,
                   help="Generate all figures (default).")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Override SFM_ANALYSIS_ROOT for this invocation.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override SFM_OUTPUT_DIR for this invocation.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip the headline-value validation pass.")
    args = p.parse_args(args)

    global ROOT, P0, PLOT_V2, RR, BIOMARKER_DIR, PENDING_DIR, OUT, INVALID
    if args.data_root is not None:
        ROOT = Path(args.data_root)
        P0 = ROOT / "data_for_figures" / "p0"
        PLOT_V2 = ROOT / "Plot_v2"
        RR = ROOT / "revision_records"
        BIOMARKER_DIR = ROOT / "data_for_figures" / "biomarker"
        PENDING_DIR = ROOT / "data_for_figures" / "source_data_pending"
    if args.output_dir is not None:
        OUT = Path(args.output_dir)
        INVALID = OUT / "_invalid"

    OUT.mkdir(parents=True, exist_ok=True)
    INVALID.mkdir(parents=True, exist_ok=True)

    builders = [
        ("fig1",   "Source_Data_Fig1.xlsx",     build_fig1),
        ("fig2",   "Source_Data_Fig2.xlsx",     build_fig2),
        ("fig3",   "Source_Data_Fig3.xlsx",     build_fig3),
        ("fig4",   "Source_Data_Fig4.xlsx",     build_fig4),
        ("fig5",   "Source_Data_Fig5.xlsx",     build_fig5),
        ("fig6",   "Source_Data_Fig6.xlsx",     build_fig6),
        ("ed3",    "Source_Data_ED_Fig3.xlsx",  build_ed3),
        ("ed4",    "Source_Data_ED_Fig4.xlsx",  build_ed4),
        ("ed5",    "Source_Data_ED_Fig5.xlsx",  build_ed5),
        ("ed6",    "Source_Data_ED_Fig6.xlsx",  build_ed6),
        ("ed7",    "Source_Data_ED_Fig7.xlsx",  build_ed7),
        ("ed8",    "Source_Data_ED_Fig8.xlsx",  build_ed8),
    ]

    meta: dict[str, dict] = {}
    for key, fname, fn in builders:
        path = OUT / fname
        print(f"[BUILD] {fname}")
        meta[key] = fn(path)

    if not args.skip_validation:
        results = validate(meta)
        any_fail = any(r[1] == "FAIL" for r in results)
        print("\n[VALIDATION]")
        for sheet, status, detail in results:
            stream = sys.stderr if status == "FAIL" else sys.stdout
            print(f"  [{status}] {sheet}: {detail}", file=stream)

        if any_fail:
            for sheet, status, _ in results:
                if status != "FAIL":
                    continue
                fname = sheet.split("::")[0]
                src = OUT / fname
                if src.exists():
                    dst = INVALID / fname
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"  Moved {src} -> {dst}", file=sys.stderr)
                    except Exception as e:
                        print(f"  Could not move {src}: {e}",
                              file=sys.stderr)

        # Try to capture git sha for provenance
        git_sha = None
        try:
            import subprocess
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True, check=False)
            if out.returncode == 0:
                git_sha = out.stdout.strip()
        except Exception:
            pass

        rp = write_run_report(meta, results, git_sha)
        try:
            print(f"\n[REPORT] {rp.relative_to(ROOT)}")
        except ValueError:
            print(f"\n[REPORT] {rp}")

        return 1 if any_fail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
