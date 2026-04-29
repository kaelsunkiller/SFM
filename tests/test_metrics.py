"""Unit tests for core evaluation metrics.

Reference: Methods §Statistical analysis.
"""

from __future__ import annotations

import numpy as np

from sfm.utils.metrics import c_index, decision_curve_table, macro_aupr, macro_auroc, nri_idi


def test_macro_auroc_binary_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert macro_auroc(y_true, y_score) == 1.0


def test_macro_aupr_multiclass_nontrivial() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_score = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    score = macro_aupr(y_true, y_score)
    assert 0.95 <= score <= 1.0


def test_nri_idi_improvement_positive() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    base = np.array([0.40, 0.30, 0.50, 0.55, 0.60, 0.35])
    new = np.array([0.30, 0.20, 0.70, 0.75, 0.80, 0.25])
    out = nri_idi(y_true, base, new)
    assert out["nri"] > 0.0
    assert out["idi"] > 0.0


def test_decision_curve_table_shape_and_bounds() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    score = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.4, 0.85])
    dca = decision_curve_table(y_true, score)
    assert len(dca) == 99
    assert np.all(dca["threshold"].between(0.01, 0.99))


def test_c_index_perfect_ranking() -> None:
    time_to_event = np.array([1, 2, 3, 4])
    event_observed = np.array([1, 1, 1, 1])
    risk_score = np.array([0.9, 0.8, 0.7, 0.6])
    assert c_index(time_to_event, event_observed, risk_score) == 1.0
