"""Unit tests for CEA decision-tree and sensitivity modules.

Reference: Methods §Cost-effectiveness analysis.
"""

from __future__ import annotations

import numpy as np

from analysis.cea.ceac import ceac_curve
from analysis.cea.decision_model import DecisionModelParameters, evaluate_no_screening, evaluate_screening
from analysis.cea.psa import PSASpec, run_psa
from analysis.cea.tornado import one_way_sensitivity


def _base_params() -> DecisionModelParameters:
    return DecisionModelParameters(
        prevalence=0.25,
        sensitivity=0.80,
        specificity=0.85,
        cost_screen=50.0,
        cost_tp=200.0,
        cost_fp=80.0,
        cost_fn=400.0,
        cost_tn=20.0,
        qaly_tp=0.90,
        qaly_fp=0.88,
        qaly_fn=0.70,
        qaly_tn=0.95,
    )


def test_screening_beats_no_screening_on_qaly() -> None:
    params = _base_params()
    scr = evaluate_screening(params)
    nos = evaluate_no_screening(params)
    assert scr.qaly > nos.qaly


def test_psa_runs_expected_rows() -> None:
    spec = PSASpec(n_samples=200, random_seed=1, wtp=10000.0)
    out = run_psa(_base_params(), spec)
    assert len(out) == 200
    assert {"delta_cost", "delta_qaly", "nmb"}.issubset(set(out.columns))


def test_tornado_has_expected_columns() -> None:
    table = one_way_sensitivity(
        _base_params(),
        ranges={"sensitivity": (0.70, 0.90), "specificity": (0.75, 0.95)},
        wtp=10000.0,
    )
    assert {"parameter", "low_nmb", "high_nmb", "swing"}.issubset(set(table.columns))
    assert np.all(table["swing"] >= 0.0)


def test_ceac_curve_probability_range() -> None:
    delta_cost = np.array([-50.0, 20.0, -10.0, 30.0])
    delta_qaly = np.array([0.02, 0.01, 0.00, 0.03])
    curve = ceac_curve(delta_cost, delta_qaly, wtps=[0.0, 1000.0, 5000.0])
    assert np.all(curve["p_cost_effective"].between(0.0, 1.0))
