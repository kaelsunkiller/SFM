"""One-way sensitivity analysis for CEA decision models.

Reference: Methods §Cost-effectiveness analysis.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .decision_model import DecisionModelParameters, evaluate_no_screening, evaluate_screening


def one_way_sensitivity(
    base: DecisionModelParameters,
    ranges: dict[str, tuple[float, float]],
    wtp: float,
) -> pd.DataFrame:
    """Compute one-way sensitivity swings for each parameter.

    Parameters
    ----------
    base : DecisionModelParameters
        Baseline model parameters.
    ranges : dict
        Mapping from parameter name to ``(low, high)``.
    wtp : float
        Willingness-to-pay threshold.

    Returns
    -------
    pandas.DataFrame
        Sensitivity table sorted by NMB swing.
    """

    rows: list[dict[str, float]] = []
    for key, (low, high) in ranges.items():
        low_params = replace(base, **{key: low})
        high_params = replace(base, **{key: high})

        low_scr = evaluate_screening(low_params)
        low_nos = evaluate_no_screening(low_params)
        high_scr = evaluate_screening(high_params)
        high_nos = evaluate_no_screening(high_params)

        nmb_low = (low_scr.qaly - low_nos.qaly) * wtp - (low_scr.cost - low_nos.cost)
        nmb_high = (high_scr.qaly - high_nos.qaly) * wtp - (high_scr.cost - high_nos.cost)
        rows.append(
            {
                "parameter": key,
                "low_nmb": nmb_low,
                "high_nmb": nmb_high,
                "swing": abs(nmb_high - nmb_low),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("swing", ascending=False).reset_index(drop=True)
