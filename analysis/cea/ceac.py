"""Cost-effectiveness acceptability curve utilities.

Reference: Methods §Cost-effectiveness analysis.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def ceac_curve(delta_cost: np.ndarray, delta_qaly: np.ndarray, wtps: Iterable[float]) -> pd.DataFrame:
    """Compute CEAC probabilities across willingness-to-pay thresholds.

    Parameters
    ----------
    delta_cost : numpy.ndarray
        Incremental cost array.
    delta_qaly : numpy.ndarray
        Incremental QALY array.
    wtps : iterable of float
        Willingness-to-pay thresholds.

    Returns
    -------
    pandas.DataFrame
        Columns ``wtp`` and ``p_cost_effective``.
    """

    dc = np.asarray(delta_cost, dtype=float)
    dq = np.asarray(delta_qaly, dtype=float)

    rows: list[dict[str, float]] = []
    for wtp in wtps:
        nmb = dq * float(wtp) - dc
        rows.append({"wtp": float(wtp), "p_cost_effective": float(np.mean(nmb > 0.0))})
    return pd.DataFrame(rows)
