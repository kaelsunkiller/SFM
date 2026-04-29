"""Probabilistic sensitivity analysis utilities.

Reference: Methods §Cost-effectiveness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from .decision_model import DecisionModelParameters, evaluate_no_screening, evaluate_screening


@dataclass(frozen=True)
class PSASpec:
    """PSA runtime settings.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples. Manuscript uses 10,000.
    random_seed : int
        Random seed for reproducibility. Manuscript uses 42.
    wtp : float
        Willingness-to-pay threshold per QALY (USD). Default 37,653 =
        3 x China 2023 GDP per capita per the manuscript Methods
        §Cost-effectiveness analysis (World Bank 2023). The
        manuscript also reports CEAC at 1 x GDP = 12,551 USD.
    """

    n_samples: int = 10_000
    random_seed: int = 42
    wtp: float = 37_653.0


def run_psa(
    base: DecisionModelParameters,
    spec: PSASpec,
    samplers: dict[str, Callable[[np.random.Generator], float]] | None = None,
) -> pd.DataFrame:
    """Run probabilistic sensitivity analysis.

    Parameters
    ----------
    base : DecisionModelParameters
        Baseline parameter set.
    spec : PSASpec
        PSA runtime specification.
    samplers : dict, optional
        Mapping of parameter names to random samplers.

    Returns
    -------
    pandas.DataFrame
        Monte Carlo draws with incremental outcomes and NMB.
    """

    rng = np.random.default_rng(spec.random_seed)
    samplers = samplers or {}

    rows: list[dict[str, float]] = []
    for _ in range(spec.n_samples):
        sampled = base
        for key, fn in samplers.items():
            sampled = replace(sampled, **{key: float(fn(rng))})

        scr = evaluate_screening(sampled)
        nos = evaluate_no_screening(sampled)
        delta_cost = scr.cost - nos.cost
        delta_qaly = scr.qaly - nos.qaly
        nmb = delta_qaly * spec.wtp - delta_cost
        rows.append(
            {
                "cost_screening": scr.cost,
                "qaly_screening": scr.qaly,
                "cost_no_screening": nos.cost,
                "qaly_no_screening": nos.qaly,
                "delta_cost": delta_cost,
                "delta_qaly": delta_qaly,
                "nmb": nmb,
            }
        )
    return pd.DataFrame(rows)
