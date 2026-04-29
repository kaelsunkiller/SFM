"""Decision-tree model for screening strategy evaluation.

Reference: Methods §Cost-effectiveness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionModelParameters:
    """Input parameters for the screening decision tree.

    Parameters
    ----------
    prevalence : float
        Baseline event prevalence.
    sensitivity : float
        Screening sensitivity.
    specificity : float
        Screening specificity.
    cost_screen : float
        Per-subject screening cost.
    cost_tp : float
        Cost in the true-positive branch.
    cost_fp : float
        Cost in the false-positive branch.
    cost_fn : float
        Cost in the false-negative branch.
    cost_tn : float
        Cost in the true-negative branch.
    qaly_tp : float
        QALY in the true-positive branch.
    qaly_fp : float
        QALY in the false-positive branch.
    qaly_fn : float
        QALY in the false-negative branch.
    qaly_tn : float
        QALY in the true-negative branch.
    """

    prevalence: float
    sensitivity: float
    specificity: float
    cost_screen: float
    cost_tp: float
    cost_fp: float
    cost_fn: float
    cost_tn: float
    qaly_tp: float
    qaly_fp: float
    qaly_fn: float
    qaly_tn: float


@dataclass(frozen=True)
class StrategyOutcome:
    """Expected outcomes from one strategy.

    Parameters
    ----------
    cost : float
        Expected cost per subject.
    qaly : float
        Expected QALY per subject.
    """

    cost: float
    qaly: float


def evaluate_screening(params: DecisionModelParameters) -> StrategyOutcome:
    """Evaluate expected cost and QALY for screening.

    Parameters
    ----------
    params : DecisionModelParameters
        Model parameters.

    Returns
    -------
    StrategyOutcome
        Expected per-subject outcomes.
    """

    p = params.prevalence
    se = params.sensitivity
    sp = params.specificity

    tp = p * se
    fn = p * (1.0 - se)
    tn = (1.0 - p) * sp
    fp = (1.0 - p) * (1.0 - sp)

    cost = (
        params.cost_screen
        + tp * params.cost_tp
        + fp * params.cost_fp
        + fn * params.cost_fn
        + tn * params.cost_tn
    )
    qaly = (
        tp * params.qaly_tp
        + fp * params.qaly_fp
        + fn * params.qaly_fn
        + tn * params.qaly_tn
    )
    return StrategyOutcome(cost=float(cost), qaly=float(qaly))


def evaluate_no_screening(params: DecisionModelParameters) -> StrategyOutcome:
    """Evaluate expected outcomes when no organised screening is used.

    Parameters
    ----------
    params : DecisionModelParameters
        Model parameters.

    Returns
    -------
    StrategyOutcome
        Expected outcomes for no-screening strategy.
    """

    p = params.prevalence
    fn = p
    tn = 1.0 - p
    cost = fn * params.cost_fn + tn * params.cost_tn
    qaly = fn * params.qaly_fn + tn * params.qaly_tn
    return StrategyOutcome(cost=float(cost), qaly=float(qaly))
