"""Evaluation metrics used in the manuscript analyses.

Reference: Methods §Statistical analysis and §Cost-effectiveness analysis.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate_binary_target(y_true: np.ndarray) -> None:
    """Validate binary target labels.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary label vector.

    Returns
    -------
    None
        Raises when labels are not binary.
    """

    uniq = np.unique(y_true)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError("Expected binary labels encoded as 0/1.")


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute one-vs-rest macro AUROC.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer class labels with shape ``(N,)``.
    y_score : numpy.ndarray
        Score matrix ``(N, C)`` or binary score vector ``(N,)``.

    Returns
    -------
    float
        Macro AUROC over represented classes.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        _validate_binary_target(y_true)
        return float(roc_auc_score(y_true, y_score))

    classes = list(range(y_score.shape[1]))
    aucs: list[float] = []
    for cls in classes:
        y_bin = (y_true == cls).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_bin, y_score[:, cls])))
    if not aucs:
        raise ValueError("AUROC is undefined because no class has both positive and negative examples.")
    return float(np.mean(aucs))


def macro_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute one-vs-rest macro AUPR.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer class labels.
    y_score : numpy.ndarray
        Score matrix or binary score vector.

    Returns
    -------
    float
        Macro AUPR over represented classes.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        _validate_binary_target(y_true)
        return float(average_precision_score(y_true, y_score))

    classes = list(range(y_score.shape[1]))
    aps: list[float] = []
    for cls in classes:
        y_bin = (y_true == cls).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        aps.append(float(average_precision_score(y_bin, y_score[:, cls])))
    if not aps:
        raise ValueError("AUPR is undefined because no class has both positive and negative examples.")
    return float(np.mean(aps))


def nri_idi(y_true: np.ndarray, base_prob: np.ndarray, new_prob: np.ndarray) -> dict[str, float]:
    """Compute category-free NRI and IDI for binary outcomes.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels in ``{0, 1}``.
    base_prob : numpy.ndarray
        Baseline model probability of the positive outcome.
    new_prob : numpy.ndarray
        New model probability of the positive outcome.

    Returns
    -------
    dict
        Dictionary with keys ``nri_event``, ``nri_nonevent``, ``nri``, and ``idi``.
    """

    y_true = np.asarray(y_true).astype(int)
    base_prob = np.asarray(base_prob, dtype=float)
    new_prob = np.asarray(new_prob, dtype=float)
    _validate_binary_target(y_true)

    event = y_true == 1
    nonevent = y_true == 0

    if event.sum() == 0 or nonevent.sum() == 0:
        raise ValueError("NRI/IDI requires both event and non-event groups.")

    up_event = np.mean(new_prob[event] > base_prob[event])
    down_event = np.mean(new_prob[event] < base_prob[event])
    up_nonevent = np.mean(new_prob[nonevent] > base_prob[nonevent])
    down_nonevent = np.mean(new_prob[nonevent] < base_prob[nonevent])

    nri_event = float(up_event - down_event)
    nri_nonevent = float(down_nonevent - up_nonevent)
    nri = float(nri_event + nri_nonevent)

    idi_base = float(np.mean(base_prob[event]) - np.mean(base_prob[nonevent]))
    idi_new = float(np.mean(new_prob[event]) - np.mean(new_prob[nonevent]))
    idi = float(idi_new - idi_base)

    return {
        "nri_event": nri_event,
        "nri_nonevent": nri_nonevent,
        "nri": nri,
        "idi": idi,
    }


def decision_curve_table(
    y_true: np.ndarray,
    score: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Compute decision-curve net benefit over thresholds.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    score : numpy.ndarray
        Positive-class probabilities.
    thresholds : iterable of float, optional
        Threshold set. Defaults to ``0.01..0.99``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``threshold``, ``net_benefit``, ``treat_all``, ``treat_none``.
    """

    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    _validate_binary_target(y_true)

    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    n = len(y_true)
    prevalence = float(np.mean(y_true == 1))

    rows: list[dict[str, float]] = []
    for t in thresholds:
        if t <= 0.0 or t >= 1.0:
            continue
        pred_pos = score >= t
        tp = float(np.sum((pred_pos == 1) & (y_true == 1)))
        fp = float(np.sum((pred_pos == 1) & (y_true == 0)))
        odds = t / (1.0 - t)
        net_benefit = tp / n - (fp / n) * odds
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": float(t),
                "net_benefit": float(net_benefit),
                "treat_all": float(treat_all),
                "treat_none": 0.0,
            }
        )

    return pd.DataFrame(rows)


def c_index(time_to_event: np.ndarray, event_observed: np.ndarray, risk_score: np.ndarray) -> float:
    """Compute concordance index for right-censored outcomes.

    Parameters
    ----------
    time_to_event : numpy.ndarray
        Follow-up durations.
    event_observed : numpy.ndarray
        Event indicators (1 for observed event, 0 for censored).
    risk_score : numpy.ndarray
        Higher score indicates higher risk.

    Returns
    -------
    float
        Concordance index in ``[0, 1]``.
    """

    t = np.asarray(time_to_event, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    r = np.asarray(risk_score, dtype=float)
    _validate_binary_target(e)

    concordant = 0.0
    comparable = 0.0
    n = len(t)

    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] and e[i] == e[j] == 0:
                continue

            if t[i] < t[j] and e[i] == 1:
                comparable += 1.0
                if r[i] > r[j]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5
            elif t[j] < t[i] and e[j] == 1:
                comparable += 1.0
                if r[j] > r[i]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5

    if comparable == 0:
        raise ValueError("No comparable pairs for C-index computation.")
    return float(concordant / comparable)
