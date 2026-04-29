"""Calibration utilities for post-hoc temperature scaling.

Reference: Methods §Calibration analysis and post-hoc temperature scaling.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax.

    Parameters
    ----------
    logits : numpy.ndarray
        Input logits.

    Returns
    -------
    numpy.ndarray
        Probability matrix.
    """

    x = logits - logits.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to logits.

    Parameters
    ----------
    logits : numpy.ndarray
        Logit matrix.
    temperature : float
        Positive scalar temperature.

    Returns
    -------
    numpy.ndarray
        Calibrated probabilities.
    """

    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    return _softmax(np.asarray(logits, dtype=float) / temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature by minimizing negative log-likelihood.

    Parameters
    ----------
    logits : numpy.ndarray
        Uncalibrated logits of shape ``(N, C)``.
    labels : numpy.ndarray
        Integer labels of shape ``(N,)``.

    Returns
    -------
    float
        Fitted scalar temperature.
    """

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def objective(x: np.ndarray) -> float:
        t = float(np.exp(x[0]))
        prob = apply_temperature(logits, t)
        nll = -np.log(prob[np.arange(len(labels)), labels] + 1e-12).mean()
        return float(nll)

    res = minimize(objective, x0=np.array([0.0]), method="Nelder-Mead")
    return float(np.exp(res.x[0]))


def expected_calibration_error(prob: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability matrix of shape ``(N, C)``.
    labels : numpy.ndarray
        True class labels.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    float
        Expected calibration error.
    """

    prob = np.asarray(prob, dtype=float)
    labels = np.asarray(labels, dtype=int)

    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += np.abs(acc - avg_conf) * (mask.sum() / n)

    return float(ece)
