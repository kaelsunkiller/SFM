"""Six-disease CKM comorbidity head and training utilities.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm._optional import import_torch


DEFAULT_DISEASES = ("CKD", "Diabetes", "Hypertension", "Stroke", "Obesity", "Cardiopathy")


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor-like data to NumPy.

    Parameters
    ----------
    x : Any
        Input array or tensor.

    Returns
    -------
    numpy.ndarray
        Converted array.
    """

    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass(frozen=True)
class CKMComorbidityConfig:
    """Configuration for six-label comorbidity prediction.

    Parameters
    ----------
    input_dim : int
        Input feature dimension from the visual encoder.
    hidden_dim : int
        Hidden layer width of the prediction head.
    dropout : float
        Dropout probability before the output layer.
    diseases : tuple of str
        Ordered disease names.
    """

    input_dim: int = 1024
    hidden_dim: int = 512
    dropout: float = 0.2
    diseases: tuple[str, ...] = DEFAULT_DISEASES


class CKMComorbidityHead:
    """Multi-label classification head for CKM comorbidity screening.

    Parameters
    ----------
    config : CKMComorbidityConfig, optional
        Head hyperparameters.

    Notes
    -----
    The head predicts six disease probabilities with sigmoid activation
    and is optimized with binary cross-entropy over each disease label.
    """

    def __init__(self, config: CKMComorbidityConfig | None = None) -> None:
        self.config = config or CKMComorbidityConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        self.norm = nn.LayerNorm(self.config.input_dim)
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.hidden_dim, len(self.config.diseases))

    def parameters(self):
        """Return trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        yield from self.norm.parameters()
        yield from self.fc1.parameters()
        yield from self.fc2.parameters()

    def train(self, mode: bool = True):
        """Set training mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        CKMComorbidityHead
            Self.
        """

        self.norm.train(mode)
        self.fc1.train(mode)
        self.act.train(mode)
        self.drop.train(mode)
        self.fc2.train(mode)
        return self

    def eval(self):
        """Set eval mode.

        Returns
        -------
        CKMComorbidityHead
            Self.
        """

        return self.train(False)

    def to(self, device: Any):
        """Move module to device.

        Parameters
        ----------
        device : Any
            Torch device.

        Returns
        -------
        CKMComorbidityHead
            Self.
        """

        self.norm.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        return self

    def _pool_features(self, features: Any) -> Any:
        """Pool token features when needed.

        Parameters
        ----------
        features : Any
            Tensor ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        Any
            Pooled feature matrix ``(B, D)``.
        """

        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected features with rank 2 or 3, got {tuple(features.shape)}")

    def logits(self, features: Any) -> Any:
        """Compute raw disease logits.

        Parameters
        ----------
        features : Any
            Encoder features.

        Returns
        -------
        Any
            Logits with shape ``(B, 6)``.
        """

        pooled = self._pool_features(features)
        x = self.norm(pooled)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def __call__(self, features: Any, return_logits: bool = False) -> Any:
        """Predict comorbidity probabilities.

        Parameters
        ----------
        features : Any
            Encoder features.
        return_logits : bool
            If true, return logits instead of probabilities.

        Returns
        -------
        Any
            Probabilities or logits.
        """

        logits = self.logits(features)
        if return_logits:
            return logits
        return self._torch.sigmoid(logits)

    def compute_loss(self, logits: Any, targets: Any, pos_weight: Any | None = None) -> Any:
        """Compute binary cross-entropy loss.

        Parameters
        ----------
        logits : Any
            Predicted logits ``(B, 6)``.
        targets : Any
            Binary targets ``(B, 6)``.
        pos_weight : Any, optional
            Class-wise positive weight tensor.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        criterion = self._nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion(logits, targets.float())

    def macro_auroc(self, probs: Any, labels: Any) -> float:
        """Compute macro AUROC across six diseases.

        Parameters
        ----------
        probs : Any
            Probability matrix.
        labels : Any
            Binary label matrix.

        Returns
        -------
        float
            Mean AUROC over valid classes.
        """

        from sklearn.metrics import roc_auc_score

        p = _to_numpy(probs)
        y = _to_numpy(labels).astype(int)
        scores: list[float] = []
        for idx in range(y.shape[1]):
            if np.unique(y[:, idx]).size < 2:
                continue
            scores.append(float(roc_auc_score(y[:, idx], p[:, idx])))
        return float(np.mean(scores)) if scores else float("nan")
