"""Four-class CKM staging head and evaluation helpers.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm._optional import import_torch


@dataclass(frozen=True)
class CKMStagingConfig:
    """Configuration for CKM staging classification.

    Parameters
    ----------
    input_dim : int
        Feature dimension from the visual encoder.
    hidden_dim : int
        Hidden layer width.
    num_stages : int
        Number of stage classes.
    dropout : float
        Dropout probability before output projection.
    """

    input_dim: int = 1024
    hidden_dim: int = 512
    num_stages: int = 4
    dropout: float = 0.2


class CKMStagingHead:
    """Multi-class classifier for CKM stage prediction.

    Parameters
    ----------
    config : CKMStagingConfig, optional
        Classifier hyperparameters.

    Notes
    -----
    Stage probabilities are computed with softmax and optimized with
    cross-entropy loss.
    """

    def __init__(self, config: CKMStagingConfig | None = None) -> None:
        self.config = config or CKMStagingConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        self.norm = nn.LayerNorm(self.config.input_dim)
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.hidden_dim, self.config.num_stages)

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
        CKMStagingHead
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
        CKMStagingHead
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
        CKMStagingHead
            Self.
        """

        self.norm.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        return self

    def _pool_features(self, features: Any) -> Any:
        """Pool token-level features when available.

        Parameters
        ----------
        features : Any
            Tensor ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        Any
            Pooled matrix ``(B, D)``.
        """

        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected features with rank 2 or 3, got {tuple(features.shape)}")

    def logits(self, features: Any) -> Any:
        """Compute stage logits.

        Parameters
        ----------
        features : Any
            Encoder features.

        Returns
        -------
        Any
            Logits with shape ``(B, num_stages)``.
        """

        x = self._pool_features(features)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def __call__(self, features: Any, return_logits: bool = False) -> Any:
        """Predict CKM stage probabilities.

        Parameters
        ----------
        features : Any
            Encoder features.
        return_logits : bool
            If true, return logits.

        Returns
        -------
        Any
            Stage probabilities or logits.
        """

        logits = self.logits(features)
        if return_logits:
            return logits
        return self._torch.softmax(logits, dim=-1)

    def compute_loss(self, logits: Any, targets: Any) -> Any:
        """Compute cross-entropy loss.

        Parameters
        ----------
        logits : Any
            Predicted logits.
        targets : Any
            Integer stage labels.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        return self._nn.CrossEntropyLoss()(logits, targets.long())

    def stage_ge2_score(self, probs: Any) -> Any:
        """Compute stage-2-or-higher probability.

        Parameters
        ----------
        probs : Any
            Stage probability matrix.

        Returns
        -------
        Any
            Binary-risk score ``P(stage >= 2)``.
        """

        return probs[:, 2] + probs[:, 3]

    def macro_auroc_ovr(self, probs: Any, labels: Any) -> float:
        """Compute OVR macro AUROC.

        Parameters
        ----------
        probs : Any
            Probability matrix.
        labels : Any
            Integer labels.

        Returns
        -------
        float
            Macro AUROC.
        """

        from sklearn.metrics import roc_auc_score

        p = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
        y = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
