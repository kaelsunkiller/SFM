"""Biomarker grading heads for eGFR, HbA1c, and triglycerides.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm._optional import import_torch


BIOMARKERS = ("eGFR", "HbA1c", "TG")
ABNORMAL_CLASS_START = {"eGFR": 4, "HbA1c": 2, "TG": 3}


@dataclass(frozen=True)
class BiomarkerConfig:
    """Configuration for biomarker multi-head classification.

    Parameters
    ----------
    input_dim : int
        Feature dimension from the visual encoder.
    hidden_dim : int
        Shared hidden layer width.
    class_counts : tuple of int
        Number of classes for ``(eGFR, HbA1c, TG)``.
    dropout : float
        Dropout probability in the shared trunk.
    """

    input_dim: int = 1024
    hidden_dim: int = 512
    class_counts: tuple[int, int, int] = (5, 3, 4)
    dropout: float = 0.2


class BiomarkerHead:
    """Shared-trunk, multi-head biomarker classifier.

    Parameters
    ----------
    config : BiomarkerConfig, optional
        Head hyperparameters.

    Notes
    -----
    The three biomarker heads share the same visual trunk projection and
    each is optimized with task-specific cross-entropy.
    """

    def __init__(self, config: BiomarkerConfig | None = None) -> None:
        self.config = config or BiomarkerConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        self.norm = nn.LayerNorm(self.config.input_dim)
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.config.dropout)

        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(self.config.hidden_dim, n_cls)
                for name, n_cls in zip(BIOMARKERS, self.config.class_counts)
            }
        )

    def parameters(self):
        """Return trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        yield from self.norm.parameters()
        yield from self.fc1.parameters()
        yield from self.heads.parameters()

    def train(self, mode: bool = True):
        """Set training mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        BiomarkerHead
            Self.
        """

        self.norm.train(mode)
        self.fc1.train(mode)
        self.act.train(mode)
        self.drop.train(mode)
        self.heads.train(mode)
        return self

    def eval(self):
        """Set eval mode.

        Returns
        -------
        BiomarkerHead
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
        BiomarkerHead
            Self.
        """

        self.norm.to(device)
        self.fc1.to(device)
        self.heads.to(device)
        return self

    def _pool_features(self, features: Any) -> Any:
        """Pool token features when available.

        Parameters
        ----------
        features : Any
            Tensor ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        Any
            Pooled tensor ``(B, D)``.
        """

        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected features with rank 2 or 3, got {tuple(features.shape)}")

    def _shared(self, features: Any) -> Any:
        """Compute shared hidden representation.

        Parameters
        ----------
        features : Any
            Input features.

        Returns
        -------
        Any
            Shared hidden features.
        """

        x = self._pool_features(features)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

    def logits(self, features: Any) -> dict[str, Any]:
        """Compute per-biomarker logits.

        Parameters
        ----------
        features : Any
            Encoder features.

        Returns
        -------
        dict
            Logits dictionary keyed by biomarker name.
        """

        shared = self._shared(features)
        return {name: head(shared) for name, head in self.heads.items()}

    def __call__(self, features: Any, return_logits: bool = False) -> dict[str, Any]:
        """Predict per-biomarker class probabilities.

        Parameters
        ----------
        features : Any
            Encoder features.
        return_logits : bool
            If true, return logits.

        Returns
        -------
        dict
            Probabilities or logits per biomarker.
        """

        logits = self.logits(features)
        if return_logits:
            return logits
        return {name: self._torch.softmax(values, dim=-1) for name, values in logits.items()}

    def compute_loss(self, logits: dict[str, Any], targets: dict[str, Any]) -> Any:
        """Compute mean cross-entropy across biomarker heads.

        Parameters
        ----------
        logits : dict
            Predicted logits per biomarker.
        targets : dict
            Integer labels per biomarker.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        loss_terms = []
        criterion = self._nn.CrossEntropyLoss()
        for name in BIOMARKERS:
            loss_terms.append(criterion(logits[name], targets[name].long()))
        return sum(loss_terms) / len(loss_terms)

    def abnormal_probability(self, probs: dict[str, Any], biomarker: str) -> Any:
        """Convert multi-class probabilities to abnormal-class probability.

        Parameters
        ----------
        probs : dict
            Probability dictionary from ``__call__``.
        biomarker : str
            One of ``eGFR``, ``HbA1c``, ``TG``.

        Returns
        -------
        Any
            Binary abnormal probability.
        """

        if biomarker not in ABNORMAL_CLASS_START:
            raise ValueError(f"Unsupported biomarker: {biomarker}")
        start_idx = ABNORMAL_CLASS_START[biomarker]
        return probs[biomarker][:, start_idx:].sum(dim=-1)

    def binary_auroc(self, probs: dict[str, Any], labels: Any, biomarker: str) -> float:
        """Compute binary AUROC for normal-versus-abnormal evaluation.

        Parameters
        ----------
        probs : dict
            Probability dictionary.
        labels : Any
            Integer biomarker labels.
        biomarker : str
            Biomarker name.

        Returns
        -------
        float
            Binary AUROC.
        """

        from sklearn.metrics import roc_auc_score

        abnormal_prob = self.abnormal_probability(probs, biomarker)
        abnormal_prob = abnormal_prob.detach().cpu().numpy() if hasattr(abnormal_prob, "detach") else np.asarray(abnormal_prob)
        y = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
        y_binary = (y >= ABNORMAL_CLASS_START[biomarker]).astype(int)
        return float(roc_auc_score(y_binary, abnormal_prob))
