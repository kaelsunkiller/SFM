"""Generic ophthalmic-transfer classifier for public benchmarks.

Reference: Methods §Ophthalmic disease diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm._optional import import_torch


OPHTHALMIC_CLASS_COUNTS = {
    "ophthalmic_aptos": 5,
    "ophthalmic_glaucoma": 3,
    "ophthalmic_idrid": 5,
    "ophthalmic_jsiec": 39,
    "ophthalmic_messidor2": 5,
    "ophthalmic_odir5k": 8,
    "ophthalmic_papila": 3,
    "ophthalmic_retina": 4,
}


@dataclass(frozen=True)
class OphthalmicTransferConfig:
    """Configuration for ophthalmic-transfer classification.

    Parameters
    ----------
    input_dim : int
        Visual feature dimension.
    hidden_dim : int
        Hidden layer width.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability.
    """

    input_dim: int = 1024
    hidden_dim: int = 512
    num_classes: int = 2
    dropout: float = 0.2


class OphthalmicTransferHead:
    """Classifier head reused across eight ophthalmic tasks.

    Parameters
    ----------
    config : OphthalmicTransferConfig, optional
        Head hyperparameters.

    Notes
    -----
    This head supports all ophthalmic transfer tasks by changing only the
    final class count in configuration.
    """

    def __init__(self, config: OphthalmicTransferConfig | None = None) -> None:
        self.config = config or OphthalmicTransferConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        self.norm = nn.LayerNorm(self.config.input_dim)
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.hidden_dim, self.config.num_classes)

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
        OphthalmicTransferHead
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
        OphthalmicTransferHead
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
        OphthalmicTransferHead
            Self.
        """

        self.norm.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        return self

    @staticmethod
    def _pool_features(features: Any) -> Any:
        """Pool token-level features when provided.

        Parameters
        ----------
        features : Any
            Tensor ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        Any
            Pooled feature matrix.
        """

        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected features with rank 2 or 3, got {tuple(features.shape)}")

    def logits(self, features: Any) -> Any:
        """Compute class logits.

        Parameters
        ----------
        features : Any
            Encoder features.

        Returns
        -------
        Any
            Logits matrix.
        """

        x = self._pool_features(features)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def __call__(self, features: Any, return_logits: bool = False) -> Any:
        """Predict class probabilities.

        Parameters
        ----------
        features : Any
            Encoder features.
        return_logits : bool
            If true, return logits.

        Returns
        -------
        Any
            Probabilities or logits.
        """

        logits = self.logits(features)
        if return_logits:
            return logits
        return self._torch.softmax(logits, dim=-1)

    def compute_loss(self, logits: Any, labels: Any) -> Any:
        """Compute cross-entropy loss.

        Parameters
        ----------
        logits : Any
            Predicted logits.
        labels : Any
            Integer class labels.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        return self._nn.CrossEntropyLoss()(logits, labels.long())

    def macro_auroc_ovr(self, probs: Any, labels: Any) -> float:
        """Compute one-vs-rest macro AUROC.

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


def config_for_task(task_name: str, input_dim: int = 1024) -> OphthalmicTransferConfig:
    """Create a task-specific configuration.

    Parameters
    ----------
    task_name : str
        Canonical ophthalmic task name.
    input_dim : int
        Input feature dimension.

    Returns
    -------
    OphthalmicTransferConfig
        Config with dataset-specific class count.
    """

    if task_name not in OPHTHALMIC_CLASS_COUNTS:
        raise ValueError(f"Unsupported task: {task_name}")
    return OphthalmicTransferConfig(input_dim=input_dim, num_classes=OPHTHALMIC_CLASS_COUNTS[task_name])
