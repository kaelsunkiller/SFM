"""Progression-risk head with temporal and baseline-label fusion.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sfm._optional import import_torch


def _as_time_column(time_interval: Any, torch_module: Any, device: Any) -> Any:
    """Convert time input into a float tensor column.

    Parameters
    ----------
    time_interval : Any
        Scalar, list, or tensor time intervals.
    torch_module : Any
        Torch module namespace.
    device : Any
        Target device.

    Returns
    -------
    Any
        Tensor of shape ``(N, 1)``.
    """

    if torch_module.is_tensor(time_interval):
        t = time_interval.float()
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return t.to(device)
    if isinstance(time_interval, (list, tuple)):
        values = [float(x) for x in time_interval]
        return torch_module.tensor(values, dtype=torch_module.float32, device=device).unsqueeze(1)
    return torch_module.tensor([[float(time_interval)]], dtype=torch_module.float32, device=device)


@dataclass(frozen=True)
class ProgressionConfig:
    """Configuration for progression prediction.

    Parameters
    ----------
    input_dim : int
        Visual feature dimension.
    output_dim : int
        Number of progression outputs.
    use_baseline_labels : bool
        Whether to fuse baseline labels with temporal features.
    dropout : float
        Dropout in fusion and prediction blocks.
    """

    input_dim: int = 1024
    output_dim: int = 6
    use_baseline_labels: bool = False
    dropout: float = 0.1


class _PrototypeFusion:
    """Temporal-label prototype fusion block.

    Parameters
    ----------
    label_dim : int
        Baseline-label dimension.
    embed_dim : int
        Visual embedding dimension.
    dropout : float
        Dropout probability.
    use_baseline_labels : bool
        If false, only temporal signal is used.
    """

    def __init__(self, label_dim: int, embed_dim: int, dropout: float, use_baseline_labels: bool) -> None:
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn
        self.use_baseline_labels = use_baseline_labels

        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.label_mlp = None
        if use_baseline_labels:
            self.label_mlp = nn.Sequential(
                nn.Linear(label_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

    def parameters(self):
        """Return trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        yield from self.time_mlp.parameters()
        if self.label_mlp is not None:
            yield from self.label_mlp.parameters()
        yield from self.gate.parameters()

    def train(self, mode: bool = True):
        """Set mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        _PrototypeFusion
            Self.
        """

        self.time_mlp.train(mode)
        if self.label_mlp is not None:
            self.label_mlp.train(mode)
        self.gate.train(mode)
        return self

    def to(self, device: Any):
        """Move to device.

        Parameters
        ----------
        device : Any
            Torch device.

        Returns
        -------
        _PrototypeFusion
            Self.
        """

        self.time_mlp.to(device)
        if self.label_mlp is not None:
            self.label_mlp.to(device)
        self.gate.to(device)
        return self

    def __call__(self, image_embed: Any, time_interval: Any, baseline_labels: Any | None = None) -> Any:
        """Fuse temporal prototype into visual embedding.

        Parameters
        ----------
        image_embed : Any
            Visual embeddings ``(N, D)``.
        time_interval : Any
            Time interval input.
        baseline_labels : Any, optional
            Baseline labels ``(N, C)``.

        Returns
        -------
        Any
            Fused embeddings ``(N, D)``.
        """

        device = image_embed.device
        time_vec = self.time_mlp(_as_time_column(time_interval, self._torch, device))
        proto = time_vec
        if self.label_mlp is not None:
            if baseline_labels is None:
                raise ValueError("Baseline labels are required when use_baseline_labels=True.")
            proto = proto + self.label_mlp(baseline_labels.float().to(device))
        gate = self.gate(proto)
        return image_embed + gate * proto


class ProgressionHead:
    """Prediction head for CKM progression risk.

    Parameters
    ----------
    config : ProgressionConfig, optional
        Head hyperparameters.

    Notes
    -----
    The head supports temporal fusion and optional baseline-label fusion,
    then predicts progression logits for each target outcome.
    """

    def __init__(self, config: ProgressionConfig | None = None) -> None:
        self.config = config or ProgressionConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        self.norm = nn.LayerNorm(self.config.input_dim)
        self.fusion = _PrototypeFusion(
            label_dim=self.config.output_dim,
            embed_dim=self.config.input_dim,
            dropout=self.config.dropout,
            use_baseline_labels=self.config.use_baseline_labels,
        )
        self.drop = nn.Dropout(self.config.dropout)
        self.head = nn.Linear(self.config.input_dim, self.config.output_dim)

    def parameters(self):
        """Return trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        yield from self.norm.parameters()
        yield from self.fusion.parameters()
        yield from self.head.parameters()

    def train(self, mode: bool = True):
        """Set training mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        ProgressionHead
            Self.
        """

        self.norm.train(mode)
        self.fusion.train(mode)
        self.drop.train(mode)
        self.head.train(mode)
        return self

    def eval(self):
        """Set eval mode.

        Returns
        -------
        ProgressionHead
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
        ProgressionHead
            Self.
        """

        self.norm.to(device)
        self.fusion.to(device)
        self.head.to(device)
        return self

    @staticmethod
    def _pool_features(features: Any) -> Any:
        """Pool token features when needed.

        Parameters
        ----------
        features : Any
            Tensor ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        Any
            Pooled ``(B, D)`` tensor.
        """

        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected features with rank 2 or 3, got {tuple(features.shape)}")

    def logits(self, features: Any, time_interval: Any | None = None, baseline_labels: Any | None = None) -> Any:
        """Compute progression logits.

        Parameters
        ----------
        features : Any
            Encoder features.
        time_interval : Any, optional
            Time interval feature.
        baseline_labels : Any, optional
            Baseline labels for fusion.

        Returns
        -------
        Any
            Logit matrix ``(B, output_dim)``.
        """

        pooled = self._pool_features(features)
        pooled = self.norm(pooled)

        if time_interval is not None or self.config.use_baseline_labels:
            if time_interval is None:
                raise ValueError("time_interval is required for progression fusion.")
            pooled = self.fusion(pooled, time_interval=time_interval, baseline_labels=baseline_labels)

        pooled = self.drop(pooled)
        return self.head(pooled)

    def __call__(
        self,
        features: Any,
        time_interval: Any | None = None,
        baseline_labels: Any | None = None,
        return_logits: bool = False,
    ) -> Any:
        """Predict progression probabilities.

        Parameters
        ----------
        features : Any
            Encoder features.
        time_interval : Any, optional
            Time interval input.
        baseline_labels : Any, optional
            Baseline labels.
        return_logits : bool
            If true, return logits.

        Returns
        -------
        Any
            Probabilities or logits.
        """

        logits = self.logits(features, time_interval=time_interval, baseline_labels=baseline_labels)
        if return_logits:
            return logits
        return self._torch.sigmoid(logits)

    def compute_loss(self, logits: Any, targets: Any) -> Any:
        """Compute binary cross-entropy loss.

        Parameters
        ----------
        logits : Any
            Predicted logits.
        targets : Any
            Target binary matrix.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        criterion = self._nn.BCEWithLogitsLoss()
        return criterion(logits, targets.float())
