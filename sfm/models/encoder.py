"""Swin-based retinal image encoder used by SFM-M2.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from sfm._optional import import_torch


@dataclass(frozen=True)
class SwinEncoderConfig:
    """Configuration for the Swin visual backbone.

    Parameters
    ----------
    model_name : str
        Backbone identifier understood by ``timm.create_model``.
    pretrained : bool
        Whether to initialize from timm ImageNet weights.
    out_dim : int
        Output embedding dimension for downstream heads.
    drop_path_rate : float
        Stochastic depth rate used by the backbone.
    checkpoint_name : str or None
        Optional checkpoint file name resolved under ``SFM_CKPT_DIR``.
    """

    model_name: str = "swin_base_patch4_window7_224"
    pretrained: bool = False
    out_dim: int = 1024
    drop_path_rate: float = 0.1
    checkpoint_name: str | None = None


class SwinEncoder:
    """Torch module wrapper for Swin feature extraction.

    Parameters
    ----------
    config : SwinEncoderConfig, optional
        Runtime configuration for backbone creation.

    Notes
    -----
    The implementation follows the released SFM-M2 setup: a Swin visual
    trunk with pooled features projected to a fixed embedding dimension.
    """

    def __init__(self, config: SwinEncoderConfig | None = None) -> None:
        self.config = config or SwinEncoderConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn
        self.backbone = self._build_backbone()
        self.feature_dim = int(getattr(self.backbone, "num_features", self.config.out_dim))
        self.proj = nn.Identity() if self.feature_dim == self.config.out_dim else nn.Linear(self.feature_dim, self.config.out_dim)

    def _build_backbone(self) -> Any:
        """Create the Swin backbone and load optional checkpoint.

        Returns
        -------
        Any
            Instantiated timm model.
        """

        try:
            import timm
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("timm is required for SwinEncoder. Install extras: `pip install sfm[torch]`.") from exc

        backbone = timm.create_model(
            self.config.model_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=self.config.drop_path_rate,
        )
        if self.config.checkpoint_name:
            ckpt_dir = Path(os.getenv("SFM_CKPT_DIR", "."))
            ckpt_path = ckpt_dir / self.config.checkpoint_name
            if ckpt_path.exists():
                self._load_checkpoint(backbone, ckpt_path)
        return backbone

    def _load_checkpoint(self, backbone: Any, checkpoint_path: Path) -> None:
        """Load weights with permissive key matching.

        Parameters
        ----------
        backbone : Any
            Torch model receiving the checkpoint weights.
        checkpoint_path : pathlib.Path
            Checkpoint path.

        Returns
        -------
        None
            Loads matching parameters into ``backbone``.
        """

        checkpoint = self._torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "teacher" in state:
            state = state["teacher"]

        model_state = backbone.state_dict()
        matched: dict[str, Any] = {}
        for key, value in state.items():
            key_norm = key
            if key_norm.startswith("module."):
                key_norm = key_norm[len("module.") :]
            if key_norm.startswith("backbone."):
                key_norm = key_norm[len("backbone.") :]
            if key_norm in model_state and model_state[key_norm].shape == value.shape:
                matched[key_norm] = value

        model_state.update(matched)
        backbone.load_state_dict(model_state)

    def parameters(self):
        """Return encoder parameters for optimizer construction.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        for p in self.backbone.parameters():
            yield p
        for p in self.proj.parameters():
            yield p

    def train(self, mode: bool = True):
        """Set module training mode.

        Parameters
        ----------
        mode : bool
            Desired training mode.

        Returns
        -------
        SwinEncoder
            Self.
        """

        self.backbone.train(mode)
        self.proj.train(mode)
        return self

    def eval(self):
        """Set module to eval mode.

        Returns
        -------
        SwinEncoder
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
        SwinEncoder
            Self.
        """

        self.backbone.to(device)
        self.proj.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Serialize encoder state.

        Returns
        -------
        dict
            State dictionary.
        """

        return {"backbone": self.backbone.state_dict(), "proj": self.proj.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load encoder state.

        Parameters
        ----------
        state : dict
            Serialized state dictionary.

        Returns
        -------
        None
            Updates module parameters.
        """

        if "backbone" in state:
            self.backbone.load_state_dict(state["backbone"], strict=False)
        if "proj" in state:
            self.proj.load_state_dict(state["proj"], strict=False)

    def _extract_tokens(self, images: Any) -> Any:
        """Extract token-level features from the backbone.

        Parameters
        ----------
        images : Any
            Input tensor with shape ``(N, C, H, W)``.

        Returns
        -------
        Any
            Token tensor with shape ``(N, T, D)``.
        """

        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(images)
        else:
            feats = self.backbone(images)

        if isinstance(feats, tuple):
            feats = feats[0]

        if feats.ndim == 4:
            feats = feats.flatten(2).transpose(1, 2)
        elif feats.ndim == 2:
            feats = feats.unsqueeze(1)
        elif feats.ndim != 3:
            raise ValueError(f"Unsupported feature shape from backbone: {tuple(feats.shape)}")
        return feats

    def encode_with_tokens(self, images: Any) -> tuple[Any, Any]:
        """Encode images and return pooled + token features.

        Parameters
        ----------
        images : Any
            Batch tensor ``(N, C, H, W)``.

        Returns
        -------
        tuple
            ``(pooled_embeddings, token_embeddings)``.
        """

        tokens = self._extract_tokens(images)
        pooled = tokens.mean(dim=1)
        projected = self.proj(pooled)
        return projected, tokens

    def __call__(self, images: Any) -> Any:
        """Encode images into pooled embeddings.

        Parameters
        ----------
        images : Any
            Batch tensor ``(N, C, H, W)``.

        Returns
        -------
        Any
            Embedding matrix ``(N, out_dim)``.
        """

        pooled, _ = self.encode_with_tokens(images)
        return pooled
