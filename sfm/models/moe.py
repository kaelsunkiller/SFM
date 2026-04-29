"""Sparse mixture-of-experts block for SFM-M2.

Reference: Methods §MoE architecture for image encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sfm._optional import import_torch


@dataclass(frozen=True)
class SparseMoEConfig:
    """Configuration for sparse MoE routing.

    Parameters
    ----------
    input_dim : int
        Input token dimension.
    hidden_dim : int
        Hidden dimension inside each expert feed-forward network.
    n_experts : int
        Number of *local* experts per process / GPU. The global expert
        pool seen by the gate during distributed training equals
        ``n_experts * world_size``. The released configuration uses
        ``n_experts=8`` and ``world_size=2``, giving the 16-expert
        spectrum reported in Methods §MoE architecture for image encoder.
    top_k : int
        Number of experts selected per token.
    capacity_factor : float
        Capacity multiplier used by Tutel routing.
    dropout : float
        Dropout probability inside experts.
    """

    input_dim: int = 1024
    hidden_dim: int = 4096
    n_experts: int = 8
    top_k: int = 1
    capacity_factor: float = 1.25
    dropout: float = 0.1


class SparseMoE:
    """Top-k sparse routing layer with optional Tutel acceleration.

    Parameters
    ----------
    config : SparseMoEConfig, optional
        Routing and expert hyperparameters.

    Notes
    -----
    When Tutel is available, this class uses its expert-parallel MoE layer.
    Otherwise it falls back to a deterministic torch implementation that
    keeps the same API and auxiliary load-balancing signal.
    """

    def __init__(self, config: SparseMoEConfig | None = None) -> None:
        self.config = config or SparseMoEConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        try:
            from tutel import moe as tutel_moe  # type: ignore
        except Exception:
            tutel_moe = None

        self._tutel_layer = None
        self._gate = None
        self._experts = None
        if tutel_moe is not None:
            gate_type = {
                "type": "top",
                "k": self.config.top_k,
                "capacity_factor": self.config.capacity_factor,
                "fp32_gate": True,
            }
            self._tutel_layer = tutel_moe.moe_layer(
                gate_type=gate_type,
                model_dim=self.config.input_dim,
                experts={
                    "type": "ffn",
                    "count_per_node": self.config.n_experts,
                    "hidden_size_per_expert": self.config.hidden_dim,
                    "activation_fn": lambda x: self._torch.nn.functional.gelu(x),
                },
            )
        else:
            self._gate = nn.Linear(self.config.input_dim, self.config.n_experts)
            self._experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.config.input_dim, self.config.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.config.hidden_dim, self.config.input_dim),
                    )
                    for _ in range(self.config.n_experts)
                ]
            )

    def parameters(self):
        """Return all trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        if self._tutel_layer is not None:
            yield from self._tutel_layer.parameters()
            return
        if self._gate is not None:
            yield from self._gate.parameters()
        if self._experts is not None:
            yield from self._experts.parameters()

    def train(self, mode: bool = True):
        """Set training mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        SparseMoE
            Self.
        """

        if self._tutel_layer is not None:
            self._tutel_layer.train(mode)
        if self._gate is not None:
            self._gate.train(mode)
        if self._experts is not None:
            self._experts.train(mode)
        return self

    def eval(self):
        """Set eval mode.

        Returns
        -------
        SparseMoE
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
        SparseMoE
            Self.
        """

        if self._tutel_layer is not None:
            self._tutel_layer.to(device)
        if self._gate is not None:
            self._gate.to(device)
        if self._experts is not None:
            self._experts.to(device)
        return self

    def _flatten(self, features: Any) -> tuple[Any, tuple[int, ...]]:
        """Flatten input to ``(N, D)`` tokens.

        Parameters
        ----------
        features : Any
            Input tensor with shape ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        tuple
            Flattened tensor and original shape.
        """

        if features.ndim == 2:
            return features, tuple(features.shape)
        if features.ndim == 3:
            bsz, n_tok, dim = features.shape
            return features.reshape(bsz * n_tok, dim), tuple(features.shape)
        raise ValueError(f"Expected 2D or 3D features, got shape {tuple(features.shape)}")

    def _restore(self, flat_out: Any, original_shape: tuple[int, ...]) -> Any:
        """Restore flattened output shape.

        Parameters
        ----------
        flat_out : Any
            Flattened output ``(N, D)``.
        original_shape : tuple of int
            Original shape from ``_flatten``.

        Returns
        -------
        Any
            Reshaped output tensor.
        """

        if len(original_shape) == 2:
            return flat_out
        bsz, n_tok, dim = original_shape
        return flat_out.reshape(bsz, n_tok, dim)

    def _fallback_forward(self, tokens: Any) -> tuple[Any, Any]:
        """Run fallback sparse routing without Tutel.

        Parameters
        ----------
        tokens : Any
            Flattened token matrix ``(N, D)``.

        Returns
        -------
        tuple
            ``(mixed_output, auxiliary_loss)``.
        """

        gate_logits = self._gate(tokens)
        gate_prob = self._torch.softmax(gate_logits, dim=-1)
        top_vals, top_idx = self._torch.topk(gate_prob, k=self.config.top_k, dim=-1)

        mixed = self._torch.zeros_like(tokens)
        for k_idx in range(self.config.top_k):
            idx = top_idx[:, k_idx]
            weight = top_vals[:, k_idx].unsqueeze(-1)
            expert_out = self._torch.zeros_like(tokens)
            for expert_id, expert in enumerate(self._experts):
                mask = idx == expert_id
                if mask.any():
                    expert_out[mask] = expert(tokens[mask])
            mixed = mixed + weight * expert_out

        load = gate_prob.mean(dim=0)
        aux_loss = (load * load * self.config.n_experts).sum()
        return mixed, aux_loss

    def __call__(self, features: Any) -> tuple[Any, Any]:
        """Route features through sparse experts.

        Parameters
        ----------
        features : Any
            Input tensor of shape ``(B, D)`` or ``(B, T, D)``.

        Returns
        -------
        tuple
            ``(routed_features, auxiliary_loss)``.
        """

        flat_tokens, original_shape = self._flatten(features)
        if self._tutel_layer is not None:
            routed = self._tutel_layer(flat_tokens)
            aux_loss = getattr(routed, "l_aux", self._torch.tensor(0.0, device=flat_tokens.device))
            out = routed
        else:
            out, aux_loss = self._fallback_forward(flat_tokens)

        return self._restore(out, original_shape), aux_loss
