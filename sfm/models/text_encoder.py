"""BERT-based text encoder for SFM-M2 image-text alignment.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sfm._optional import import_torch


@dataclass(frozen=True)
class BertTextEncoderConfig:
    """Configuration for text encoding and projection.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    output_dim : int
        Output embedding size used for image-text alignment.
    max_length : int
        Maximum token length during tokenization.
    temperature : float
        Initial temperature for contrastive logits.
    """

    model_name: str = "bert-base-uncased"
    output_dim: int = 384
    max_length: int = 100
    temperature: float = 0.03


class BertTextEncoder:
    """Text encoder with contrastive projection head.

    Parameters
    ----------
    config : BertTextEncoderConfig, optional
        Runtime model configuration.

    Notes
    -----
    The released implementation mirrors the SFM-M2 pretraining stage:
    BERT sentence embeddings are projected and optimized by symmetric
    image-text contrastive loss.
    """

    def __init__(self, config: BertTextEncoderConfig | None = None) -> None:
        self.config = config or BertTextEncoderConfig()
        torch, nn = import_torch()
        self._torch = torch
        self._nn = nn

        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "transformers is required for BertTextEncoder. Install extras: `pip install sfm[torch]`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.bert = AutoModel.from_pretrained(self.config.model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, self.config.output_dim)
        init_temperature = 1.0 / max(self.config.temperature, 1e-6)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_temperature)
        self.cross_entropy = nn.CrossEntropyLoss()

    def parameters(self):
        """Return trainable parameters.

        Returns
        -------
        iterator
            Parameter iterator.
        """

        yield from self.bert.parameters()
        yield from self.proj.parameters()
        yield self.logit_scale

    def train(self, mode: bool = True):
        """Set module training mode.

        Parameters
        ----------
        mode : bool
            Desired mode.

        Returns
        -------
        BertTextEncoder
            Self.
        """

        self.bert.train(mode)
        self.proj.train(mode)
        return self

    def eval(self):
        """Set evaluation mode.

        Returns
        -------
        BertTextEncoder
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
        BertTextEncoder
            Self.
        """

        self.bert.to(device)
        self.proj.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return self

    def encode_text(self, texts: list[str], device: Any | None = None) -> Any:
        """Encode text strings into projected embeddings.

        Parameters
        ----------
        texts : list of str
            Input text samples.
        device : Any, optional
            Target device. If omitted, uses the BERT parameter device.

        Returns
        -------
        Any
            Text embedding tensor with shape ``(N, output_dim)``.
        """

        if device is None:
            device = next(self.bert.parameters()).device

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = self.bert(**tokenized)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_tokens)

    def _all_gather_with_grad(self, x: Any) -> Any:
        """Gather tensors across distributed ranks with gradient support.

        Parameters
        ----------
        x : Any
            Local rank tensor.

        Returns
        -------
        Any
            Concatenated tensor across ranks.
        """

        if not self._torch.distributed.is_available() or not self._torch.distributed.is_initialized():
            return x

        class _Gather(self._torch.autograd.Function):
            @staticmethod
            def forward(ctx, local_tensor):
                world_size = self._torch.distributed.get_world_size()
                gathered = [self._torch.zeros_like(local_tensor) for _ in range(world_size)]
                self._torch.distributed.all_gather(gathered, local_tensor)
                return tuple(gathered)

            @staticmethod
            def backward(ctx, *grad_outputs):
                rank = self._torch.distributed.get_rank()
                stacked = self._torch.stack(grad_outputs)
                self._torch.distributed.all_reduce(stacked)
                return stacked[rank]

        gathered = _Gather.apply(x)
        return self._torch.cat(gathered, dim=0)

    def contrastive_logits(self, image_embed: Any, text_embed: Any) -> tuple[Any, Any]:
        """Create image-to-text and text-to-image logits.

        Parameters
        ----------
        image_embed : Any
            Image embedding matrix ``(N, D)``.
        text_embed : Any
            Text embedding matrix ``(N, D)``.

        Returns
        -------
        tuple
            ``(logits_image_to_text, logits_text_to_image)``.
        """

        image_norm = self._torch.nn.functional.normalize(image_embed, dim=-1)
        text_norm = self._torch.nn.functional.normalize(text_embed, dim=-1)
        image_norm = self._all_gather_with_grad(image_norm)
        text_norm = self._all_gather_with_grad(text_norm)

        scale = self.logit_scale.clamp(max=100.0)
        logits_i2t = scale * (image_norm @ text_norm.t())
        logits_t2i = scale * (text_norm @ image_norm.t())
        return logits_i2t, logits_t2i

    def contrastive_loss(self, image_embed: Any, text_embed: Any) -> Any:
        """Compute symmetric image-text contrastive loss.

        Parameters
        ----------
        image_embed : Any
            Image embedding matrix.
        text_embed : Any
            Text embedding matrix.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        logits_i2t, logits_t2i = self.contrastive_logits(image_embed, text_embed)
        target = self._torch.arange(logits_i2t.shape[0], device=logits_i2t.device)
        loss = self.cross_entropy(logits_i2t, target) + self.cross_entropy(logits_t2i, target)
        return 0.5 * loss

    def __call__(self, texts: list[str], device: Any | None = None) -> Any:
        """Alias for ``encode_text``.

        Parameters
        ----------
        texts : list of str
            Text samples.
        device : Any, optional
            Torch device.

        Returns
        -------
        Any
            Encoded text embeddings.
        """

        return self.encode_text(texts, device=device)
