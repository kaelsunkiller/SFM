"""Image-text contrastive alignment objective for SFM-M2 pretraining.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from typing import Any

from sfm._optional import import_torch


def _torch_module():
    """Load torch lazily.

    Returns
    -------
    module
        Imported torch module.
    """

    torch, _nn = import_torch()
    return torch


def gather_with_grad(x: Any, dim: int = 0) -> Any:
    """Gather tensors across distributed processes with gradient support.

    Parameters
    ----------
    x : Any
        Local tensor.
    dim : int
        Concatenation dimension.

    Returns
    -------
    Any
        Concatenated tensor across processes.
    """

    torch = _torch_module()

    class _GatherWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local_tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, local_tensor)
                return tuple(gathered)
            return (local_tensor,)

        @staticmethod
        def backward(ctx, *grads):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                stacked = torch.stack(grads)
                torch.distributed.all_reduce(stacked)
                grad_out = stacked[torch.distributed.get_rank()]
                return grad_out
            return grads[0]

    return torch.cat(_GatherWithGrad.apply(x), dim=dim)


def create_logits(image_embed: Any, text_embed: Any, logit_scale: Any) -> tuple[Any, Any]:
    """Build bidirectional contrastive logits.

    Parameters
    ----------
    image_embed : Any
        Image embeddings with shape ``(N, D)``.
    text_embed : Any
        Text embeddings with shape ``(N, D)``.
    logit_scale : Any
        Scalar multiplier for cosine similarities.

    Returns
    -------
    tuple
        ``(image_to_text_logits, text_to_image_logits)``.
    """

    torch = _torch_module()
    image_embed = torch.nn.functional.normalize(image_embed, dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
    logits_i2t = logit_scale * (image_embed @ text_embed.t())
    logits_t2i = logit_scale * (text_embed @ image_embed.t())
    return logits_i2t, logits_t2i


def clip_contrastive_loss(
    image_embed: Any,
    text_embed: Any,
    temperature: float = 0.07,
    gather_distributed: bool = True,
) -> Any:
    """Compute symmetric InfoNCE loss for image-text alignment.

    Parameters
    ----------
    image_embed : Any
        Image embeddings.
    text_embed : Any
        Text embeddings.
    temperature : float
        Softmax temperature.
    gather_distributed : bool
        If true, perform distributed gather before computing logits.

    Returns
    -------
    Any
        Scalar contrastive loss tensor.
    """

    torch = _torch_module()
    if image_embed.shape[0] != text_embed.shape[0]:
        raise ValueError("Image and text embedding batch size must match.")

    if gather_distributed:
        image_embed = gather_with_grad(image_embed)
        text_embed = gather_with_grad(text_embed)

    logit_scale = 1.0 / max(float(temperature), 1e-6)
    logits_i2t, logits_t2i = create_logits(image_embed, text_embed, logit_scale=logit_scale)

    target = torch.arange(logits_i2t.shape[0], device=logits_i2t.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_i2t, target)
    loss_t2i = torch.nn.functional.cross_entropy(logits_t2i, target)
    return 0.5 * (loss_i2t + loss_t2i)
