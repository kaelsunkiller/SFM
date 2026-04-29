"""Joint pretraining utilities for DINO and image-text alignment.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sfm._optional import import_torch

from .clip_align import clip_contrastive_loss
from .dino import DINOLoss


def _torch_module():
    """Load torch lazily.

    Returns
    -------
    module
        Imported torch module.
    """

    torch, _nn = import_torch()
    return torch


@dataclass
class PretrainBatch:
    """Container for one multimodal pretraining mini-batch.

    Parameters
    ----------
    images : list
        Multi-crop image tensors. The first two entries are global crops.
    texts : list of str
        Text descriptions aligned to images in the first global crop.
    """

    images: list[Any]
    texts: list[str]


class PretrainTrainer:
    """Run SFM-M2 joint pretraining updates.

    Parameters
    ----------
    dino_loss : DINOLoss
        DINO distillation objective.
    contrastive_weight : float
        Weight for image-text alignment loss.
    grad_clip : float
        Optional max gradient norm.

    Notes
    -----
    The training step follows the released formulation: teacher receives
    two global crops, student receives all crops, and total loss is
    DINO plus weighted image-text contrastive loss.
    """

    def __init__(self, dino_loss: DINOLoss, contrastive_weight: float = 0.15, grad_clip: float = 3.0) -> None:
        self.dino_loss = dino_loss
        self.contrastive_weight = float(contrastive_weight)
        self.grad_clip = float(grad_clip)

    @staticmethod
    def _set_optimizer_lr(optimizer: Any, lr: float, weight_decay: float | None = None) -> None:
        """Set optimizer LR and optional weight decay for all parameter groups.

        Parameters
        ----------
        optimizer : Any
            Torch optimizer.
        lr : float
            Learning rate for all groups.
        weight_decay : float, optional
            If provided, overwrite weight decay for regularized groups.

        Returns
        -------
        None
            Updates optimizer in place.
        """

        for idx, group in enumerate(optimizer.param_groups):
            group["lr"] = lr
            if weight_decay is not None and idx == 0:
                group["weight_decay"] = weight_decay

    def compute_loss(
        self,
        student_logits: Any,
        teacher_logits: Any,
        image_embeddings: Any,
        text_embeddings: Any,
        epoch: int,
    ) -> dict[str, Any]:
        """Compute component and total pretraining losses.

        Parameters
        ----------
        student_logits : Any
            Student DINO logits from all crops.
        teacher_logits : Any
            Teacher logits from two global crops.
        image_embeddings : Any
            Image embeddings for contrastive alignment.
        text_embeddings : Any
            Text embeddings for contrastive alignment.
        epoch : int
            Current epoch index.

        Returns
        -------
        dict
            Dictionary containing ``dino_loss``, ``clip_loss``, and ``total_loss``.
        """

        dino = self.dino_loss(student_logits, teacher_logits, epoch=epoch)
        clip = clip_contrastive_loss(image_embeddings, text_embeddings, gather_distributed=True)
        total = dino + self.contrastive_weight * clip
        return {"dino_loss": dino, "clip_loss": clip, "total_loss": total}

    def train_step(
        self,
        student_model: Any,
        teacher_model: Any,
        text_encoder: Any,
        optimizer: Any,
        batch: PretrainBatch,
        epoch: int,
        momentum: float,
        use_amp: bool = True,
        scaler: Any | None = None,
    ) -> dict[str, float]:
        """Run one optimization step.

        Parameters
        ----------
        student_model : Any
            Student visual model.
        teacher_model : Any
            Teacher visual model.
        text_encoder : Any
            Text encoder module.
        optimizer : Any
            Optimizer for student and text encoder.
        batch : PretrainBatch
            Input batch with image crops and texts.
        epoch : int
            Current epoch index.
        momentum : float
            EMA coefficient for teacher update.
        use_amp : bool
            Whether to use autocast.
        scaler : Any, optional
            GradScaler instance.

        Returns
        -------
        dict
            Scalar loss values for logging.
        """

        torch = _torch_module()

        student_model.train(True)
        teacher_model.train(True)
        text_encoder.train(True)

        image_crops = [crop for crop in batch.images]
        if len(image_crops) < 2:
            raise ValueError("At least two global crops are required for DINO pretraining.")

        device = image_crops[0].device
        with torch.cuda.amp.autocast(enabled=bool(use_amp and scaler is not None)):
            with torch.no_grad():
                teacher_outputs = teacher_model(image_crops[:2])
                if isinstance(teacher_outputs, tuple):
                    student_ref, teacher_logits = teacher_outputs
                else:
                    student_ref = teacher_outputs
                    teacher_logits = teacher_outputs

            student_outputs = student_model(image_crops)
            if isinstance(student_outputs, tuple):
                student_features, student_logits = student_outputs
            else:
                student_features = student_outputs
                student_logits = student_outputs

            text_embeddings = text_encoder(batch.texts, device=device)
            if student_features.ndim == 3:
                image_embeddings = student_features.mean(dim=1)
            else:
                image_embeddings = student_features

            _ = student_ref
            losses = self.compute_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                image_embeddings=image_embeddings[: text_embeddings.shape[0]],
                text_embeddings=text_embeddings,
                epoch=epoch,
            )

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            losses["total_loss"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student_model.parameters()) + list(text_encoder.parameters()),
                    self.grad_clip,
                )
            optimizer.step()
        else:
            scaler.scale(losses["total_loss"]).backward()
            if self.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student_model.parameters()) + list(text_encoder.parameters()),
                    self.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            student_params = dict(student_model.named_parameters())
            for name, teacher_param in teacher_model.named_parameters():
                if name in student_params:
                    teacher_param.data.mul_(momentum).add_(student_params[name].data * (1.0 - momentum))

        return {
            "dino_loss": float(losses["dino_loss"].detach().cpu().item()),
            "clip_loss": float(losses["clip_loss"].detach().cpu().item()),
            "total_loss": float(losses["total_loss"].detach().cpu().item()),
        }

    def run_epoch(
        self,
        student_model: Any,
        teacher_model: Any,
        text_encoder: Any,
        optimizer: Any,
        batches: Iterable[PretrainBatch],
        epoch: int,
        momentum_schedule: list[float],
        lr_schedule: list[float] | None = None,
        wd_schedule: list[float] | None = None,
        scaler: Any | None = None,
    ) -> dict[str, float]:
        """Run one full epoch over an iterable of pretraining batches.

        Parameters
        ----------
        student_model : Any
            Student model.
        teacher_model : Any
            Teacher model.
        text_encoder : Any
            Text encoder module.
        optimizer : Any
            Torch optimizer.
        batches : iterable
            Iterable yielding ``PretrainBatch``.
        epoch : int
            Current epoch index.
        momentum_schedule : list of float
            Momentum value per iteration.
        lr_schedule : list of float, optional
            Learning-rate schedule per iteration.
        wd_schedule : list of float, optional
            Weight-decay schedule per iteration.
        scaler : Any, optional
            Optional GradScaler.

        Returns
        -------
        dict
            Mean losses for the epoch.
        """

        stats = {"dino_loss": 0.0, "clip_loss": 0.0, "total_loss": 0.0}
        n_steps = 0

        for step, batch in enumerate(batches):
            if lr_schedule is not None:
                lr = float(lr_schedule[min(step, len(lr_schedule) - 1)])
                wd = float(wd_schedule[min(step, len(wd_schedule) - 1)]) if wd_schedule is not None else None
                self._set_optimizer_lr(optimizer, lr=lr, weight_decay=wd)

            momentum = float(momentum_schedule[min(step, len(momentum_schedule) - 1)])
            step_stats = self.train_step(
                student_model=student_model,
                teacher_model=teacher_model,
                text_encoder=text_encoder,
                optimizer=optimizer,
                batch=batch,
                epoch=epoch,
                momentum=momentum,
                use_amp=True,
                scaler=scaler,
            )
            n_steps += 1
            for key in stats:
                stats[key] += step_stats[key]

        if n_steps == 0:
            raise RuntimeError("No pretraining batches were provided.")

        return {key: val / n_steps for key, val in stats.items()}
