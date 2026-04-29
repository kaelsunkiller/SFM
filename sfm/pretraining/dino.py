"""DINO self-distillation objective used in SFM-M2 pretraining.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm._optional import import_torch


@dataclass(frozen=True)
class DINOLossConfig:
    """Hyperparameters for teacher-student distillation.

    Parameters
    ----------
    student_temp : float
        Student softmax temperature.
    teacher_temp : float
        Final teacher temperature.
    warmup_teacher_temp : float
        Initial teacher temperature used in warmup.
    warmup_teacher_temp_epochs : int
        Number of warmup epochs for teacher temperature.
    num_epochs : int
        Total epoch count used to build the temperature schedule.
    center_momentum : float
        EMA momentum for teacher-output centering.
    """

    student_temp: float = 0.1
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 0
    num_epochs: int = 51
    center_momentum: float = 0.9


class DINOLoss:
    """Cross-view DINO loss with temperature schedule and center EMA.

    Parameters
    ----------
    output_dim : int
        Projection dimension of DINO logits.
    num_crops : int
        Number of student views used per sample.
    config : DINOLossConfig, optional
        Distillation hyperparameters.

    Notes
    -----
    This implementation follows the published multi-crop training setup:
    teacher outputs are centered and sharpened, and the student is trained
    against teacher predictions from non-matching views.
    """

    def __init__(self, output_dim: int, num_crops: int = 6, config: DINOLossConfig | None = None) -> None:
        self.config = config or DINOLossConfig()
        self.output_dim = output_dim
        self.num_crops = num_crops

        torch, _ = import_torch()
        self._torch = torch
        self.center = torch.zeros(1, output_dim)
        self.teacher_temp_schedule = self._build_teacher_schedule()

    def _build_teacher_schedule(self) -> np.ndarray:
        """Construct per-epoch teacher temperature schedule.

        Returns
        -------
        numpy.ndarray
            Schedule of shape ``(num_epochs,)``.
        """

        warmup = self.config.warmup_teacher_temp_epochs
        total = self.config.num_epochs
        if warmup > 0:
            warmup_values = np.linspace(
                self.config.warmup_teacher_temp,
                self.config.teacher_temp,
                warmup,
            )
        else:
            warmup_values = np.empty((0,), dtype=float)
        remain = max(total - warmup, 0)
        tail = np.full((remain,), self.config.teacher_temp, dtype=float)
        schedule = np.concatenate([warmup_values, tail], axis=0)
        if schedule.size == 0:
            schedule = np.array([self.config.teacher_temp], dtype=float)
        return schedule

    def to(self, device: Any):
        """Move state buffers to device.

        Parameters
        ----------
        device : Any
            Torch device.

        Returns
        -------
        DINOLoss
            Self.
        """

        self.center = self.center.to(device)
        return self

    def _split_student_outputs(self, student_output: Any) -> list[Any]:
        """Normalize student output into per-crop tensor list.

        Parameters
        ----------
        student_output : Any
            Student logits tensor or list.

        Returns
        -------
        list
            Per-crop student logits.
        """

        if isinstance(student_output, (list, tuple)):
            return list(student_output)

        student_scaled = student_output / self.config.student_temp
        if student_scaled.ndim != 2:
            raise ValueError("Student output must be rank-2 tensor or list of rank-2 tensors.")
        return list(student_scaled.chunk(self.num_crops))

    def _split_teacher_outputs(self, teacher_output: Any, epoch: int) -> list[Any]:
        """Apply centering/sharpening and split teacher outputs.

        Parameters
        ----------
        teacher_output : Any
            Teacher logits.
        epoch : int
            Current training epoch.

        Returns
        -------
        list
            Two teacher-view logits after sharpening.
        """

        epoch_idx = int(max(0, min(epoch, len(self.teacher_temp_schedule) - 1)))
        temperature = float(self.teacher_temp_schedule[epoch_idx])
        centered = (teacher_output - self.center.to(teacher_output.device)) / temperature
        probs = self._torch.softmax(centered, dim=-1).detach()
        return list(probs.chunk(2))

    def _update_center(self, teacher_output: Any) -> None:
        """Update teacher-output center with distributed EMA.

        Parameters
        ----------
        teacher_output : Any
            Teacher logits for the current batch.

        Returns
        -------
        None
            Updates ``self.center`` in place.
        """

        batch_center = teacher_output.detach().mean(dim=0, keepdim=True)
        if self._torch.distributed.is_available() and self._torch.distributed.is_initialized():
            self._torch.distributed.all_reduce(batch_center)
            batch_center = batch_center / self._torch.distributed.get_world_size()
        self.center = self.center.to(batch_center.device)
        self.center.mul_(self.config.center_momentum).add_(batch_center * (1.0 - self.config.center_momentum))

    def __call__(self, student_output: Any, teacher_output: Any, epoch: int = 0) -> Any:
        """Compute multi-view DINO loss.

        Parameters
        ----------
        student_output : Any
            Student logits.
        teacher_output : Any
            Teacher logits from two global views.
        epoch : int
            Current epoch index.

        Returns
        -------
        Any
            Scalar loss tensor.
        """

        student_chunks = self._split_student_outputs(student_output)
        student_chunks = [chunk / self.config.student_temp for chunk in student_chunks]
        teacher_chunks = self._split_teacher_outputs(teacher_output, epoch)

        total_loss = self._torch.tensor(0.0, device=teacher_output.device)
        n_terms = 0
        for teacher_idx, teacher_prob in enumerate(teacher_chunks):
            for student_idx, student_logits in enumerate(student_chunks):
                if student_idx == teacher_idx:
                    continue
                loss = -(teacher_prob * self._torch.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()
                total_loss = total_loss + loss
                n_terms += 1

        if n_terms == 0:
            raise RuntimeError("No valid DINO loss terms were created from the provided crop configuration.")

        total_loss = total_loss / n_terms
        self._update_center(teacher_output)
        return total_loss


def update_teacher_weights(teacher_state: dict[str, Any], student_state: dict[str, Any], momentum: float) -> dict[str, Any]:
    """Apply EMA update for teacher weights.

    Parameters
    ----------
    teacher_state : dict
        Existing teacher state dictionary.
    student_state : dict
        Student state dictionary.
    momentum : float
        EMA momentum in ``[0, 1)``.

    Returns
    -------
    dict
        Updated teacher-state dictionary.
    """

    updated: dict[str, Any] = {}
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key]
        updated[key] = momentum * teacher_value + (1.0 - momentum) * student_value
    return updated
