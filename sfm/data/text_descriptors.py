"""Text-descriptor templates for multimodal SFM pretraining.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from typing import Iterable, Sequence


CKM_LABELS = ("CKD", "Diabetes", "Hypertension", "Stroke", "Obesity", "Cardiopathy")


def _normalize_binary(value: int | bool | float) -> int:
    """Normalize binary-like inputs to ``0`` or ``1``.

    Parameters
    ----------
    value : int, bool, or float
        Input value.

    Returns
    -------
    int
        Binary flag.
    """

    return 1 if int(value) > 0 else 0


def build_ckm_descriptor(
    ckd: int,
    diabetes: int,
    hypertension: int,
    stroke: int,
    obesity: int,
    cardiopathy: int,
) -> str:
    """Build a CKM descriptor sentence from six binary flags.

    Parameters
    ----------
    ckd : int
        CKD flag.
    diabetes : int
        Diabetes flag.
    hypertension : int
        Hypertension flag.
    stroke : int
        Stroke flag.
    obesity : int
        Obesity flag.
    cardiopathy : int
        Cardiopathy flag.

    Returns
    -------
    str
        Canonical descriptor text.
    """

    flags = {
        "CKD": _normalize_binary(ckd),
        "Diabetes": _normalize_binary(diabetes),
        "Hypertension": _normalize_binary(hypertension),
        "Stroke": _normalize_binary(stroke),
        "Obesity": _normalize_binary(obesity),
        "Cardiopathy": _normalize_binary(cardiopathy),
    }
    positive = [name for name, val in flags.items() if val == 1]
    negative = [name for name, val in flags.items() if val == 0]

    if positive:
        return (
            f"Detected CKM conditions: {', '.join(positive)}. "
            f"No evidence of: {', '.join(negative)}."
        )
    return f"No detected CKM comorbidity among: {', '.join(negative)}."


def build_descriptor_from_sequence(flags: Sequence[int | bool | float], label_names: Sequence[str] = CKM_LABELS) -> str:
    """Build descriptor from an ordered binary sequence.

    Parameters
    ----------
    flags : sequence
        Ordered binary flags.
    label_names : sequence of str
        Ordered label names matching ``flags``.

    Returns
    -------
    str
        Descriptor sentence.
    """

    if len(flags) != len(label_names):
        raise ValueError("flags and label_names must have identical length")

    norm = [_normalize_binary(v) for v in flags]
    positives = [name for name, flag in zip(label_names, norm) if flag == 1]
    negatives = [name for name, flag in zip(label_names, norm) if flag == 0]

    if positives:
        return f"Detected conditions: {', '.join(positives)}. Absent conditions: {', '.join(negatives)}."
    return f"No detected conditions in label set: {', '.join(negatives)}."


def batch_descriptors(flag_rows: Iterable[Sequence[int | bool | float]], label_names: Sequence[str] = CKM_LABELS) -> list[str]:
    """Build descriptor text for a batch of label rows.

    Parameters
    ----------
    flag_rows : iterable of sequences
        Batch of binary label rows.
    label_names : sequence of str
        Label names aligned with each row.

    Returns
    -------
    list of str
        Generated descriptors.
    """

    return [build_descriptor_from_sequence(row, label_names=label_names) for row in flag_rows]
