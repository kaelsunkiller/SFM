"""Optional dependency helpers used by model modules.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations


def import_torch():
    """Import torch lazily.

    Returns
    -------
    tuple
        ``(torch_module, nn_module)`` when torch is available.

    Raises
    ------
    RuntimeError
        Raised when torch is not installed in the environment.
    """

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(
            "PyTorch is required for this module. Install extras: `pip install sfm[torch]`."
        ) from exc
    return torch, nn
