"""Checkpoint I/O utilities with optional Hugging Face download support.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


def resolve_checkpoint_dir(path: str | Path | None = None) -> Path:
    """Resolve checkpoint directory from argument or environment.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Optional explicit checkpoint directory.

    Returns
    -------
    pathlib.Path
        Directory path used for checkpoint files.
    """

    if path is not None:
        out = Path(path)
    else:
        out = Path(os.getenv("SFM_CKPT_DIR", "checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_checkpoint(payload: dict[str, Any], filename: str, directory: str | Path | None = None) -> Path:
    """Serialize a checkpoint payload.

    Parameters
    ----------
    payload : dict
        Serializable state dictionary.
    filename : str
        Output file name.
    directory : str or pathlib.Path or None
        Optional output directory.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """

    out_dir = resolve_checkpoint_dir(directory)
    path = out_dir / filename
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint file path.

    Returns
    -------
    dict
        Deserialized checkpoint dictionary.
    """

    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def download_checkpoint_from_hf(repo_id: str, filename: str, cache_dir: str | Path | None = None) -> Path:
    """Download a checkpoint file from Hugging Face Hub.

    Parameters
    ----------
    repo_id : str
        Hub repository identifier.
    filename : str
        File path in the repository.
    cache_dir : str or pathlib.Path or None
        Optional cache directory.

    Returns
    -------
    pathlib.Path
        Local path to downloaded file.
    """

    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(resolved)
