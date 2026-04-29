"""Path-agnostic retinal image dataset loader.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for CSV-backed retinal datasets.

    Parameters
    ----------
    csv_path : pathlib.Path
        Label table path.
    image_column : str
        Column containing image-relative paths.
    label_columns : tuple of str
        Target label columns.
    data_root : pathlib.Path or None
        Image root directory; falls back to ``SFM_DATA_ROOT`` when omitted.
    """

    csv_path: Path
    image_column: str = "image_path"
    label_columns: tuple[str, ...] = ("label",)
    data_root: Path | None = None


def _resolve_data_root(data_root: Path | None = None) -> Path:
    """Resolve data root from explicit value or environment variable.

    Parameters
    ----------
    data_root : pathlib.Path or None
        Preferred root path.

    Returns
    -------
    pathlib.Path
        Resolved data root path.
    """

    if data_root is not None:
        return Path(data_root)
    env_root = os.getenv("SFM_DATA_ROOT", ".")
    return Path(env_root)


def load_label_table(csv_path: Path) -> pd.DataFrame:
    """Load label CSV.

    Parameters
    ----------
    csv_path : pathlib.Path
        CSV path.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """

    return pd.read_csv(csv_path)


class RetinalImageDataset:
    """CSV-indexed dataset returning images and labels.

    Parameters
    ----------
    config : DatasetConfig
        Dataset metadata configuration.
    transform : callable, optional
        Optional image transformation function.
    strict : bool
        If true, missing files raise an exception.

    Notes
    -----
    This class is framework-agnostic and returns NumPy arrays by default.
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Callable[[Image.Image], np.ndarray] | None = None,
        strict: bool = True,
    ) -> None:
        self.config = config
        self.transform = transform
        self.strict = strict
        self.data_root = _resolve_data_root(config.data_root)
        self.table = load_label_table(config.csv_path)
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate required CSV columns.

        Returns
        -------
        None
            Raises when required columns are absent.
        """

        required = [self.config.image_column, *self.config.label_columns]
        missing = [col for col in required if col not in self.table.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        """Return number of samples.

        Returns
        -------
        int
            Dataset length.
        """

        return int(len(self.table))

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one sample.

        Parameters
        ----------
        index : int
            Row index.

        Returns
        -------
        dict
            Dictionary with keys ``image``, ``labels``, and ``path``.
        """

        row = self.table.iloc[index]
        rel_path = Path(str(row[self.config.image_column]))
        image_path = self.data_root / rel_path
        if not image_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Missing image file: {image_path}")
            image = np.zeros((3, 224, 224), dtype=np.float32)
        else:
            with Image.open(image_path) as pil:
                pil = pil.convert("RGB")
                if self.transform is None:
                    arr = np.asarray(pil, dtype=np.float32) / 255.0
                    image = np.transpose(arr, (2, 0, 1))
                else:
                    image = self.transform(pil)
        labels = np.asarray([row[c] for c in self.config.label_columns], dtype=np.float32)
        return {"image": image, "labels": labels, "path": rel_path.as_posix()}
