"""Unit tests for CSV-based data loading and transforms.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sfm.data.retinal_dataset import DatasetConfig, RetinalImageDataset
from sfm.data.transforms import TransformConfig, build_eval_transform


def test_retinal_dataset_with_synthetic_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "sample.png"

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 1] = 255
    Image.fromarray(arr).save(image_path)

    table = pd.DataFrame(
        [
            {
                "image_path": "images/sample.png",
                "label": 1,
            }
        ]
    )
    csv_path = tmp_path / "labels.csv"
    table.to_csv(csv_path, index=False)

    cfg = DatasetConfig(csv_path=csv_path, label_columns=("label",), data_root=tmp_path)
    ds = RetinalImageDataset(
        config=cfg,
        transform=build_eval_transform(TransformConfig(image_size=32)),
    )

    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["labels"].tolist() == [1.0]
    assert sample["path"] == "images/sample.png"
