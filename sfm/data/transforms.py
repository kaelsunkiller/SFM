"""Image transformation pipelines for SFM pretraining and fine-tuning.

Reference: Methods §Model architecture and pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for image transforms.

    Parameters
    ----------
    image_size : int
        Output square size.
    train_flip_prob : float
        Horizontal flip probability in training transforms.
    global_crops_scale : tuple of float
        Scale range for global crops in DINO augmentation.
    local_crops_scale : tuple of float
        Scale range for local crops in DINO augmentation.
    local_crops_number : int
        Number of local crops.
    """

    image_size: int = 224
    train_flip_prob: float = 0.5
    global_crops_scale: tuple[float, float] = (0.4, 1.0)
    local_crops_scale: tuple[float, float] = (0.05, 0.4)
    local_crops_number: int = 4


def _to_numpy_rgb(image: Image.Image, image_size: int) -> np.ndarray:
    """Convert PIL image to normalized CHW NumPy tensor.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    image_size : int
        Target output size.

    Returns
    -------
    numpy.ndarray
        Normalized tensor with shape ``(3, image_size, image_size)``.
    """

    resized = image.convert("RGB").resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (arr - mean) / std


def build_train_transform(config: TransformConfig | None = None) -> Callable[[Image.Image], np.ndarray]:
    """Build fine-tuning train transform.

    Parameters
    ----------
    config : TransformConfig, optional
        Transform hyperparameters.

    Returns
    -------
    callable
        Transformation callable returning a NumPy tensor.
    """

    cfg = config or TransformConfig()

    def _transform(image: Image.Image) -> np.ndarray:
        out = image
        if np.random.random() < cfg.train_flip_prob:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        return _to_numpy_rgb(out, cfg.image_size)

    return _transform


def build_eval_transform(config: TransformConfig | None = None) -> Callable[[Image.Image], np.ndarray]:
    """Build deterministic evaluation transform.

    Parameters
    ----------
    config : TransformConfig, optional
        Transform hyperparameters.

    Returns
    -------
    callable
        Transformation callable returning a NumPy tensor.
    """

    cfg = config or TransformConfig()

    def _transform(image: Image.Image) -> np.ndarray:
        return _to_numpy_rgb(image, cfg.image_size)

    return _transform


class DataAugmentationDINO:
    """Multi-crop DINO augmentation with two global and local crops.

    Parameters
    ----------
    config : TransformConfig, optional
        Configuration for crop scales and local-crop count.

    Notes
    -----
    This class is used in pretraining and returns torch tensors for each crop.
    """

    def __init__(self, config: TransformConfig | None = None) -> None:
        self.config = config or TransformConfig()
        try:
            import torchvision.transforms as tvt
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("torchvision is required for DataAugmentationDINO.") from exc

        flip_and_color = tvt.Compose(
            [
                tvt.RandomHorizontalFlip(p=self.config.train_flip_prob),
                tvt.RandomApply([tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                tvt.RandomGrayscale(p=0.2),
            ]
        )
        normalize = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_transfo1 = tvt.Compose(
            [
                tvt.RandomResizedCrop(self.config.image_size, scale=self.config.global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color,
                tvt.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                normalize,
            ]
        )
        self.global_transfo2 = tvt.Compose(
            [
                tvt.RandomResizedCrop(self.config.image_size, scale=self.config.global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color,
                tvt.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                tvt.RandomSolarize(threshold=128.0, p=0.2),
                normalize,
            ]
        )
        self.local_transfo = tvt.Compose(
            [
                tvt.RandomResizedCrop(self.config.image_size, scale=self.config.local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color,
                tvt.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                normalize,
            ]
        )

    def __call__(self, image: Image.Image):
        """Generate multi-crop augmentations for one image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image.

        Returns
        -------
        list
            Two global crops followed by local crops.
        """

        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.config.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
