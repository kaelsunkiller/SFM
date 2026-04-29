"""Public SFM package for CKM retinal screening workflows.

Reference: Methods §Model architecture and pretraining.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sfm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
