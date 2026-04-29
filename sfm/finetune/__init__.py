"""Downstream CKM and ophthalmic task heads.

Reference: Methods §Fine-tuning for downstream tasks.
"""

from .ckm_comorbidity import CKMComorbidityHead
from .ckm_staging import CKMStagingHead
from .biomarker import BiomarkerHead
from .progression import ProgressionHead
from .ophthalmic_transfer import OphthalmicTransferHead

__all__ = [
    "CKMComorbidityHead",
    "CKMStagingHead",
    "BiomarkerHead",
    "ProgressionHead",
    "OphthalmicTransferHead",
]
