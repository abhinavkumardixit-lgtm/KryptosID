"""KryptosID Training and Data Pipeline Package.

Includes:
- splits.py: Identity-disjoint and generator-disjoint dataset splitting and manifest hashing.
- augment.py: JPEG Q=85 normalization, random JPEG compression, and crop augmentations.
- datasets.py: Dataset loaders and manifests.
"""

from training.splits import (
    SplitManifest,
    create_identity_disjoint_split,
    create_generator_disjoint_split,
    verify_split_manifest,
)
from training.augment import (
    ForensicAugmentationPipeline,
    jpeg_normalize_q85,
    random_jpeg_compression,
)

__all__ = [
    "SplitManifest",
    "create_identity_disjoint_split",
    "create_generator_disjoint_split",
    "verify_split_manifest",
    "ForensicAugmentationPipeline",
    "jpeg_normalize_q85",
    "random_jpeg_compression",
]
