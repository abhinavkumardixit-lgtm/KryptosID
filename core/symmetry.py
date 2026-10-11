"""D4 Dihedral Group Actions and Document Copy-Move Detection.

Implements Section 4.5 of DESIGN.md:
- Group: D4 (symmetry group of the square, 8 elements):
  Rotations: 0 deg, 90 deg, 180 deg, 270 deg
  Reflections: Horizontal, Vertical, Main Diagonal, Anti-Diagonal.
- Use: Copy-move forgery detection on documents. Non-overlapping image patches
  are matched under all 8 D4 actions.
- Output: Candidate region pairs, the transform used, and a similarity score (evidence for review).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


D4_TRANSFORMS = [
    "rot0",               # Identity
    "rot90",              # 90 deg counter-clockwise
    "rot180",             # 180 deg
    "rot270",             # 270 deg
    "flip_horizontal",    # Horizontal reflection (left-right)
    "flip_vertical",      # Vertical reflection (up-down)
    "diag_reflection",    # Main diagonal transpose
    "antidiag_reflection" # Anti-diagonal reflection
]


def apply_d4_transform(image: np.ndarray, transform_name: str) -> np.ndarray:
    """Apply one of the 8 D4 dihedral group actions to a 2D or 3D numpy image."""
    if transform_name == "rot0":
        return image.copy()
    elif transform_name == "rot90":
        return np.rot90(image, k=1, axes=(0, 1)).copy()
    elif transform_name == "rot180":
        return np.rot90(image, k=2, axes=(0, 1)).copy()
    elif transform_name == "rot270":
        return np.rot90(image, k=3, axes=(0, 1)).copy()
    elif transform_name == "flip_horizontal":
        # Left-right flip along axis 1 (width)
        return np.fliplr(image).copy() if image.ndim == 2 else np.flip(image, axis=1).copy()
    elif transform_name == "flip_vertical":
        # Up-down flip along axis 0 (height)
        return np.flipud(image).copy() if image.ndim == 2 else np.flip(image, axis=0).copy()
    elif transform_name == "diag_reflection":
        # Transpose row and column
        if image.ndim == 2:
            return image.T.copy()
        else:
            return np.transpose(image, (1, 0, 2)).copy()
    elif transform_name == "antidiag_reflection":
        # Transpose across anti-diagonal: rot90 then horizontal flip
        rot = np.rot90(image, k=1, axes=(0, 1))
        return np.fliplr(rot).copy() if image.ndim == 2 else np.flip(rot, axis=1).copy()
    else:
        raise ValueError(f"Unknown D4 transform: {transform_name}. Must be one of {D4_TRANSFORMS}")


@dataclass
class CopyMoveMatch:
    """Candidate copy-move match between two non-overlapping regions."""
    region_a: Tuple[int, int, int, int]  # (y1, x1, y2, x2)
    region_b: Tuple[int, int, int, int]  # (y1, x1, y2, x2)
    transform: str
    similarity: float  # Normalized cross-correlation in [0, 1]


def detect_copy_move_d4(
    image: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
    similarity_threshold: float = 0.95,
    min_distance: int = 40,
    max_candidates: int = 5,
) -> List[CopyMoveMatch]:
    """Detect candidate copy-move regions under D4 symmetry transforms.
    
    Args:
        image: 2D grayscale or 3D RGB uint8 image.
        patch_size: Square patch dimension (e.g. 32x32).
        stride: Scanning stride.
        similarity_threshold: Minimum normalized correlation (default 0.95).
        min_distance: Minimum Euclidean distance between patch centers to prevent self-matching.
        max_candidates: Maximum candidate matches to return.
        
    Returns:
        List of candidate CopyMoveMatch objects sorted by similarity.
    """
    if image.ndim == 3:
        # Convert to float grayscale for correlation
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape
    if h < patch_size or w < patch_size:
        return []

    # Extract non-overlapping / strided patches
    patches = []
    coords = []
    patch_stds = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p = gray[y : y + patch_size, x : x + patch_size]
            std = float(np.std(p))
            # Ignore flat, textureless patches (e.g. plain background)
            if std > 5.0:
                p_norm = (p - np.mean(p)) / (std + 1e-7)
                patches.append(p_norm)
                coords.append((y, x))
                patch_stds.append(std)

    if len(patches) < 2:
        return []

    n_patches = len(patches)
    matches: List[CopyMoveMatch] = []

    # Compare patches under D4 group actions
    for i in range(n_patches):
        y1, x1 = coords[i]
        p1 = patches[i]

        for t_name in D4_TRANSFORMS:
            p1_t = apply_d4_transform(p1, t_name)

            for j in range(i + 1, n_patches):
                y2, x2 = coords[j]
                # Distance filter
                dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
                if dist < min_distance:
                    continue

                p2 = patches[j]
                # Normalized correlation
                corr = float(np.mean(p1_t * p2))

                if corr >= similarity_threshold:
                    matches.append(
                        CopyMoveMatch(
                            region_a=(y1, x1, y1 + patch_size, x1 + patch_size),
                            region_b=(y2, x2, y2 + patch_size, x2 + patch_size),
                            transform=t_name,
                            similarity=round(corr, 4),
                        )
                    )

    # Sort descending by similarity and take top matches
    matches.sort(key=lambda m: -m.similarity)
    return matches[:max_candidates]
