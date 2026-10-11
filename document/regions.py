"""Document Photo and MRZ (Machine Readable Zone) Region Extraction.

Implements Section 3.5 of DESIGN.md:
- Extracts photo crop for face artifact detection.
- Extracts MRZ region for copy-move tampering and text alignment checks.
"""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np


def extract_id_regions(
    rectified_doc: np.ndarray,
    target_photo_size: Tuple[int, int] = (224, 224),
) -> Dict[str, np.ndarray]:
    """Extract standard ID card regions from canonical rectified document (856x540).
    
    Returns:
        Dictionary with 'photo_region', 'mrz_region', and 'full_document'.
    """
    h, w = rectified_doc.shape[:2]

    # Standard ID photo coordinates: Left side (approx x: 5% to 38%, y: 15% to 80%)
    photo_y1, photo_y2 = int(h * 0.15), int(h * 0.80)
    photo_x1, photo_x2 = int(w * 0.05), int(w * 0.38)
    photo_raw = rectified_doc[photo_y1:photo_y2, photo_x1:photo_x2]
    photo_resized = cv2.resize(photo_raw, target_photo_size, interpolation=cv2.INTER_LINEAR)

    # MRZ region: Bottom strip (approx y: 75% to 98%, x: 2% to 98%)
    mrz_y1, mrz_y2 = int(h * 0.75), int(h * 0.98)
    mrz_x1, mrz_x2 = int(w * 0.02), int(w * 0.98)
    mrz_raw = rectified_doc[mrz_y1:mrz_y2, mrz_x1:mrz_x2]

    return {
        "photo_region": photo_resized,
        "mrz_region": mrz_raw,
        "full_document": rectified_doc,
    }
