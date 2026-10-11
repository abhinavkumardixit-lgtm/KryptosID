"""Document Quadrilateral Detection and Perspective Rectification.

Implements Section 3.5 of DESIGN.md:
- Finds outer document edges using contour detection.
- Applies 4-point perspective transform to rectify document to canonical aspect ratio (e.g. ID-1 format 856x540).
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray, canonical_size: Tuple[int, int] = (856, 540)) -> np.ndarray:
    """Warp quadrilateral to canonical rectangle."""
    rect = order_points(pts)
    dst_w, dst_h = canonical_size

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
    return warped


def detect_and_rectify_document(
    image: np.ndarray,
    canonical_size: Tuple[int, int] = (856, 540),
) -> Tuple[np.ndarray, bool]:
    """Detect document contour and rectify to canonical size."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_quad = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.15):
            doc_quad = approx.reshape(4, 2).astype("float32")
            break

    if doc_quad is not None:
        rectified = four_point_transform(image, doc_quad, canonical_size=canonical_size)
        return rectified, True
    else:
        # Fallback to direct resize
        resized = cv2.resize(image, canonical_size, interpolation=cv2.INTER_LINEAR)
        return resized, False
