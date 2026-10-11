"""Video Frame Sampling and Face Crop Extraction.

Implements Section 3.4 and Section 8.1 of DESIGN.md:
- Samples frames at a fixed rate (e.g. 4 fps) up to max_frames limit.
- Face alignment and aspect-preserving 224x224 cropping.
"""

from typing import List, Optional, Tuple
import os
import tempfile
import cv2
import numpy as np


def sample_frames_from_bytes(
    video_bytes: bytes,
    target_fps: float = 4.0,
    max_frames: int = 120,
) -> List[np.ndarray]:
    """Sample frames at a fixed frame rate from raw video bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return []

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 25.0

        sample_interval = max(1, int(round(orig_fps / target_fps)))
        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

            frame_idx += 1

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def extract_face_crops(
    frames: List[np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Extract face crops (or aspect-preserving center crop fallback) for each frame."""
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

    crops = []
    for frame in frames:
        h, w, _ = frame.shape
        detected = False

        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                # Largest detected face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces[0]
                # Margin around face
                margin_x, margin_y = int(fw * 0.15), int(fh * 0.15)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + fw + margin_x)
                y2 = min(h, y + fh + margin_y)
                crop = frame[y1:y2, x1:x2]
                detected = True
                crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
                crops.append(crop_resized)

        if not detected:
            # Aspect-preserving center crop
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            center_crop = frame[top : top + min_dim, left : left + min_dim]
            crop_resized = cv2.resize(center_crop, target_size, interpolation=cv2.INTER_LINEAR)
            crops.append(crop_resized)

    return crops
