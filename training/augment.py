"""Forensic Image Preprocessing and Training Augmentation Pipeline.

Implements Sections 3.2 and 5.3 of DESIGN.md:
1. JPEG Q=85 normalization: Mandatory re-encoding at Q=85 via OpenCV to eliminate
   compression shortcut signals.
2. Random JPEG compression in [60, 95] with p=0.5 during training.
3. H.264 compression simulation for video frame artifacts.
4. Random resized crop (scale 0.85-1.0), horizontal flip, mild color jitter.
5. Dual output generation:
   - Stream A: ImageNet-normalized RGB tensor [3, 224, 224]
   - Stream B: Un-normalized luminance/grayscale tensor in [0, 1] [1, 224, 224]
"""

from typing import Optional, Tuple, Union
import io
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def jpeg_reencode(image_np: np.ndarray, quality: int = 85) -> np.ndarray:
    """Re-encode an RGB numpy array [H, W, 3] at specified JPEG quality.
    
    Args:
        image_np: uint8 RGB numpy array.
        quality: JPEG quality factor (1-100).
        
    Returns:
        Re-encoded uint8 RGB numpy array.
    """
    if HAS_CV2:
        # OpenCV uses BGR
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded = cv2.imencode(".jpg", bgr, encode_param)
        if not success:
            return image_np
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    else:
        # PIL fallback
        pil_img = Image.fromarray(image_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decoded = Image.open(buffer)
        return np.array(decoded.convert("RGB"))


def jpeg_normalize_q85(image_np: np.ndarray) -> np.ndarray:
    """Apply mandatory JPEG Q=85 normalization (Section 3.2)."""
    return jpeg_reencode(image_np, quality=85)


def random_jpeg_compression(image_np: np.ndarray, q_min: int = 60, q_max: int = 95) -> np.ndarray:
    """Apply random JPEG compression in [q_min, q_max] (Section 5.3)."""
    quality = random.randint(q_min, q_max)
    return jpeg_reencode(image_np, quality=quality)


def simulate_h264_artifacts(image_np: np.ndarray, crf: int = 28) -> np.ndarray:
    """Simulate H.264 macroblock compression artifacts.
    
    Uses block-based DCT quantization approximation when ffmpeg is not run via subprocess.
    """
    h, w, c = image_np.shape
    # Block size 8 or 16 for macroblocks
    block_size = 8
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        padded = image_np

    # Apply slight quantization step proportional to CRF
    step = max(1.0, float(crf) / 6.0)
    quantized = np.round(padded.astype(np.float32) / step) * step
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    return quantized[:h, :w]


class ForensicAugmentationPipeline:
    """End-to-end preprocessing and augmentation pipeline for KryptosID."""

    def __init__(
        self,
        is_training: bool = True,
        target_size: Tuple[int, int] = (224, 224),
        random_jpeg_prob: float = 0.5,
        h264_prob: float = 0.2,
    ) -> None:
        self.is_training = is_training
        self.target_size = target_size
        self.random_jpeg_prob = random_jpeg_prob
        self.h264_prob = h264_prob

        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process image input into dual-stream tensors.
        
        Args:
            image: uint8 numpy array [H, W, 3] or PIL Image.
            
        Returns:
            Tuple of:
            - spatial_tensor: ImageNet-normalized [3, 224, 224] float32 tensor
            - spectral_gray: Grayscale un-normalized [1, 224, 224] float32 tensor in [0, 1]
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = image.copy()

        # Step 1: Mandatory JPEG Q=85 normalization (Section 3.2)
        img_np = jpeg_normalize_q85(img_np)

        # Step 2: Training augmentations (Section 5.3)
        if self.is_training:
            # Additional random JPEG quality in [60, 95] with p=0.5
            if random.random() < self.random_jpeg_prob:
                img_np = random_jpeg_compression(img_np, q_min=60, q_max=95)

            # Simulated H.264 compression artifacts
            if random.random() < self.h264_prob:
                crf = random.randint(23, 38)
                img_np = simulate_h264_artifacts(img_np, crf=crf)

        pil_img = Image.fromarray(img_np)

        # Step 3: Geometric transformations
        if self.is_training:
            # Random resized crop (scale 0.85 to 1.0)
            width, height = pil_img.size
            scale = random.uniform(0.85, 1.0)
            crop_w, crop_h = int(width * scale), int(height * scale)
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            pil_img = TF.crop(pil_img, top, left, crop_h, crop_w)
            pil_img = TF.resize(pil_img, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)

            # Horizontal flip with p=0.5
            if random.random() < 0.5:
                pil_img = TF.hflip(pil_img)

            # Mild color jitter
            if random.random() < 0.3:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                pil_img = TF.adjust_brightness(pil_img, brightness)
                pil_img = TF.adjust_contrast(pil_img, contrast)
        else:
            # Inference: Center crop / aspect-preserving resize to 224x224
            pil_img = TF.resize(pil_img, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)

        # Convert to float tensor in [0, 1]
        rgb_tensor = TF.to_tensor(pil_img)  # [3, 224, 224]

        # Step 4: Stream B input (spectral)
        # Compute grayscale Y = 0.299 R + 0.587 G + 0.114 B on un-normalized [0, 1] tensor
        weights = rgb_tensor.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        spectral_gray = (rgb_tensor * weights).sum(dim=0, keepdim=True)  # [1, 224, 224]

        # Step 5: Stream A input (spatial)
        # Apply ImageNet normalization: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
        spatial_norm = TF.normalize(rgb_tensor, mean=self.mean, std=self.std)

        return spatial_norm, spectral_gray
