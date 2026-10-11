"""Butterworth 2D High-Pass Filter and Spectral Processing.

Implements Section 4.2 of DESIGN.md:
- 224x224 grid centered at DC (112, 112)
- Nyquist at index 112
- Butterworth transfer function: H(u, v) = 1 / (1 + (D0 / D(u, v))^(2n)), with D0=25, n=2
- Stored as non-trainable registered buffer
- 2D FFT/IFFT with per-sample standardization (zero mean, unit variance)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


def compute_butterworth_mask(
    height: int = 224,
    width: int = 224,
    d0: float = 25.0,
    order: int = 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute 2D Butterworth high-pass transfer function centered at (height/2, width/2).
    
    Args:
        height: Spatial height (default 224).
        width: Spatial width (default 224).
        d0: Cutoff distance in index units (default 25.0, ~22% of Nyquist).
        order: Filter order n (default 2).
        device: Target torch device.
        dtype: Data type (float32).
        
    Returns:
        Tensor of shape [1, 1, height, width] representing H(u, v).
    """
    center_u = float(height) / 2.0
    center_v = float(width) / 2.0

    u = torch.arange(height, dtype=dtype, device=device) - center_u
    v = torch.arange(width, dtype=dtype, device=device) - center_v

    grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
    dist = torch.sqrt(grid_u**2 + grid_v**2)

    # Butterworth high-pass transfer function:
    # H(u, v) = 1 / (1 + (D0 / D(u, v))^(2*n))
    # At D(u, v) == 0, response is 0 to avoid division by zero and eliminate DC.
    mask = torch.zeros_like(dist)
    nonzero = dist > 0
    mask[nonzero] = 1.0 / (1.0 + (d0 / dist[nonzero]) ** (2 * order))

    return mask.unsqueeze(0).unsqueeze(0)


def rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [B, 3, H, W] to grayscale [B, 1, H, W] using standard luminance weights.
    
    Luminance weights: Y = 0.299 * R + 0.587 * G + 0.114 * B
    """
    if rgb.shape[1] == 1:
        return rgb
    if rgb.shape[1] != 3:
        raise ValueError(f"Expected 3 channels for RGB input, got {rgb.shape[1]}")

    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


class ButterworthHighPassFilter2D(nn.Module):
    """2D Butterworth High-Pass Filter module.
    
    Filters 2D inputs in the frequency domain via 2D FFT, applies the centered
    Butterworth high-pass mask, computes the inverse 2D FFT, and applies
    per-sample standardization (zero mean, unit variance).
    """

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        d0: float = 25.0,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.d0 = d0
        self.order = order

        # Register non-trainable transfer function buffer [1, 1, H, W]
        h_mask = compute_butterworth_mask(height, width, d0, order)
        self.register_buffer("filter", h_mask, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Butterworth high-pass filter and standardize residual.
        
        Args:
            x: Input image tensor of shape [B, C, H, W] (typically grayscale C=1 or RGB C=3).
               If C == 3, converts to grayscale first.
               
        Returns:
            Standardized high-pass residual of shape [B, 1, H, W].
        """
        if x.shape[1] == 3:
            x_gray = rgb_to_grayscale(x)
        elif x.shape[1] == 1:
            x_gray = x
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")

        b, c, h, w = x_gray.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Input spatial dimensions ({h}, {w}) do not match filter dimensions "
                f"({self.height}, {self.width})"
            )

        # 1. 2D FFT and shift zero-frequency component to center
        # norm='backward' ensures standard scaling
        f = torch.fft.fftshift(torch.fft.fft2(x_gray, dim=(-2, -1)), dim=(-2, -1))

        # 2. High-pass filter in frequency domain
        f_hp = f * self.filter

        # 3. Inverse shift and 2D IFFT to return to spatial domain
        x_hp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_hp, dim=(-2, -1)), dim=(-2, -1)))

        # 4. Per-sample standardization (zero mean, unit variance)
        mean = x_hp.mean(dim=(-2, -1), keepdim=True)
        std = x_hp.std(dim=(-2, -1), keepdim=True)
        residual = (x_hp - mean) / (std + 1e-7)

        return residual

    def compute_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-magnitude shifted spectrum for explainability plots.
        
        Args:
            x: Grayscale or RGB tensor [B, C, H, W].
            
        Returns:
            Log magnitude spectrum [B, 1, H, W].
        """
        if x.shape[1] == 3:
            x = rgb_to_grayscale(x)
        f = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(f)
        return torch.log1p(mag)
