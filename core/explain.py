"""Explainability Engine for KryptosID.

Implements Section 7 of DESIGN.md:
1. Grad-CAM on Stream A: Gradients of the output logit with respect to the last ConvNeXt stage,
   visualized as a 224x224 heatmap overlaid on the input image.
2. Cross-Attention Maps: 49x49 attention weights reshaped to 7x7 and upsampled to 224x224
   (spatial-queries-spectral and spectral-queries-spatial).
3. Frequency-Band Attribution: Spectrum partitioned into 8 concentric annular rings;
   each band is ablated in turn to compute logit attribution delta.
4. Spectrum Plot: 2D log-magnitude FFT spectrum overlaid with the Butterworth D0=25 cutoff circle.
5. All explanation artifacts saved as signed/hashed PNGs for review.
"""

from typing import Dict, List, Optional, Tuple, Any
import hashlib
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.frequency import rgb_to_grayscale
from core.model import KryptosIDNet


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a generated file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    background_rgb: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a [0, 1] float heatmap onto an RGB uint8 image [H, W, 3]."""
    h, w, _ = background_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized * 255.0, 0, 255))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.clip(
        (1.0 - alpha) * background_rgb.astype(np.float32) + alpha * heatmap_colored_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return blended


def generate_gradcam(
    model: nn.Module,
    input_spatial: torch.Tensor,
    input_spectral: Optional[torch.Tensor] = None,
    raw_rgb_np: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Compute Grad-CAM for Stream A spatial features (Section 7.1).
    
    Args:
        model: KryptosIDNet model instance.
        input_spatial: [1, 3, 224, 224] spatial tensor.
        input_spectral: Optional [1, 1, 224, 224] spectral tensor.
        raw_rgb_np: Optional [224, 224, 3] uint8 image for visual overlay.
        output_path: Optional path to save PNG.
        
    Returns:
        Tuple of (blended_image_rgb [224, 224, 3], output_file_path).
    """
    model.eval()

    # Find target layer in spatial_backbone: last convolutional or stage layer
    target_layer = None
    if hasattr(model, "spatial_backbone"):
        sb = model.spatial_backbone
        if hasattr(sb, "stages"):
            target_layer = sb.stages[-1]
        elif hasattr(sb, "blocks"):
            target_layer = sb.blocks[-1].norm1 if hasattr(sb.blocks[-1], "norm1") else sb.blocks[-1]
        elif hasattr(sb, "conv"):
            target_layer = sb.conv[-1]
        else:
            for mod in reversed(list(sb.modules())):
                if isinstance(mod, (nn.Conv2d, nn.Linear)):
                    target_layer = mod
                    break

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handles = []
    if target_layer is not None:
        handles.append(target_layer.register_forward_hook(forward_hook))
        handles.append(target_layer.register_full_backward_hook(backward_hook))

    # Enable grad for input and forward pass
    input_spatial_grad = input_spatial.clone().detach().requires_grad_(True)
    if input_spectral is None:
        input_spectral = rgb_to_grayscale(input_spatial)

    model.zero_grad()
    logits = model((input_spatial_grad, input_spectral))
    score = logits[0, 0]
    score.backward()

    for h in handles:
        h.remove()

    if activations and gradients:
        act = activations[0].detach()  # [1, C, H, W] or [1, H, W, C] or [1, N, C]
        grad = gradients[0].detach()

        if act.ndim == 4:
            if act.shape[1] == 768 or act.shape[1] < act.shape[-1]:
                # Channel first: [1, C, H, W]
                weights = grad.mean(dim=(-2, -1), keepdim=True)  # [1, C, 1, 1]
                cam = (weights * act).sum(dim=1, keepdim=True)    # [1, 1, H, W]
            else:
                # Channel last: [1, H, W, C]
                weights = grad.mean(dim=(1, 2), keepdim=True)     # [1, 1, 1, C]
                cam = (weights * act).sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        elif act.ndim == 3:
            # ViT token format: [1, 197, C] -> strip CLS token to get [1, 196, C]
            act_tokens = act[:, 1:, :] if act.shape[1] == 197 else act
            grad_tokens = grad[:, 1:, :] if grad.shape[1] == 197 else grad
            weights = grad_tokens.mean(dim=1, keepdim=True)  # [1, 1, C]
            cam_tokens = (weights * act_tokens).sum(dim=-1, keepdim=True)  # [1, 196, 1]
            if cam_tokens.shape[1] == 196:
                cam = cam_tokens.view(1, 1, 14, 14)
            else:
                cam = torch.ones(1, 1, 7, 7)
        else:
            cam = torch.ones(1, 1, 7, 7)
    else:
        # Fallback to gradient of input tensor if no hook captured
        cam = input_spatial_grad.grad.abs().mean(dim=1, keepdim=True)

    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()
    norm_denom = cam_np.max() - cam_np.min()
    if norm_denom > 1e-7:
        cam_np = (cam_np - cam_np.min()) / norm_denom
    else:
        cam_np = np.zeros_like(cam_np)

    # Prepare background
    if raw_rgb_np is not None:
        bg = raw_rgb_np
    else:
        # Denormalize ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        unnorm = input_spatial[0].permute(1, 2, 0).detach().cpu().numpy() * std + mean
        bg = np.uint8(np.clip(unnorm * 255.0, 0, 255))

    blended = overlay_heatmap_on_image(cam_np, bg, alpha=0.55)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    return blended, output_path or ""


def generate_cross_attention_maps(
    attn_sp_to_pe: torch.Tensor,
    attn_pe_to_sp: torch.Tensor,
    raw_rgb_np: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate and export cross-attention heatmaps (Section 7.2).
    
    Args:
        attn_sp_to_pe: Attention weights spatial queries spectral [1, 49, 49].
        attn_pe_to_sp: Attention weights spectral queries spatial [1, 49, 49].
        raw_rgb_np: Optional background RGB image.
        output_dir: Directory to save PNGs.
        
    Returns:
        Dictionary with paths to generated attention maps.
    """
    # Average across query tokens -> [49]
    sp_weights = attn_sp_to_pe[0].mean(dim=0).detach().cpu().numpy()
    pe_weights = attn_pe_to_sp[0].mean(dim=0).detach().cpu().numpy()

    # Reshape 49 to 7x7 grid
    sp_grid = sp_weights.reshape(7, 7)
    pe_grid = pe_weights.reshape(7, 7)

    # Normalize
    sp_norm = (sp_grid - sp_grid.min()) / (sp_grid.max() - sp_grid.min() + 1e-7)
    pe_norm = (pe_grid - pe_grid.min()) / (pe_grid.max() - pe_grid.min() + 1e-7)

    bg = raw_rgb_np if raw_rgb_np is not None else np.ones((224, 224, 3), dtype=np.uint8) * 200

    sp_blend = overlay_heatmap_on_image(sp_norm, bg, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
    pe_blend = overlay_heatmap_on_image(pe_norm, bg, alpha=0.5, colormap=cv2.COLORMAP_MAGMA)

    paths = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sp_path = os.path.join(output_dir, "attn_spatial_queries_spectral.png")
        pe_path = os.path.join(output_dir, "attn_spectral_queries_spatial.png")
        cv2.imwrite(sp_path, cv2.cvtColor(sp_blend, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pe_path, cv2.cvtColor(pe_blend, cv2.COLOR_RGB2BGR))
        paths["attn_spatial_queries_spectral"] = sp_path
        paths["attn_spectral_queries_spatial"] = pe_path

    return paths


def generate_frequency_band_attribution(
    model: KryptosIDNet,
    input_spatial: torch.Tensor,
    input_spectral: Optional[torch.Tensor] = None,
    num_bands: int = 8,
    output_path: Optional[str] = None,
) -> Tuple[List[float], str]:
    """Compute and plot frequency band attribution across 8 concentric rings (Section 7.3).
    
    Ablates each concentric annular frequency band in turn and records score delta.
    
    Returns:
        Tuple of (band_attributions_list, output_file_path).
    """
    model.eval()
    if input_spectral is None:
        input_spectral = rgb_to_grayscale(input_spatial)

    b, c, h, w = input_spectral.shape
    center_u, center_v = h / 2.0, w / 2.0

    u = torch.arange(h, dtype=torch.float32, device=input_spectral.device) - center_u
    v = torch.arange(w, dtype=torch.float32, device=input_spectral.device) - center_v
    grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
    dist = torch.sqrt(grid_u**2 + grid_v**2)

    # 1. Base logit
    with torch.no_grad():
        base_logit = model((input_spatial, input_spectral))[0, 0].item()

    # 2. 2D FFT shifted
    f = torch.fft.fftshift(torch.fft.fft2(input_spectral, dim=(-2, -1)), dim=(-2, -1))

    band_width = 112.0 / num_bands
    attributions = []
    band_labels = []

    for k in range(num_bands):
        r_inner = k * band_width
        r_outer = (k + 1) * band_width
        band_labels.append(f"B{k+1}\n[{int(r_inner)}-{int(r_outer)}px]")

        in_band = (dist >= r_inner) & (dist < r_outer)

        # Mask out band k
        f_ablated = f.clone()
        f_ablated[:, :, in_band] = 0.0

        # Inverse FFT
        x_ablated_gray = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_ablated, dim=(-2, -1)), dim=(-2, -1)))

        with torch.no_grad():
            ablated_logit = model((input_spatial, x_ablated_gray))[0, 0].item()

        # Delta score: positive means band drove decision towards manipulated
        delta = base_logit - ablated_logit
        attributions.append(float(delta))

    # Plot attribution bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    colors = ["#d9534f" if a >= 0 else "#5cb85c" for a in attributions]
    bars = ax.bar(range(num_bands), attributions, color=colors, width=0.6, edgecolor="#333333", linewidth=0.8)

    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(num_bands))
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.set_ylabel("Logit Attribution (Δ Score)", fontsize=9, fontweight="bold")
    ax.set_title("KryptosID: Frequency-Band Attribution (8 Concentric Annuli)", fontsize=10, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -8),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7,
            fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return attributions, output_path or ""


def generate_spectrum_plot(
    input_tensor: torch.Tensor,
    d0: float = 25.0,
    output_path: Optional[str] = None,
) -> str:
    """Generate 2D log-magnitude FFT spectrum plot with Butterworth D0 cutoff circle (Section 7.4)."""
    if input_tensor.shape[1] == 3:
        x_gray = rgb_to_grayscale(input_tensor)
    else:
        x_gray = input_tensor

    f = torch.fft.fftshift(torch.fft.fft2(x_gray, dim=(-2, -1)), dim=(-2, -1))
    mag = torch.abs(f)[0, 0].detach().cpu().numpy()
    log_mag = np.log1p(mag)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    im = ax.imshow(log_mag, cmap="inferno", extent=[-112, 112, -112, 112])

    # Butterworth cutoff circle at D0
    circle = plt.Circle((0, 0), d0, color="#00ffff", fill=False, linestyle="--", linewidth=1.8, label=f"D0={int(d0)} Cutoff")
    ax.add_patch(circle)

    ax.scatter([0], [0], color="#00ffff", s=15, marker="+", label="DC (0, 0)")
    ax.set_title("Log-Magnitude Spectrum & Butterworth Cutoff", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency U", fontsize=8)
    ax.set_ylabel("Frequency V", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Log(1 + |F|)")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path or ""


def generate_all_explanations(
    model: KryptosIDNet,
    input_spatial: torch.Tensor,
    output_dir: str,
    raw_rgb_np: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, str]]:
    """Compute and export all 4 forensic explanations with individual SHA-256 integrity hashes.
    
    Returns:
        Dictionary mapping explanation key to {path, sha256}.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_spectral = rgb_to_grayscale(input_spatial)

    # 1. Forward pass with explanations
    with torch.no_grad():
        _, exp_dict = model((input_spatial, input_spectral), return_explanations=True)

    # Paths
    gradcam_path = os.path.join(output_dir, "gradcam_stream_a.png")
    band_attr_path = os.path.join(output_dir, "frequency_band_attribution.png")
    spectrum_path = os.path.join(output_dir, "spectrum_plot.png")

    # Generate Grad-CAM
    generate_gradcam(
        model=model,
        input_spatial=input_spatial,
        input_spectral=input_spectral,
        raw_rgb_np=raw_rgb_np,
        output_path=gradcam_path,
    )

    # Generate Cross-Attention Maps
    attn_paths = generate_cross_attention_maps(
        attn_sp_to_pe=exp_dict["attn_sp_to_pe"],
        attn_pe_to_sp=exp_dict["attn_pe_to_sp"],
        raw_rgb_np=raw_rgb_np,
        output_dir=output_dir,
    )

    # Generate Frequency Band Attribution
    generate_frequency_band_attribution(
        model=model,
        input_spatial=input_spatial,
        input_spectral=input_spectral,
        num_bands=8,
        output_path=band_attr_path,
    )

    # Generate Spectrum Plot
    generate_spectrum_plot(
        input_tensor=input_spatial,
        d0=25.0,
        output_path=spectrum_path,
    )

    all_files = {
        "gradcam": gradcam_path,
        "attn_spatial": attn_paths.get("attn_spatial_queries_spectral", ""),
        "attn_spectral": attn_paths.get("attn_spectral_queries_spatial", ""),
        "band_attribution": band_attr_path,
        "spectrum_plot": spectrum_path,
    }

    result = {}
    for key, path in all_files.items():
        if os.path.exists(path):
            result[key] = {
                "file_path": path,
                "sha256": compute_file_sha256(path),
            }

    return result
