"""Baseline and Ablation Architectures for KryptosID.

Implements Section 6.2 of DESIGN.md:
- SpatialOnlyNet: ConvNeXt-S spatial only (measures value of spectral stream)
- SpectralOnlyNet: ResNet-34 spectral only (measures value of spatial stream)
- NoButterworthNet: Fusion without the Butterworth filter (measures value of high-pass filter)
- RandomInitResNetNet: KryptosIDNet with random init ResNet-34 (tests pretraining claim)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from core.frequency import ButterworthHighPassFilter2D, rgb_to_grayscale
from core.fusion import CrossAttentionFusion
from core.model import (
    ClassificationHead,
    MockSpatialBackbone,
    MockSpectralBackbone,
    KryptosIDNet,
)


class SpatialOnlyNet(nn.Module):
    """ConvNeXt-S spatial stream only baseline."""

    def __init__(
        self,
        pretrained: bool = False,
        token_dim: int = 256,
        dropout: float = 0.2,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        if HAS_TIMM and not use_mock:
            self.backbone = timm.create_model("convnext_small", pretrained=pretrained, num_classes=0)
            feat_dim = 768
        else:
            self.backbone = MockSpatialBackbone(in_channels=3, out_channels=768)
            feat_dim = 768

        self.proj = nn.Linear(feat_dim, token_dim)
        self.head = ClassificationHead(in_features=token_dim, hidden_dim=64, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on RGB tensor [B, 3, 224, 224]."""
        feat = self.backbone.forward_features(x)
        if feat.ndim == 4:
            if feat.shape[1] == 768:
                tokens = feat.flatten(2).transpose(1, 2)
            else:
                b, h, w, c = feat.shape
                tokens = feat.view(b, h * w, c)
        else:
            tokens = feat

        tokens_proj = self.proj(tokens)  # [B, 49, 256]
        pooled = tokens_proj.mean(dim=1)  # [B, 256]
        return self.head(pooled)


class SpectralOnlyNet(nn.Module):
    """Butterworth HPF + ResNet-34 spectral stream only baseline."""

    def __init__(
        self,
        pretrained: bool = False,
        d0: float = 25.0,
        filter_order: int = 2,
        token_dim: int = 256,
        dropout: float = 0.2,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.hpf = ButterworthHighPassFilter2D(height=224, width=224, d0=d0, order=filter_order)
        if HAS_TIMM and not use_mock:
            self.backbone = timm.create_model("resnet34", in_chans=1, pretrained=pretrained, num_classes=0)
            feat_dim = 512
        else:
            self.backbone = MockSpectralBackbone(in_channels=1, out_channels=512)
            feat_dim = 512

        self.proj = nn.Linear(feat_dim, token_dim)
        self.head = ClassificationHead(in_features=token_dim, hidden_dim=64, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on Grayscale [B, 1, 224, 224] or RGB [B, 3, 224, 224]."""
        residual = self.hpf(x)
        feat = self.backbone.forward_features(residual)
        if feat.ndim == 4:
            tokens = feat.flatten(2).transpose(1, 2)
        else:
            tokens = feat

        tokens_proj = self.proj(tokens)  # [B, 49, 256]
        pooled = tokens_proj.mean(dim=1)  # [B, 256]
        return self.head(pooled)


class NoButterworthNet(nn.Module):
    """Dual-stream fusion without the Butterworth filter ablation."""

    def __init__(
        self,
        pretrained_spatial: bool = False,
        pretrained_spectral: bool = False,
        token_dim: int = 256,
        dropout: float = 0.2,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        if HAS_TIMM and not use_mock:
            self.spatial_backbone = timm.create_model("convnext_small", pretrained=pretrained_spatial, num_classes=0)
            self.spectral_backbone = timm.create_model("resnet34", in_chans=1, pretrained=pretrained_spectral, num_classes=0)
            spatial_dim, spectral_dim = 768, 512
        else:
            self.spatial_backbone = MockSpatialBackbone(in_channels=3, out_channels=768)
            self.spectral_backbone = MockSpectralBackbone(in_channels=1, out_channels=512)
            spatial_dim, spectral_dim = 768, 512

        self.proj_spatial = nn.Linear(spatial_dim, token_dim)
        self.proj_spectral = nn.Linear(spectral_dim, token_dim)
        self.fusion = CrossAttentionFusion(dim=token_dim, num_heads=1, dropout=dropout)
        self.head = ClassificationHead(in_features=token_dim, hidden_dim=64, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass bypassing Butterworth filter."""
        if x.shape[1] == 3:
            x_spatial = x
            x_gray = rgb_to_grayscale(x)
        else:
            x_spatial = x.repeat(1, 3, 1, 1)
            x_gray = x

        # Spatial stream
        sp_feat = self.spatial_backbone.forward_features(x_spatial)
        if sp_feat.ndim == 4:
            sp_tokens_raw = sp_feat.flatten(2).transpose(1, 2) if sp_feat.shape[1] == 768 else sp_feat.view(sp_feat.shape[0], -1, 768)
        else:
            sp_tokens_raw = sp_feat
        sp_tokens = self.proj_spatial(sp_tokens_raw)

        # Spectral stream directly on standardized grayscale (No Butterworth)
        mean = x_gray.mean(dim=(-2, -1), keepdim=True)
        std = x_gray.std(dim=(-2, -1), keepdim=True)
        x_norm_gray = (x_gray - mean) / (std + 1e-7)

        pe_feat = self.spectral_backbone.forward_features(x_norm_gray)
        pe_tokens_raw = pe_feat.flatten(2).transpose(1, 2) if pe_feat.ndim == 4 else pe_feat
        pe_tokens = self.proj_spectral(pe_tokens_raw)

        # Fusion & Head
        fused, _ = self.fusion(sp_tokens, pe_tokens)
        return self.head(fused)


def build_ablation_model(model_type: str, use_mock: bool = False) -> nn.Module:
    """Factory function for models and ablations in Section 6.2."""
    if model_type == "kryptos_id_net":
        return KryptosIDNet(pretrained_spatial=False, pretrained_spectral=True, use_mock_backbones=use_mock)
    elif model_type == "spatial_only":
        return SpatialOnlyNet(pretrained=False, use_mock=use_mock)
    elif model_type == "spectral_only":
        return SpectralOnlyNet(pretrained=False, use_mock=use_mock)
    elif model_type == "no_butterworth":
        return NoButterworthNet(pretrained_spatial=False, pretrained_spectral=False, use_mock=use_mock)
    elif model_type == "random_init_resnet":
        return KryptosIDNet(pretrained_spatial=False, pretrained_spectral=False, use_mock_backbones=use_mock)
    else:
        raise ValueError(f"Unknown ablation model type: {model_type}")
