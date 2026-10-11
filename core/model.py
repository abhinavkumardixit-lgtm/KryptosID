"""KryptosIDNet: Dual-Stream Spatial-Spectral Deepfake Detection Architecture.

Implements Section 4 of DESIGN.md:
- Stream A (Spatial): ConvNeXt-Small backbone (using forward_features),
  feature map [B, 768, 7, 7] -> 49 tokens [B, 49, 768] -> projected to [B, 49, 256].
- Stream B (Spectral): Butterworth 2D high-pass filter -> standardized residual [B, 1, 224, 224]
  -> ResNet-34 (in_chans=1, using forward_features), feature map [B, 512, 7, 7]
  -> 49 tokens [B, 49, 512] -> projected to [B, 49, 256].
- Cross-Attention Fusion: Bidirectional cross-attention between spatial and spectral tokens,
  mean pooled and summed -> fused vector [B, 256].
- Calibrated Classification Head:
  LayerNorm(256) -> Linear(256, 64) -> GELU -> Dropout(0.2) -> Linear(64, 1) -> logit.
  Calibrated probability: sigmoid(logit / T).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from core.frequency import ButterworthHighPassFilter2D, rgb_to_grayscale
from core.fusion import CrossAttentionFusion


class ClassificationHead(nn.Module):
    """Calibrated classification head according to Section 4.4 of DESIGN.md."""

    def __init__(self, in_features: int = 256, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from fused vector [B, in_features] -> [B, 1]."""
        return self.net(x)


class MockSpatialBackbone(nn.Module):
    """Mock spatial backbone used when timm is not installed or for lightweight shape testing."""

    def __init__(self, in_channels: int = 3, out_channels: int = 768) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MockSpectralBackbone(nn.Module):
    """Mock spectral backbone used when timm is not installed or for lightweight shape testing."""

    def __init__(self, in_channels: int = 1, out_channels: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class KryptosIDNet(nn.Module):
    """KryptosIDNet dual-stream forensic architecture."""

    def __init__(
        self,
        spatial_backbone: str = "convnext_small",
        pretrained_spatial: bool = False,
        pretrained_spectral: bool = False,
        d0: float = 25.0,
        filter_order: int = 2,
        token_dim: int = 256,
        num_attention_heads: int = 1,
        dropout: float = 0.2,
        temperature: float = 1.0,
        use_mock_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_backbone_name = spatial_backbone
        self.token_dim = token_dim
        self.num_attention_heads = num_attention_heads
        self.initial_temperature = temperature

        # Registered non-trainable temperature buffer for calibrated probabilities
        self.register_buffer("temperature", torch.tensor([temperature], dtype=torch.float32))

        # Stream B Preprocessing: Butterworth 2D HPF
        self.hpf = ButterworthHighPassFilter2D(height=224, width=224, d0=d0, order=filter_order)

        # Stream A: Spatial Backbone (ConvNeXt or Vision Transformer)
        if HAS_TIMM and not use_mock_backbones:
            self.spatial_backbone = timm.create_model(
                spatial_backbone,
                pretrained=pretrained_spatial,
                num_classes=0,
            )
            spatial_feat_dim = getattr(self.spatial_backbone, "num_features", 768)
        else:
            self.spatial_backbone = MockSpatialBackbone(in_channels=3, out_channels=768)
            spatial_feat_dim = 768

        # Stream B: Spectral Backbone (ResNet-34, in_chans=1)
        if HAS_TIMM and not use_mock_backbones:
            self.spectral_backbone = timm.create_model(
                "resnet34",
                in_chans=1,
                pretrained=pretrained_spectral,
                num_classes=0,
            )
            spectral_feat_dim = 512
        else:
            self.spectral_backbone = MockSpectralBackbone(in_channels=1, out_channels=512)
            spectral_feat_dim = 512

        # Token projections to token_dim (256)
        self.proj_spatial = nn.Linear(spatial_feat_dim, token_dim)
        self.proj_spectral = nn.Linear(spectral_feat_dim, token_dim)

        # Bidirectional Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            dim=token_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Calibrated Classification Head
        self.head = ClassificationHead(
            in_features=token_dim,
            hidden_dim=64,
            dropout=dropout,
        )

    def set_temperature(self, t: float) -> None:
        """Update calibration temperature T."""
        if t <= 0:
            raise ValueError(f"Temperature must be positive, got {t}")
        self.temperature.copy_(torch.tensor([t], dtype=torch.float32))

    def freeze_backbones(self) -> None:
        """Freeze both backbones for stage-1 warm-up training (Section 6.1)."""
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
        for param in self.spectral_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self, last_stages_only: bool = True) -> None:
        """Unfreeze backbones for stage-2 fine-tuning (Section 6.1)."""
        if not last_stages_only:
            for param in self.spatial_backbone.parameters():
                param.requires_grad = True
            for param in self.spectral_backbone.parameters():
                param.requires_grad = True
            return

        # Fine-tune only the last stages:
        # For ConvNeXt, stages[3] (the final block)
        if hasattr(self.spatial_backbone, "stages"):
            for i, stage in enumerate(self.spatial_backbone.stages):
                req = (i >= len(self.spatial_backbone.stages) - 1)
                for p in stage.parameters():
                    p.requires_grad = req
        elif hasattr(self.spatial_backbone, "blocks"):
            for i, block in enumerate(self.spatial_backbone.blocks):
                req = (i >= len(self.spatial_backbone.blocks) - 2)
                for p in block.parameters():
                    p.requires_grad = req
        else:
            for p in self.spatial_backbone.parameters():
                p.requires_grad = True

        # For ResNet-34, layer4 (the final residual stage)
        if hasattr(self.spectral_backbone, "layer4"):
            for p in self.spectral_backbone.layer4.parameters():
                p.requires_grad = True
        else:
            for p in self.spectral_backbone.parameters():
                p.requires_grad = True

    def _extract_spatial_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract spatial tokens [B, 49, 256] from input [B, 3, 224, 224]."""
        feat = self.spatial_backbone.forward_features(x)

        if feat.ndim == 4:
            in_c = self.proj_spatial.in_features
            if feat.shape[1] == in_c:
                b, c, h, w = feat.shape
                tokens_raw = feat.flatten(2).transpose(1, 2)
            elif feat.shape[-1] == in_c:
                b, h, w, c = feat.shape
                tokens_raw = feat.view(b, h * w, c)
            else:
                b = feat.shape[0]
                tokens_raw = feat.flatten(2).transpose(1, 2)
        elif feat.ndim == 3:
            # Vision Transformer patch tokens: [B, 197, C] or [B, 196, C]
            b, n, c = feat.shape
            if n == 197:  # Discard CLS token to retain 14x14 spatial patches
                patch_tokens = feat[:, 1:, :]
                n = 196
            else:
                patch_tokens = feat

            if n == 196:
                # 14x14 grid -> pool to 7x7 grid (49 tokens)
                grid = patch_tokens.transpose(1, 2).reshape(b, c, 14, 14)
                pooled = F.adaptive_avg_pool2d(grid, (7, 7))
                tokens_raw = pooled.flatten(2).transpose(1, 2)
            else:
                tokens_raw = patch_tokens
        else:
            tokens_raw = feat

        tokens_proj = self.proj_spatial(tokens_raw)  # [B, 49, 256]
        return tokens_proj, feat

    def _extract_spectral_tokens(self, x_gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Butterworth HPF and extract spectral tokens [B, 49, 256]."""
        # High-pass filter residual with per-sample standardization: [B, 1, 224, 224]
        residual = self.hpf(x_gray)

        # ResNet-34 forward_features: [B, 512, 7, 7]
        feat = self.spectral_backbone.forward_features(residual)

        if feat.ndim == 4:
            if feat.shape[1] == 512:
                b, c, h, w = feat.shape
                tokens_raw = feat.flatten(2).transpose(1, 2)  # [B, 49, 512]
            elif feat.shape[-1] == 512:
                b, h, w, c = feat.shape
                tokens_raw = feat.view(b, h * w, c)
            else:
                tokens_raw = feat.flatten(2).transpose(1, 2)
        else:
            tokens_raw = feat

        tokens_proj = self.proj_spectral(tokens_raw)  # [B, 49, 256]
        return tokens_proj, feat, residual

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        return_explanations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through KryptosIDNet.
        
        Args:
            x: Either RGB tensor of shape [B, 3, 224, 224], or a tuple of
               (x_spatial [B, 3, 224, 224], x_gray [B, 1, 224, 224]).
            return_explanations: If True, returns dictionary with attention maps,
                                 tokens, and HPF residuals for explainability.
                                 
        Returns:
            Logits of shape [B, 1], or (logits, explanations_dict).
        """
        if isinstance(x, (tuple, list)):
            x_spatial, x_spectral = x[0], x[1]
        else:
            x_spatial = x
            x_spectral = rgb_to_grayscale(x) if x.shape[1] == 3 else x

        # 1. Spatial stream
        spatial_tokens, spatial_feat = self._extract_spatial_tokens(x_spatial)

        # 2. Spectral stream
        spectral_tokens, spectral_feat, residual = self._extract_spectral_tokens(x_spectral)

        # 3. Cross-Attention Fusion
        fused, (attn_sp_to_pe, attn_pe_to_sp) = self.fusion(spatial_tokens, spectral_tokens)

        # 4. Calibrated Classification Head
        logits = self.head(fused)  # [B, 1]

        if return_explanations:
            explanations = {
                "spatial_tokens": spatial_tokens,
                "spectral_tokens": spectral_tokens,
                "spatial_features": spatial_feat,
                "spectral_features": spectral_feat,
                "residual_hp": residual,
                "fused_vector": fused,
                "attn_sp_to_pe": attn_sp_to_pe,
                "attn_pe_to_sp": attn_pe_to_sp,
            }
            return logits, explanations

        return logits

    def predict_probability(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute calibrated probability: sigmoid(logits / T)."""
        temp = temperature if temperature is not None else self.temperature.item()
        return torch.sigmoid(logits / temp)
