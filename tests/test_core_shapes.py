"""Unit and Tensor Shape Tests for KryptosID Phase 1 Core Components.

Tests cover:
1. Butterworth 2D High-Pass Filter mathematical properties and standardization.
2. CrossAttentionBlock and CrossAttentionFusion tensor dimensions and probability axioms.
3. ClassificationHead forward pass and dimensions.
4. KryptosIDNet end-to-end forward pass, explanation dictionary, and temperature calibration.
"""

import math
import pytest
import torch

from core.frequency import (
    ButterworthHighPassFilter2D,
    compute_butterworth_mask,
    rgb_to_grayscale,
)
from core.fusion import CrossAttentionBlock, CrossAttentionFusion
from core.model import ClassificationHead, KryptosIDNet


def test_compute_butterworth_mask_properties():
    """Verify Butterworth transfer function at DC, D0, and Nyquist according to DESIGN.md Section 4.2."""
    h, w = 224, 224
    d0 = 25.0
    order = 2
    mask = compute_butterworth_mask(height=h, width=w, d0=d0, order=order)

    assert mask.shape == (1, 1, 224, 224), f"Expected shape (1, 1, 224, 224), got {mask.shape}"

    # DC component is at index (112, 112)
    center_u, center_v = 112, 112
    dc_val = mask[0, 0, center_u, center_v].item()
    assert dc_val == 0.0, f"DC response must be exactly 0.0, got {dc_val}"

    # At distance D0 = 25.0 from center, H(D0) must be 0.5 (-3 dB cutoff)
    # Point at (112 + 25, 112) = (137, 112)
    cutoff_val = mask[0, 0, center_u + int(d0), center_v].item()
    assert math.isclose(cutoff_val, 0.5, rel_tol=1e-5), f"H(D0) must be 0.5, got {cutoff_val}"

    # At Nyquist frequency (index distance 112 from center, e.g. (224-1, 112) or (0, 112))
    # D = 112 -> H = 1 / (1 + (25/112)^4) ~= 1 / (1 + 0.00248) ~= 0.9975
    nyquist_val = mask[0, 0, 0, center_v].item()
    expected_nyquist = 1.0 / (1.0 + (25.0 / 112.0) ** (2 * order))
    assert math.isclose(nyquist_val, expected_nyquist, rel_tol=1e-4)
    assert nyquist_val > 0.99, f"Nyquist response must exceed 0.99, got {nyquist_val}"


def test_butterworth_filter_module_forward_and_standardization():
    """Verify ButterworthHighPassFilter2D forward pass and per-sample standardization."""
    b, c, h, w = 4, 3, 224, 224
    filter_module = ButterworthHighPassFilter2D(height=h, width=w, d0=25.0, order=2)

    # Test with RGB input
    x_rgb = torch.randn(b, c, h, w)
    residual_rgb = filter_module(x_rgb)
    assert residual_rgb.shape == (b, 1, h, w), f"Expected shape (4, 1, 224, 224), got {residual_rgb.shape}"

    # Verify per-sample standardization: mean ~= 0, std ~= 1
    for i in range(b):
        sample_mean = residual_rgb[i].mean().item()
        sample_std = residual_rgb[i].std().item()
        assert abs(sample_mean) < 1e-4, f"Sample {i} mean must be ~0, got {sample_mean}"
        assert abs(sample_std - 1.0) < 1e-3, f"Sample {i} std must be ~1, got {sample_std}"

    # Test with 1-channel Grayscale input
    x_gray = torch.randn(b, 1, h, w)
    residual_gray = filter_module(x_gray)
    assert residual_gray.shape == (b, 1, h, w)


def test_cross_attention_block():
    """Verify single cross-attention block computation and shapes."""
    b, n_q, n_k, dim = 2, 49, 49, 256
    block = CrossAttentionBlock(dim=dim, num_heads=1)

    q = torch.randn(b, n_q, dim)
    k = torch.randn(b, n_k, dim)
    v = torch.randn(b, n_k, dim)

    out, attn = block(q, k, v)
    assert out.shape == (b, n_q, dim)
    assert attn.shape == (b, n_q, n_k)

    # Attention weights must sum to 1 along key dimension
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_cross_attention_fusion():
    """Verify bidirectional CrossAttentionFusion per Section 4.3 of DESIGN.md."""
    b, tokens, dim = 2, 49, 256
    fusion = CrossAttentionFusion(dim=dim, num_heads=1)

    spatial_tokens = torch.randn(b, tokens, dim)
    spectral_tokens = torch.randn(b, tokens, dim)

    fused, (sp_attn, pe_attn) = fusion(spatial_tokens, spectral_tokens)

    # Fused vector pooled across both streams
    assert fused.shape == (b, dim), f"Fused vector must have shape ({b}, {dim}), got {fused.shape}"
    assert sp_attn.shape == (b, tokens, tokens), f"Spatial attention map must be ({b}, 49, 49)"
    assert pe_attn.shape == (b, tokens, tokens), f"Spectral attention map must be ({b}, 49, 49)"

    # Softmax row sums check
    assert torch.allclose(sp_attn.sum(dim=-1), torch.ones(b, tokens), atol=1e-5)
    assert torch.allclose(pe_attn.sum(dim=-1), torch.ones(b, tokens), atol=1e-5)


def test_classification_head():
    """Verify classification head LayerNorm(256)->Linear(256,64)->GELU->Dropout->Linear(64,1)."""
    b, dim = 4, 256
    head = ClassificationHead(in_features=dim, hidden_dim=64, dropout=0.2)
    fused = torch.randn(b, dim)

    logits = head(fused)
    assert logits.shape == (b, 1), f"Expected logits shape ({b}, 1), got {logits.shape}"


def test_kryptos_id_net_shapes_and_explanations():
    """End-to-end forward pass test for KryptosIDNet with mock backbones."""
    b = 2
    model = KryptosIDNet(
        pretrained_spatial=False,
        pretrained_spectral=False,
        use_mock_backbones=True,
    )
    model.eval()

    x = torch.randn(b, 3, 224, 224)

    # 1. Standard forward pass
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (b, 1), f"Logits shape should be ({b}, 1), got {logits.shape}"

    # 2. Probability calibration
    probs = model.predict_probability(logits)
    assert probs.shape == (b, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]"

    # 3. Forward pass with explanations
    with torch.no_grad():
        logits_exp, explanations = model(x, return_explanations=True)

    assert logits_exp.shape == (b, 1)
    assert "spatial_tokens" in explanations
    assert "spectral_tokens" in explanations
    assert "residual_hp" in explanations
    assert "attn_sp_to_pe" in explanations
    assert "attn_pe_to_sp" in explanations
    assert "fused_vector" in explanations

    assert explanations["spatial_tokens"].shape == (b, 49, 256)
    assert explanations["spectral_tokens"].shape == (b, 49, 256)
    assert explanations["residual_hp"].shape == (b, 1, 224, 224)
    assert explanations["fused_vector"].shape == (b, 256)
    assert explanations["attn_sp_to_pe"].shape == (b, 49, 49)
    assert explanations["attn_pe_to_sp"].shape == (b, 49, 49)

    # 4. Temperature updating
    model.set_temperature(1.75)
    scaled_probs = model.predict_probability(logits)
    assert model.temperature.item() == 1.75


def test_kryptos_id_net_with_timm_backbones():
    """Verify KryptosIDNet with actual timm backbones (unpretrained for offline test speed)."""
    import timm
    b = 1
    model = KryptosIDNet(
        pretrained_spatial=False,
        pretrained_spectral=False,
        use_mock_backbones=False,
    )
    model.eval()

    x = torch.randn(b, 3, 224, 224)
    with torch.no_grad():
        logits, explanations = model(x, return_explanations=True)

    assert logits.shape == (b, 1)
    assert explanations["spatial_tokens"].shape == (b, 49, 256)
    assert explanations["spectral_tokens"].shape == (b, 49, 256)
    assert explanations["fused_vector"].shape == (b, 256)

