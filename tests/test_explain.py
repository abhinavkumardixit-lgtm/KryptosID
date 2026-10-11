"""Tests for Phase 5: Explainability Engine and Forensic Visualization Artifacts."""

import os
import tempfile
import numpy as np
import pytest
import torch

from core.explain import (
    compute_file_sha256,
    generate_all_explanations,
    generate_cross_attention_maps,
    generate_frequency_band_attribution,
    generate_gradcam,
    generate_spectrum_plot,
)
from core.model import KryptosIDNet


@pytest.fixture
def test_model():
    model = KryptosIDNet(use_mock_backbones=True)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    # Deterministic spatial tensor [1, 3, 224, 224]
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


def test_gradcam_stream_a(test_model, sample_input):
    """Verify Grad-CAM generation, heatmap overlay, and output dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "gradcam_test.png")
        blended, path = generate_gradcam(
            model=test_model,
            input_spatial=sample_input,
            output_path=out_path,
        )

        assert blended.shape == (224, 224, 3)
        assert blended.dtype == np.uint8
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


def test_cross_attention_maps(test_model, sample_input):
    """Verify bidirectional cross-attention maps extraction and upsampling to 224x224."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.no_grad():
            _, exp = test_model(sample_input, return_explanations=True)

        paths = generate_cross_attention_maps(
            attn_sp_to_pe=exp["attn_sp_to_pe"],
            attn_pe_to_sp=exp["attn_pe_to_sp"],
            output_dir=tmpdir,
        )

        assert "attn_spatial_queries_spectral" in paths
        assert "attn_spectral_queries_spatial" in paths
        assert os.path.exists(paths["attn_spatial_queries_spectral"])
        assert os.path.exists(paths["attn_spectral_queries_spatial"])


def test_frequency_band_attribution(test_model, sample_input):
    """Verify 8-band concentric frequency attribution and bar chart figure generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "band_attr_test.png")
        attributions, path = generate_frequency_band_attribution(
            model=test_model,
            input_spatial=sample_input,
            num_bands=8,
            output_path=out_path,
        )

        assert len(attributions) == 8
        assert all(isinstance(v, float) for v in attributions)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


def test_spectrum_plot(sample_input):
    """Verify 2D log-magnitude FFT spectrum plot with D0 cutoff circle overlay."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "spectrum_test.png")
        path = generate_spectrum_plot(
            input_tensor=sample_input,
            d0=25.0,
            output_path=out_path,
        )

        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


def test_generate_all_explanations_hashes(test_model, sample_input):
    """Verify comprehensive explanation generator and individual SHA-256 evidence chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_all_explanations(
            model=test_model,
            input_spatial=sample_input,
            output_dir=tmpdir,
        )

        expected_keys = [
            "gradcam",
            "attn_spatial",
            "attn_spectral",
            "band_attribution",
            "spectrum_plot",
        ]

        for k in expected_keys:
            assert k in result, f"Missing explanation key: {k}"
            info = result[k]
            assert "file_path" in info
            assert "sha256" in info
            assert os.path.exists(info["file_path"])
            assert len(info["sha256"]) == 64  # Valid SHA-256 hex string
            # Recompute to verify checksum
            assert compute_file_sha256(info["file_path"]) == info["sha256"]
