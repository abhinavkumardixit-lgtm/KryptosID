"""Tests for Phase 3: Training Pipeline, Baselines, and Checkpoint Sealing."""

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.baselines import (
    NoButterworthNet,
    SpatialOnlyNet,
    SpectralOnlyNet,
    build_ablation_model,
)
from core.model import KryptosIDNet
from training.train import Trainer, compute_eer, set_seed


def test_compute_eer_properties():
    """Verify EER calculation across boundary conditions."""
    # 1. Perfect discrimination: EER must be 0.0
    labels_perf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores_perf = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])
    eer, thresh = compute_eer(labels_perf, scores_perf)
    assert eer == 0.0, f"Expected EER 0.0, got {eer}"
    assert 0.2 <= thresh <= 0.8

    # 2. Complete confusion (identical distributions)
    labels_rand = np.array([0, 1, 0, 1, 0, 1])
    scores_rand = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    eer_rand, _ = compute_eer(labels_rand, scores_rand)
    assert 0.4 <= eer_rand <= 0.6


def test_ablation_and_baseline_models():
    """Verify forward passes and shapes for all baseline and ablation architectures."""
    b = 2
    x_rgb = torch.randn(b, 3, 224, 224)
    x_gray = torch.randn(b, 1, 224, 224)

    # 1. Spatial-only baseline
    m_spatial = SpatialOnlyNet(pretrained=False, use_mock=True)
    out_spatial = m_spatial(x_rgb)
    assert out_spatial.shape == (b, 1)

    # 2. Spectral-only baseline
    m_spectral = SpectralOnlyNet(pretrained=False, use_mock=True)
    out_spectral = m_spectral(x_gray)
    assert out_spectral.shape == (b, 1)

    # 3. No Butterworth filter ablation
    m_no_bw = NoButterworthNet(pretrained_spatial=False, pretrained_spectral=False, use_mock=True)
    out_no_bw = m_no_bw(x_rgb)
    assert out_no_bw.shape == (b, 1)

    # 4. Factory function
    for m_type in ["kryptos_id_net", "spatial_only", "spectral_only", "no_butterworth", "random_init_resnet"]:
        model = build_ablation_model(m_type, use_mock=True)
        assert isinstance(model, nn.Module)


def test_trainer_two_stage_warmup_and_checkpoint():
    """Test two-stage training loop, parameter freezing transition, and checkpoint sealing."""
    set_seed(42)
    b, h, w = 4, 224, 224

    # Synthetic dataset
    spatial_data = torch.randn(16, 3, h, w)
    spectral_data = torch.randn(16, 1, h, w)
    labels = torch.randint(0, 2, (16, 1)).float()

    dataset = TensorDataset(spatial_data, spectral_data, labels, torch.arange(16))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = KryptosIDNet(use_mock_backbones=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "training": {
                "gradient_clip_norm": 1.0,
                "weight_decay": 0.01,
                "early_stopping_patience": 2,
                "stage1_warmup": {
                    "epochs": 1,
                    "lr_head_and_fusion": 1e-3,
                },
                "stage2_finetune": {
                    "epochs": 1,
                    "lr_backbone": 1e-5,
                    "lr_head": 1e-4,
                },
            },
            "checkpoints": {
                "save_dir": tmpdir,
            },
        }

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device=torch.device("cpu"),
            manifest_sha256="test_sha256_seal",
        )

        result = trainer.run()

        assert "best_eer" in result
        assert os.path.exists(result["best_checkpoint"])

        # Inspect sealed checkpoint
        ckpt = torch.load(result["best_checkpoint"], map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "config_sha256" in ckpt
        assert "manifest_sha256" in ckpt
        assert ckpt["manifest_sha256"] == "test_sha256_seal"
        assert "calibrated_threshold" in ckpt
