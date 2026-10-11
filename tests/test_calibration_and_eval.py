"""Tests for Phase 4: Calibration, Expected Calibration Error, and ISO/IEC 30107-3 Evaluation."""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from core.calibrate import (
    compute_ece,
    export_serve_config,
    fit_temperature_scaling,
    select_operating_threshold,
)
from core.model import KryptosIDNet
from training.evaluate import (
    compute_iso_metrics_at_threshold,
    compute_roc_auc,
    evaluate_dataset_with_breakdown,
)


def test_ece_15_bins_properties():
    """Verify ECE calculation on calibrated and uncalibrated distributions."""
    # 1. Perfectly calibrated probabilities:
    # Bin 0.2 has 20% positive, Bin 0.8 has 80% positive
    probs_cal = np.array([0.2] * 10 + [0.8] * 10)
    labels_cal = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1] * 8 + [0] * 2)
    ece_val, details = compute_ece(probs_cal, labels_cal, num_bins=15)
    assert details["num_bins"] == 15
    assert ece_val < 0.05, f"ECE for calibrated data should be near 0, got {ece_val}"

    # 2. Severely miscalibrated / overconfident wrong predictions
    probs_wrong = np.ones(20) * 0.99
    labels_wrong = np.zeros(20)
    ece_wrong, _ = compute_ece(probs_wrong, labels_wrong, num_bins=15)
    assert ece_wrong > 0.95, f"ECE for completely wrong predictions must be near 1.0, got {ece_wrong}"


def test_temperature_scaling_optimization():
    """Verify that temperature scaling minimizes NLL and softens overconfident logits."""
    # 100 positive predictions (logit = +3.5) where 75 are label 1, 25 are label 0
    # 100 negative predictions (logit = -3.5) where 75 are label 0, 25 are label 1
    # Real logit should be ~1.1; model is overconfident at 3.5 -> optimal T should be ~3.0 > 1.0
    labels = np.array([1] * 75 + [0] * 25 + [0] * 75 + [1] * 25)
    overconfident_logits = np.array([3.5] * 100 + [-3.5] * 100)

    optimal_t, nll_before, nll_after = fit_temperature_scaling(
        logits=overconfident_logits,
        labels=labels,
    )

    assert optimal_t > 1.0, f"Temperature should be > 1 to soften overconfidence, got {optimal_t}"
    assert nll_after < nll_before, f"NLL after ({nll_after}) must be strictly < before ({nll_before})"



def test_operating_threshold_selection_iso_30107():
    """Verify threshold selection meeting APCER <= 1.0% constraint."""
    # 100 genuine (scores 0.05 to 0.40) and 100 manipulated (scores 0.35 to 0.95)
    bona_fide_probs = np.linspace(0.05, 0.40, 100)
    attack_probs = np.linspace(0.35, 0.95, 100)

    all_probs = np.concatenate([bona_fide_probs, attack_probs])
    all_labels = np.concatenate([np.zeros(100), np.ones(100)])

    target_apcer = 0.01  # 1%
    selected_thresh, actual_apcer, actual_bpcer = select_operating_threshold(
        calibrated_probs=all_probs,
        labels=all_labels,
        target_apcer=target_apcer,
    )

    assert actual_apcer <= target_apcer, f"Selected threshold {selected_thresh} violated APCER {actual_apcer} > {target_apcer}"
    assert 0.0 < selected_thresh < 1.0


def test_roc_auc_metrics():
    """Verify ROC-AUC exact calculation properties."""
    # Perfect classifier
    labels_perf = np.array([0, 0, 0, 1, 1, 1])
    scores_perf = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_roc_auc(labels_perf, scores_perf) == 1.0

    # Inverted classifier
    scores_inv = 1.0 - scores_perf
    assert compute_roc_auc(labels_perf, scores_inv) == 0.0

    # Random baseline
    scores_rand = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert compute_roc_auc(labels_perf, scores_rand) == 0.5


def test_export_serve_config_and_evaluate_dataset():
    """Verify serve.yaml export and full evaluation reporting pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        serve_path = os.path.join(tmpdir, "serve.yaml")
        report_path = os.path.join(tmpdir, "eval_report.json")

        # 1. Export serve.yaml
        export_serve_config(
            output_path=serve_path,
            threshold=0.4521,
            temperature=1.234,
            apcer=0.008,
            bpcer=0.024,
            ece=0.015,
        )
        assert os.path.exists(serve_path)

        with open(serve_path, "r", encoding="utf-8") as f:
            serve_cfg = yaml.safe_load(f)

        assert serve_cfg["model"]["decision_threshold"] == 0.4521
        assert serve_cfg["model"]["temperature"] == 1.234
        assert serve_cfg["operating_point_iso_30107"]["validation_apcer"] == 0.008

        # 2. Evaluation with breakdown
        model = KryptosIDNet(use_mock_backbones=True)
        spatial = torch.randn(8, 3, 224, 224)
        spectral = torch.randn(8, 1, 224, 224)
        labels = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]]).float()
        sample_ids = [f"sample_{i}" for i in range(8)]

        dataset = TensorDataset(spatial, spectral, labels, torch.arange(8))
        # Custom collate to handle sample_ids as list of strings
        def collate_fn(batch):
            sp = torch.stack([item[0] for item in batch])
            pe = torch.stack([item[1] for item in batch])
            lb = torch.stack([item[2] for item in batch])
            s_ids = [sample_ids[item[3].item()] for item in batch]
            return sp, pe, lb, s_ids

        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        report = evaluate_dataset_with_breakdown(
            model=model,
            dataloader=loader,
            threshold=0.5,
            temperature=1.0,
            device=torch.device("cpu"),
            output_report_path=report_path,
        )

        assert "summary_metrics" in report
        assert "auc" in report["summary_metrics"]
        assert "operating_apcer" in report["summary_metrics"]
        assert "confusion_matrix" in report
        assert os.path.exists(report_path)
