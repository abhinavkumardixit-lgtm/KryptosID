"""Comprehensive Forensic Evaluation and ISO/IEC 30107-3 Metrics Reporting.

Implements Section 6.4 of DESIGN.md:
- AUC, EER, APCER, BPCER, ECE after temperature scaling.
- Per-generator breakdowns and confusion matrices.
- Canonical JSON report generation: reports/eval_<date>.json.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import os
import time
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.calibrate import compute_ece
from training.train import compute_eer


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
    
    Uses Mann-Whitney U rank statistic with fractional average rank handling for ties.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute ranks with average ranks for tied scores
    ranks = rankdata(scores)
    rank_sum_pos = np.sum(ranks[labels == 1])

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = float(u_stat / (n_pos * n_neg))
    return float(np.clip(auc, 0.0, 1.0))



def compute_iso_metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate APCER, BPCER, and Confusion Matrix at operating threshold."""
    labels = np.asarray(labels, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    # Predictions: 1 if score >= threshold, else 0
    preds = (scores >= threshold).astype(int)

    # Attack presentations (manipulated, label=1)
    attacks = (labels == 1)
    n_attacks = int(np.sum(attacks))
    # APCER: Attacks falsely accepted as genuine (preds == 0)
    apcer_errors = int(np.sum((preds == 0) & attacks))
    apcer = float(apcer_errors / max(1, n_attacks))

    # Bona fide presentations (genuine, label=0)
    bona_fide = (labels == 0)
    n_bona_fide = int(np.sum(bona_fide))
    # BPCER: Bona fide falsely rejected as manipulated (preds == 1)
    bpcer_errors = int(np.sum((preds == 1) & bona_fide))
    bpcer = float(bpcer_errors / max(1, n_bona_fide))

    # Confusion Matrix
    # TP: Manipulated predicted as Manipulated
    tp = int(np.sum((preds == 1) & (labels == 1)))
    # TN: Genuine predicted as Genuine
    tn = int(np.sum((preds == 0) & (labels == 0)))
    # FP: Genuine predicted as Manipulated
    fp = int(np.sum((preds == 1) & (labels == 0)))
    # FN: Manipulated predicted as Genuine
    fn = int(np.sum((preds == 0) & (labels == 1)))

    accuracy = float((tp + tn) / max(1, len(labels)))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "total_samples": len(labels),
        },
    }


def evaluate_dataset_with_breakdown(
    model: nn.Module,
    dataloader: DataLoader,
    threshold: float,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete forensic evaluation suite and generate canonical report JSON."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    all_calibrated_probs = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        for spatial, spectral, labels, sample_ids in dataloader:
            spatial = spatial.to(device)
            spectral = spectral.to(device)

            logits = model((spatial, spectral))
            logits_np = logits.cpu().numpy().ravel()

            # Calibrate using temperature
            calibrated_prob = torch.sigmoid(logits / temperature).cpu().numpy().ravel()

            all_logits.extend(logits_np)
            all_calibrated_probs.extend(calibrated_prob)
            all_labels.extend(labels.numpy().ravel())
            all_sample_ids.extend(sample_ids)

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_calibrated_probs)

    # 1. AUC and EER
    auc = compute_roc_auc(labels_arr, probs_arr)
    eer, eer_thresh = compute_eer(labels_arr, probs_arr)

    # 2. ISO/IEC 30107-3 metrics at operating threshold
    iso_metrics = compute_iso_metrics_at_threshold(labels_arr, probs_arr, threshold=threshold)

    # 3. ECE (15 bins)
    ece_val, ece_details = compute_ece(probs_arr, labels_arr, num_bins=15)

    report_payload = {
        "metadata": {
            "model_version": "1.0.0",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operating_threshold": float(threshold),
            "temperature_used": float(temperature),
            "total_evaluated_samples": len(labels_arr),
        },
        "summary_metrics": {
            "auc": float(round(auc, 4)),
            "eer": float(round(eer, 4)),
            "eer_threshold": float(round(eer_thresh, 4)),
            "operating_apcer": float(round(iso_metrics["apcer"], 4)),
            "operating_bpcer": float(round(iso_metrics["bpcer"], 4)),
            "operating_accuracy": float(round(iso_metrics["accuracy"], 4)),
            "ece_15_bins": float(round(ece_val, 4)),
        },
        "confusion_matrix": iso_metrics["confusion_matrix"],
        "calibration": ece_details,
    }

    if output_report_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, sort_keys=True)

    return report_payload
