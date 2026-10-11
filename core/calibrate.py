"""Temperature Scaling, Expected Calibration Error (ECE), and ISO/IEC 30107-3 Threshold Selection.

Implements Section 6.3 of DESIGN.md:
- Temperature scaling: Fits single scalar T on validation set by minimizing NLL (model weights unmodified).
- Expected Calibration Error (ECE) across 15 bins before and after temperature scaling.
- Operating threshold selection: Targets APCER <= 1.0% (recording resulting BPCER).
- Thresholds and temperature exported to configs/serve.yaml.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import yaml


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) with equal-width confidence bins.
    
    Args:
        probs: 1D array of predicted probabilities in [0, 1] (for class 1 / manipulated).
        labels: 1D array of ground truth labels (0 genuine, 1 manipulated).
        num_bins: Number of bins (15 per DESIGN.md Section 6.3).
        
    Returns:
        Tuple of (ece_value, bin_details_dict).
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    n_total = len(probs)
    ece = 0.0

    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples belonging to current confidence bin
        if i == num_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(labels[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n_total) * bin_error

            bin_details.append({
                "bin": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "error": bin_error,
            })
        else:
            bin_details.append({
                "bin": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": float((bin_lower + bin_upper) / 2.0),
                "error": 0.0,
            })

    return float(ece), {"num_bins": num_bins, "bins": bin_details}


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    initial_temp: float = 1.0,
    max_iter: int = 100,
) -> Tuple[float, float, float]:
    """Fit optimal temperature scalar T > 0 by minimizing Negative Log-Likelihood (NLL).
    
    Optimizes log(T) via PyTorch L-BFGS to guarantee strictly positive temperature.
    
    Args:
        logits: Uncalibrated model logits (1D array).
        labels: Binary labels (0 for genuine, 1 for manipulated).
        initial_temp: Starting temperature guess.
        max_iter: Max iterations for L-BFGS.
        
    Returns:
        Tuple of (optimal_temperature, nll_before, nll_after).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32).ravel()
    labels_t = torch.tensor(labels, dtype=torch.float32).ravel()

    criterion = nn.BCEWithLogitsLoss()
    nll_before = criterion(logits_t, labels_t).item()

    # Parameterize via log_temp: T = exp(log_temp) > 0
    init_val = float(np.log(max(1e-3, initial_temp)))
    log_temp = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def _eval():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        loss = criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval)

    optimal_t = float(torch.exp(log_temp).item())
    nll_after = criterion(logits_t / optimal_t, labels_t).item()

    return optimal_t, float(nll_before), float(nll_after)



def select_operating_threshold(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    target_apcer: float = 0.01,
) -> Tuple[float, float, float]:
    """Select decision threshold achieving target APCER (e.g. <= 1%) on validation set.
    
    Definitions under ISO/IEC 30107-3:
    - Attack presentations (manipulated, label=1):
      APCER is False Acceptance Rate of attacks (manipulated classified as genuine: prob < threshold).
      APCER(theta) = sum(attack_probs < theta) / N_attacks
    - Bona fide presentations (genuine, label=0):
      BPCER is False Rejection Rate of genuine (genuine classified as manipulated: prob >= theta).
      BPCER(theta) = sum(bona_fide_probs >= theta) / N_bona_fide
      
    Args:
        calibrated_probs: Calibrated probabilities in [0, 1].
        labels: Ground truth labels (0 genuine, 1 manipulated).
        target_apcer: Target maximum error rate for attack presentations (default 0.01 = 1%).
        
    Returns:
        Tuple of (selected_threshold, resulting_apcer, resulting_bpcer).
    """
    calibrated_probs = np.asarray(calibrated_probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    attack_probs = calibrated_probs[labels == 1]
    bona_fide_probs = calibrated_probs[labels == 0]

    if len(attack_probs) == 0 or len(bona_fide_probs) == 0:
        return 0.5, 0.0, 0.0

    n_attacks = len(attack_probs)
    n_bona_fide = len(bona_fide_probs)

    # Search across potential threshold candidates
    candidate_thresholds = np.sort(np.unique(calibrated_probs))
    if len(candidate_thresholds) > 1000:
        candidate_thresholds = np.quantile(calibrated_probs, np.linspace(0, 1, 1000))

    valid_thresholds = []

    for t in candidate_thresholds:
        apcer = np.sum(attack_probs < t) / n_attacks
        bpcer = np.sum(bona_fide_probs >= t) / n_bona_fide
        if apcer <= target_apcer:
            valid_thresholds.append((float(t), float(apcer), float(bpcer)))

    if valid_thresholds:
        # Among thresholds meeting APCER constraint, pick the one with lowest BPCER
        valid_thresholds.sort(key=lambda x: (x[2], -x[0]))
        selected_thresh, selected_apcer, selected_bpcer = valid_thresholds[0]
    else:
        # If target cannot be strictly met, choose lowest APCER achievable
        all_evals = []
        for t in candidate_thresholds:
            apcer = np.sum(attack_probs < t) / n_attacks
            bpcer = np.sum(bona_fide_probs >= t) / n_bona_fide
            all_evals.append((float(t), float(apcer), float(bpcer)))
        all_evals.sort(key=lambda x: x[1])
        selected_thresh, selected_apcer, selected_bpcer = all_evals[0]

    return selected_thresh, selected_apcer, selected_bpcer


def export_serve_config(
    output_path: str,
    threshold: float,
    temperature: float,
    apcer: float,
    bpcer: float,
    ece: float,
    model_path: str = "checkpoints/kryptosid_best_stage2.pt",
) -> str:
    """Export calibrated threshold and serving parameters to configs/serve.yaml."""
    serve_data = {
        "version": "1.0.0",
        "service": {
            "name": "kryptosid-serving",
            "host": "0.0.0.0",
            "port": 8000,
        },
        "model": {
            "path": model_path,
            "temperature": float(round(temperature, 4)),
            "decision_threshold": float(round(threshold, 4)),
            "inconclusive_margin": 0.05,  # Score within [thresh - margin, thresh + margin] flags inconclusive
        },
        "operating_point_iso_30107": {
            "target_apcer": "1.0%",
            "validation_apcer": float(round(apcer, 4)),
            "validation_bpcer": float(round(bpcer, 4)),
            "validation_ece": float(round(ece, 4)),
            "calibrated_on": "validation_split",
        },
        "limits": {
            "max_file_size_bytes": 52428800,  # 50 MB
            "max_video_duration_seconds": 60,
            "max_frames_sampled": 120,
            "request_timeout_seconds": 30,
        },
    }

    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(serve_data, f, sort_keys=False, default_flow_style=False)

    return output_path
