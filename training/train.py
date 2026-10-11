"""Training Pipeline with Two-Stage Schedule, AMP, and Validation EER Early Stopping.

Implements Section 6.1 of DESIGN.md:
- Two-stage schedule:
  1. Warm-up: backbones frozen, only fusion and head train (LR=1e-3, 3 epochs)
  2. Fine-tune: unfreeze last stages of both backbones (Backbone LR=1e-5, Head LR=1e-4),
     AdamW (weight_decay=0.05), cosine schedule with warm-up.
- Mixed precision (AMP), gradient clipping at 1.0.
- Fixed seeds across Python, NumPy, PyTorch, CUDA.
- Early stopping on validation EER (not loss).
- Sealed checkpoints with config hash, dataset manifest hash, and metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.model import KryptosIDNet


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the optimal threshold.
    
    Args:
        labels: 1D array of ground truth labels (0 for genuine, 1 for manipulated).
        scores: 1D array of predicted scores/probabilities (higher means more manipulated).
        
    Returns:
        Tuple of (eer_value, optimal_threshold).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    if len(labels) == 0:
        return 0.0, 0.5

    pos_scores = scores[labels == 1]  # Attack
    neg_scores = scores[labels == 0]  # Bona fide

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0, 0.5

    # Threshold grid from sorted unique scores
    thresholds = np.unique(scores)
    if len(thresholds) > 1000:
        thresholds = np.quantile(scores, np.linspace(0, 1, 1000))

    thresholds = np.sort(thresholds)

    # Compute False Positive Rate (BPCER) and False Negative Rate (APCER)
    # FPR: genuine classified as manipulated (score >= thresh)
    # FNR: manipulated classified as genuine (score < thresh)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    fpr_list = []
    fnr_list = []

    for t in thresholds:
        fpr = np.sum(neg_scores >= t) / n_neg
        fnr = np.sum(pos_scores < t) / n_pos
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    fpr_arr = np.array(fpr_list)
    fnr_arr = np.array(fnr_list)

    # Find threshold where FPR and FNR are closest
    diff = np.abs(fpr_arr - fnr_arr)
    min_idx = np.argmin(diff)

    eer = float((fpr_arr[min_idx] + fnr_arr[min_idx]) / 2.0)
    thresh = float(thresholds[min_idx])

    return eer, thresh


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Clusters bona-fide facial representations tightly in hyperspherical embedding space,
    while separating them from diverse generator manipulation distributions.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        mask_sum = mask.sum(1)
        valid_positives = mask_sum > 0
        if not valid_positives.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_positives] / mask_sum[valid_positives]
        loss = -mean_log_prob_pos.mean()
        return loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    clip_norm: float = 1.0,
) -> float:
    """Execute one training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = scaler is not None and device.type == "cuda"

    for spatial, spectral, labels, _ in dataloader:
        spatial = spatial.to(device)
        spectral = spectral.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model((spatial, spectral))
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on dataset and return (avg_loss, eer, threshold)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for spatial, spectral, labels, _ in dataloader:
            spatial = spatial.to(device)
            spectral = spectral.to(device)
            labels = labels.to(device)

            logits = model((spatial, spectral))
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            total_loss += loss.item()
            num_batches += 1

            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy().ravel())

    avg_loss = total_loss / max(1, num_batches)
    eer, thresh = compute_eer(np.array(all_labels), np.array(all_scores))
    return avg_loss, eer, thresh


class Trainer:
    """Full two-stage trainer managing warmup, fine-tuning, and sealed checkpoints."""

    def __init__(
        self,
        model: KryptosIDNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        manifest_sha256: str = "unspecified",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.manifest_sha256 = manifest_sha256

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None

        self.save_dir = config.get("checkpoints", {}).get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.best_eer = float("inf")
        self.best_checkpoint_path = ""

    def run(self) -> Dict[str, Any]:
        """Execute Stage 1 Warm-up followed by Stage 2 Fine-tuning."""
        t_cfg = self.config.get("training", {})
        clip_norm = t_cfg.get("gradient_clip_norm", 1.0)

        # -------------------------------------------------------------
        # STAGE 1: Warm-up (Backbones frozen, only fusion and head train)
        # -------------------------------------------------------------
        s1_cfg = t_cfg.get("stage1_warmup", {})
        s1_epochs = s1_cfg.get("epochs", 3)
        s1_lr = s1_cfg.get("lr_head_and_fusion", 1e-3)

        self.model.freeze_backbones()
        optimizer_s1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=s1_lr,
            weight_decay=t_cfg.get("weight_decay", 0.05),
        )

        history = {"stage1": [], "stage2": []}

        for epoch in range(1, s1_epochs + 1):
            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                optimizer=optimizer_s1,
                criterion=self.criterion,
                device=self.device,
                scaler=self.scaler,
                clip_norm=clip_norm,
            )
            val_loss, val_eer, thresh = evaluate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )
            history["stage1"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_eer": val_eer,
                "thresh": thresh,
            })

            if val_eer < self.best_eer:
                self.best_eer = val_eer
                self._save_checkpoint(epoch, "stage1", val_eer, thresh)

        # -------------------------------------------------------------
        # STAGE 2: Fine-tuning (Unfreeze last stages of both backbones)
        # -------------------------------------------------------------
        s2_cfg = t_cfg.get("stage2_finetune", {})
        s2_epochs = s2_cfg.get("epochs", 20)
        lr_bb = s2_cfg.get("lr_backbone", 1e-5)
        lr_head = s2_cfg.get("lr_head", 1e-4)

        self.model.unfreeze_backbones(last_stages_only=s2_cfg.get("unfreeze_last_stages_only", True))

        # Parameter groups for different learning rates
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "spatial_backbone" in name or "spectral_backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr_bb},
            {"params": head_params, "lr": lr_head},
        ]

        optimizer_s2 = torch.optim.AdamW(
            param_groups,
            weight_decay=t_cfg.get("weight_decay", 0.05),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=s2_epochs)

        patience = t_cfg.get("early_stopping_patience", 5)
        epochs_no_improve = 0

        for epoch in range(1, s2_epochs + 1):
            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                optimizer=optimizer_s2,
                criterion=self.criterion,
                device=self.device,
                scaler=self.scaler,
                clip_norm=clip_norm,
            )
            val_loss, val_eer, thresh = evaluate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )
            scheduler.step()

            history["stage2"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_eer": val_eer,
                "thresh": thresh,
            })

            # Early stopping check on validation EER (Section 6.1)
            if val_eer < self.best_eer:
                self.best_eer = val_eer
                self._save_checkpoint(epoch, "stage2", val_eer, thresh)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return {"best_eer": self.best_eer, "best_checkpoint": self.best_checkpoint_path, "history": history}

    def _save_checkpoint(self, epoch: int, stage: str, val_eer: float, threshold: float) -> str:
        """Save sealed checkpoint with weights, hashes, and metrics."""
        ckpt_path = os.path.join(self.save_dir, f"kryptosid_best_{stage}.pt")
        state = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
            "val_eer": val_eer,
            "calibrated_threshold": threshold,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        torch.save(state, ckpt_path)
        self.best_checkpoint_path = ckpt_path
        return ckpt_path
