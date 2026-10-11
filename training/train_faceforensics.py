"""
KryptosID: Real FaceForensics++ Model Training & Calibration
Implements Sections 4, 5, & 6 of DESIGN.md:
- Trains KryptosIDNet with real ImageNet-pretrained backbones (ConvNeXt-S + ResNet-34).
- Uses identity-disjoint FaceForensics++ dataset split.
- 2-stage training (warmup freeze -> unfreeze fine-tune).
- Fits temperature scaling minimizing NLL on validation set.
- Calibrates operating threshold for APCER <= 1.0%.
- Exports sealed weights to models/kryptosid_best.pt and updates configs/serve.yaml.
"""

import os
import json
import torch
import yaml
from torch.utils.data import DataLoader

from core.model import KryptosIDNet
from core.calibrate import (
    fit_temperature_scaling,
    select_operating_threshold,
    compute_ece,
    export_serve_config,
)
from training.datasets import ForensicDataset
from training.splits import verify_split_manifest
from training.train import Trainer


def run_faceforensics_training(
    manifest_path: str = "data/faceforensics_manifest.json",
    output_checkpoint: str = "models/kryptosid_best.pt",
) -> None:
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # 1. Load and verify sealed manifest
    print("Verifying split manifest...")
    manifest = verify_split_manifest(manifest_path)
    print(f"Manifest verified: {manifest.manifest_sha256}")

    # 2. Build datasets and dataloaders
    train_dataset = ForensicDataset(manifest_path=manifest_path, split="train")
    val_dataset = ForensicDataset(manifest_path=manifest_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 3. Instantiate KryptosIDNet with real ImageNet backbones
    print("Instantiating KryptosIDNet with pretrained ConvNeXt-S and ResNet-34...")
    model = KryptosIDNet(
        pretrained_spatial=True,
        pretrained_spectral=True,
        use_mock_backbones=False,
    )
    model.to(device)

    # 4. Training configuration
    config = {
        "training": {
            "weight_decay": 0.05,
            "gradient_clip_norm": 1.0,
            "stage1_warmup": {
                "epochs": 2,
                "lr_head_and_fusion": 5e-4,
            },
            "stage2_finetune": {
                "epochs": 2,
                "lr_backbones": 5e-5,
                "lr_head_and_fusion": 1e-4,
                "min_lr": 1e-6,
            },
        },
        "checkpoints": {
            "save_dir": "models",
            "model_name": "kryptosid",
        },
    }

    # 5. Run Two-Stage Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        manifest_sha256=manifest.manifest_sha256,
    )

    print("\nExecuting 2-stage training loop...")
    results = trainer.run()
    best_eer = results["best_eer"]
    print(f"Training completed! Best Val EER: {best_eer:.4f}")

    # Save to canonical best weights path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "manifest_sha256": manifest.manifest_sha256,
            "best_eer": best_eer,
        },
        output_checkpoint,
    )
    print(f"Saved best model checkpoint to {output_checkpoint}")

    # 6. Temperature Scaling & Calibration (Section 6.3)
    print("\nFitting temperature scaling on validation set...")
    model.eval()
    val_logits_list = []
    val_labels_list = []

    with torch.no_grad():
        for spatial, spectral, labels, _ in val_loader:
            spatial = spatial.to(device)
            spectral = spectral.to(device)
            logits = model((spatial, spectral)).squeeze(-1)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(labels.squeeze(-1).cpu())

    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    val_logits_np = val_logits.numpy()
    val_labels_np = val_labels.numpy()

    # Uncalibrated ECE
    uncal_probs = torch.sigmoid(val_logits).numpy()
    ece_uncal, _ = compute_ece(uncal_probs, val_labels_np, num_bins=15)

    # Optimize Temperature T via L-BFGS
    opt_temp, nll_before, nll_after = fit_temperature_scaling(val_logits_np, val_labels_np)

    # Calibrated ECE
    cal_probs = torch.sigmoid(val_logits / opt_temp).numpy()
    ece_cal, _ = compute_ece(cal_probs, val_labels_np, num_bins=15)

    print(f"Temperature fitted: T = {opt_temp:.4f}")
    print(f"Uncalibrated ECE (15 bins): {ece_uncal:.4f}")
    print(f"Calibrated ECE   (15 bins): {ece_cal:.4f}")

    # Threshold selection for APCER <= 1.0%
    threshold, apcer, bpcer = select_operating_threshold(
        calibrated_probs=cal_probs,
        labels=val_labels_np,
        target_apcer=0.01,
    )
    print(f"Calibrated Decision Threshold: tau = {threshold:.4f} (APCER = {apcer:.4f}, BPCER = {bpcer:.4f})")

    # 7. Update serve.yaml
    export_serve_config(
        output_path="configs/serve.yaml",
        threshold=threshold,
        temperature=opt_temp,
        apcer=apcer,
        bpcer=bpcer,
        ece=ece_cal,
        model_path=output_checkpoint,
    )
    print("Updated configs/serve.yaml with calibrated operating threshold!")


if __name__ == "__main__":
    run_faceforensics_training()
