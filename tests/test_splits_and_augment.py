"""Tests for Phase 2: Data Splitting, Manifest Sealing, Leakage Prevention, and Augmentations."""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from training.augment import (
    ForensicAugmentationPipeline,
    jpeg_normalize_q85,
    random_jpeg_compression,
)
from training.datasets import ForensicDataset
from training.splits import (
    DatasetSample,
    DataLeakageError,
    ManifestVerificationError,
    create_generator_disjoint_split,
    create_identity_disjoint_split,
    verify_split_manifest,
)


def test_dataset_register_yaml():
    """Verify configs/datasets.yaml registers all datasets with license and compliance terms."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    assert os.path.exists(config_path), f"datasets.yaml not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "datasets" in data
    datasets = data["datasets"]

    expected_datasets = [
        "celeb_df_v2",
        "faceforensics_pp",
        "genimage",
        "doctamper",
        "casia_v2",
        "synthetic_vcip_identity_docs",
    ]

    for d in expected_datasets:
        assert d in datasets, f"Missing dataset in register: {d}"
        info = datasets[d]
        assert "license" in info
        assert "commercial_use" in info
        assert "access_terms" in info
        assert "split_strategy" in info

    # Verify DPDP compliance notes on synthetic V-CIP
    vcip = datasets["synthetic_vcip_identity_docs"]
    assert "dpdp_compliance" in vcip
    assert vcip["dpdp_compliance"]["pii_audit_passed"] is True


def test_identity_disjoint_split_and_leakage_protection():
    """Ensure face dataset splits never leak identities between splits."""
    samples = []
    # 10 identities, 10 samples each = 100 samples
    for id_num in range(10):
        id_str = f"actor_{id_num:03d}"
        for s_num in range(10):
            samples.append(
                DatasetSample(
                    sample_id=f"{id_str}_img_{s_num}",
                    file_path=f"/fake/path/{id_str}_{s_num}.jpg",
                    label=s_num % 2,
                    identity_id=id_str,
                    source_dataset="faceforensics_pp",
                )
            )

    manifest = create_identity_disjoint_split(
        samples=samples,
        dataset_name="faceforensics_pp",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_ids = {s["identity_id"] for s in manifest.train}
    val_ids = {s["identity_id"] for s in manifest.val}
    test_ids = {s["identity_id"] for s in manifest.test}

    # Strict identity-disjoint assertion
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share identities!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share identities!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share identities!"

    # Simulate identity leakage and verify exception is raised
    manifest.test.append(manifest.train[0])
    with pytest.raises(DataLeakageError):
        from training.splits import assert_zero_leakage
        assert_zero_leakage(manifest)


def test_generator_disjoint_split():
    """Ensure synthetic image splits isolate held-out generators strictly to the test set."""
    generators = ["sd_v14", "sd_v15", "adm", "wukong", "midjourney", "glide"]
    held_out = ["midjourney", "glide"]

    samples = []
    for g in generators:
        for idx in range(20):
            samples.append(
                DatasetSample(
                    sample_id=f"{g}_img_{idx}",
                    file_path=f"/fake/{g}_{idx}.png",
                    label=1,
                    generator_name=g,
                    source_dataset="genimage",
                )
            )

    manifest = create_generator_disjoint_split(
        samples=samples,
        dataset_name="genimage",
        held_out_generators=held_out,
        train_val_ratio=0.8,
        seed=123,
    )

    train_gens = {s["generator_name"] for s in manifest.train}
    val_gens = {s["generator_name"] for s in manifest.val}
    test_gens = {s["generator_name"] for s in manifest.test}

    assert "midjourney" not in train_gens and "midjourney" not in val_gens
    assert "glide" not in train_gens and "glide" not in val_gens
    assert "midjourney" in test_gens and "glide" in test_gens


def test_manifest_sealing_and_tamper_detection():
    """Verify SHA-256 seal computation and tamper rejection."""
    samples = [
        DatasetSample(
            sample_id=f"sample_{i}",
            file_path=f"/path/{i}.jpg",
            label=i % 2,
            identity_id=f"id_{i // 2}",
        )
        for i in range(10)
    ]
    manifest = create_identity_disjoint_split(samples, dataset_name="test_ds")

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.json")
        manifest.save(manifest_path)

        # Verification succeeds on intact manifest
        verified = verify_split_manifest(manifest_path)
        assert verified.manifest_sha256 == manifest.manifest_sha256

        # Tamper with the manifest file by altering one character
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = f.read()
        tampered_content = content.replace("sample_0", "sample_TAMPERED")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(tampered_content)

        # Tampered manifest MUST be rejected with ManifestVerificationError
        with pytest.raises(ManifestVerificationError):
            verify_split_manifest(manifest_path)


def test_forensic_augmentation_pipeline_dual_output():
    """Verify ForensicAugmentationPipeline output dimensions, ranges, and normalization."""
    pipeline_train = ForensicAugmentationPipeline(is_training=True)
    pipeline_eval = ForensicAugmentationPipeline(is_training=False)

    img_np = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    # 1. Training mode
    spatial_train, spectral_train = pipeline_train(img_np)
    assert spatial_train.shape == (3, 224, 224)
    assert spectral_train.shape == (1, 224, 224)
    # Spectral grayscale should be un-normalized in [0, 1]
    assert spectral_train.min() >= 0.0 and spectral_train.max() <= 1.0

    # 2. Evaluation mode
    spatial_eval, spectral_eval = pipeline_eval(img_np)
    assert spatial_eval.shape == (3, 224, 224)
    assert spectral_eval.shape == (1, 224, 224)
    assert spectral_eval.min() >= 0.0 and spectral_eval.max() <= 1.0


def test_forensic_dataset_dataloader():
    """Verify PyTorch DataLoader integration with sealed manifest."""
    samples = [
        DatasetSample(
            sample_id=f"s_{i}",
            file_path=f"/fake/{i}.png",
            label=i % 2,
            identity_id=f"person_{i // 3}",
        )
        for i in range(12)
    ]
    manifest = create_identity_disjoint_split(samples, dataset_name="batch_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.json")
        manifest.save(manifest_path)

        dataset = ForensicDataset(manifest_path=manifest_path, split="train", generate_mock_on_missing=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        spatial_batch, spectral_batch, labels, sample_ids = batch

        assert spatial_batch.shape == (4, 3, 224, 224)
        assert spectral_batch.shape == (4, 1, 224, 224)
        assert labels.shape == (4, 1)
        assert len(sample_ids) == 4
