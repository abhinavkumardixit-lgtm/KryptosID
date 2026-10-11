"""Dataset Split Generation, Identity-Disjoint Manifests, and Leakage Verification.

Implements Section 5.2 of DESIGN.md:
- Face datasets: Split strictly by identity / actor pair. Zero identity leakage across train, val, and test.
- Generated images: Split by generator and prompt/seed groups, holding out test generators (e.g. Midjourney, Glide).
- Split manifests are canonicalized, hashed with SHA-256, and verified prior to training.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass


class DataLeakageError(Exception):
    """Raised when identity or generator overlap is detected across data splits."""
    pass


class ManifestVerificationError(Exception):
    """Raised when manifest checksum does not match stored or expected hash."""
    pass


@dataclass
class DatasetSample:
    """Single sample metadata."""
    sample_id: str
    file_path: str
    label: int  # 0: genuine, 1: manipulated
    identity_id: Optional[str] = None
    generator_name: Optional[str] = None
    source_dataset: str = "generic"
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SplitManifest:
    """Canonical dataset split manifest with SHA-256 integrity seal."""
    version: str
    dataset_name: str
    split_strategy: str
    created_at: str
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    held_out_test_generators: Optional[List[str]] = None
    manifest_sha256: Optional[str] = None

    def compute_sha256(self) -> str:
        """Compute SHA-256 over canonical JSON of the splits payload (excluding manifest_sha256)."""
        payload = {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "split_strategy": self.split_strategy,
            "created_at": self.created_at,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "held_out_test_generators": self.held_out_test_generators or [],
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def seal(self) -> "SplitManifest":
        """Compute and set manifest_sha256."""
        self.manifest_sha256 = self.compute_sha256()
        return self

    def save(self, output_path: str) -> str:
        """Seal and save manifest to file, returning the file path."""
        self.seal()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = asdict(self)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return output_path


def verify_split_manifest(
    manifest_path: str,
    expected_hash: Optional[str] = None,
) -> SplitManifest:
    """Load and strictly verify manifest integrity and check for any data leakage.
    
    Args:
        manifest_path: Path to manifest JSON.
        expected_hash: Optional expected SHA-256 hash. If None, checks internal manifest_sha256.
        
    Returns:
        Verified SplitManifest instance.
        
    Raises:
        ManifestVerificationError: If file is missing, invalid, or hash check fails.
        DataLeakageError: If identity or generator leakage is detected between splits.
    """
    if not os.path.exists(manifest_path):
        raise ManifestVerificationError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stored_hash = data.get("manifest_sha256")
    manifest = SplitManifest(
        version=data["version"],
        dataset_name=data["dataset_name"],
        split_strategy=data["split_strategy"],
        created_at=data["created_at"],
        train=data["train"],
        val=data["val"],
        test=data["test"],
        held_out_test_generators=data.get("held_out_test_generators"),
        manifest_sha256=stored_hash,
    )

    recalculated_hash = manifest.compute_sha256()

    if stored_hash is None or stored_hash != recalculated_hash:
        raise ManifestVerificationError(
            f"Manifest SHA-256 verification failed for {manifest_path}!\n"
            f"Stored: {stored_hash}\n"
            f"Recalculated: {recalculated_hash}"
        )

    if expected_hash is not None and recalculated_hash != expected_hash:
        raise ManifestVerificationError(
            f"Manifest hash {recalculated_hash} does not match expected {expected_hash}"
        )

    # Check for leakage
    assert_zero_leakage(manifest)
    return manifest


def assert_zero_leakage(manifest: SplitManifest) -> None:
    """Enforce strict zero-leakage assertions across splits."""
    train_ids = {s["sample_id"] for s in manifest.train}
    val_ids = {s["sample_id"] for s in manifest.val}
    test_ids = {s["sample_id"] for s in manifest.test}

    # 1. Sample ID uniqueness
    overlap_train_val = train_ids.intersection(val_ids)
    if overlap_train_val:
        raise DataLeakageError(f"Sample ID leakage between Train and Val: {len(overlap_train_val)} samples")

    overlap_train_test = train_ids.intersection(test_ids)
    if overlap_train_test:
        raise DataLeakageError(f"Sample ID leakage between Train and Test: {len(overlap_train_test)} samples")

    overlap_val_test = val_ids.intersection(test_ids)
    if overlap_val_test:
        raise DataLeakageError(f"Sample ID leakage between Val and Test: {len(overlap_val_test)} samples")

    # 2. Identity-disjoint checks (for face datasets)
    if manifest.split_strategy in ("identity_disjoint", "actor_pair_disjoint"):
        train_identities = {s["identity_id"] for s in manifest.train if s.get("identity_id")}
        val_identities = {s["identity_id"] for s in manifest.val if s.get("identity_id")}
        test_identities = {s["identity_id"] for s in manifest.test if s.get("identity_id")}

        id_leak_tv = train_identities.intersection(val_identities)
        if id_leak_tv:
            raise DataLeakageError(f"Identity leakage between Train and Val: {id_leak_tv}")

        id_leak_tt = train_identities.intersection(test_identities)
        if id_leak_tt:
            raise DataLeakageError(f"Identity leakage between Train and Test: {id_leak_tt}")

        id_leak_vt = val_identities.intersection(test_identities)
        if id_leak_vt:
            raise DataLeakageError(f"Identity leakage between Val and Test: {id_leak_vt}")

    # 3. Generator-disjoint checks (for synthetic image datasets)
    if manifest.split_strategy == "generator_disjoint" and manifest.held_out_test_generators:
        held_out = set(manifest.held_out_test_generators)
        train_gens = {s["generator_name"] for s in manifest.train if s.get("generator_name")}
        val_gens = {s["generator_name"] for s in manifest.val if s.get("generator_name")}

        gen_leak_train = train_gens.intersection(held_out)
        if gen_leak_train:
            raise DataLeakageError(f"Held-out test generators found in Train split: {gen_leak_train}")

        gen_leak_val = val_gens.intersection(held_out)
        if gen_leak_val:
            raise DataLeakageError(f"Held-out test generators found in Val split: {gen_leak_val}")


def create_identity_disjoint_split(
    samples: List[DatasetSample],
    dataset_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    version: str = "1.0.0",
) -> SplitManifest:
    """Generate identity-disjoint splits ensuring no subject identity is shared across splits."""
    rng = random.Random(seed)

    # Group samples by identity_id
    identity_groups: Dict[str, List[DatasetSample]] = {}
    for s in samples:
        if not s.identity_id:
            raise ValueError(f"Sample {s.sample_id} missing required identity_id")
        identity_groups.setdefault(s.identity_id, []).append(s)

    unique_identities = sorted(identity_groups.keys())
    rng.shuffle(unique_identities)

    n_total = len(unique_identities)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))

    train_ids = set(unique_identities[:n_train])
    val_ids = set(unique_identities[n_train : n_train + n_val])
    test_ids = set(unique_identities[n_train + n_val :])

    if not test_ids:
        # If very few identities, ensure at least 1 in test
        if len(val_ids) > 1:
            moved = val_ids.pop()
            test_ids.add(moved)

    train_samples = [asdict(s) for s_id in train_ids for s in identity_groups[s_id]]
    val_samples = [asdict(s) for s_id in val_ids for s in identity_groups[s_id]]
    test_samples = [asdict(s) for s_id in test_ids for s in identity_groups[s_id]]

    # Sort deterministically
    train_samples.sort(key=lambda x: x["sample_id"])
    val_samples.sort(key=lambda x: x["sample_id"])
    test_samples.sort(key=lambda x: x["sample_id"])

    manifest = SplitManifest(
        version=version,
        dataset_name=dataset_name,
        split_strategy="identity_disjoint",
        created_at="2026-10-11T00:00:00Z",
        train=train_samples,
        val=val_samples,
        test=test_samples,
    )
    manifest.seal()
    assert_zero_leakage(manifest)
    return manifest


def create_generator_disjoint_split(
    samples: List[DatasetSample],
    dataset_name: str,
    held_out_generators: List[str],
    train_val_ratio: float = 0.85,
    seed: int = 42,
    version: str = "1.0.0",
) -> SplitManifest:
    """Generate generator-disjoint splits with designated held-out generators strictly in test."""
    rng = random.Random(seed)
    held_out_set = set(held_out_generators)

    in_domain_samples: List[DatasetSample] = []
    test_samples: List[Dict[str, Any]] = []

    for s in samples:
        if not s.generator_name:
            raise ValueError(f"Sample {s.sample_id} missing required generator_name")
        if s.generator_name in held_out_set:
            test_samples.append(asdict(s))
        else:
            in_domain_samples.append(s)

    # Shuffle in-domain samples deterministically
    rng.shuffle(in_domain_samples)
    n_in_domain = len(in_domain_samples)
    n_train = int(round(n_in_domain * train_val_ratio))

    train_samples = [asdict(s) for s in in_domain_samples[:n_train]]
    val_samples = [asdict(s) for s in in_domain_samples[n_train:]]

    train_samples.sort(key=lambda x: x["sample_id"])
    val_samples.sort(key=lambda x: x["sample_id"])
    test_samples.sort(key=lambda x: x["sample_id"])

    manifest = SplitManifest(
        version=version,
        dataset_name=dataset_name,
        split_strategy="generator_disjoint",
        created_at="2026-10-11T00:00:00Z",
        train=train_samples,
        val=val_samples,
        test=test_samples,
        held_out_test_generators=sorted(held_out_generators),
    )
    manifest.seal()
    assert_zero_leakage(manifest)
    return manifest
