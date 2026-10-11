"""Cryptographic Evidence Hashing and Canonical Representation.

Implements Section 9.1 of DESIGN.md:
- Raw input byte hashing (performed BEFORE decoding)
- Model weights, config, and artifact hashing
- Canonical JSON serialization (sorted keys, compact separators) and SHA-256 digest
"""

from typing import Any, Dict, Union
import hashlib
import json


def hash_raw_bytes(data: bytes) -> str:
    """Compute SHA-256 of raw uploaded media bytes before any decoding."""
    return hashlib.sha256(data).hexdigest()


def hash_file_path(file_path: str) -> str:
    """Compute SHA-256 of an on-disk file (model weights, configs)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dumps(data: Dict[str, Any]) -> str:
    """Serialize dictionary to deterministic, canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash over canonical JSON representation."""
    canonical_str = canonical_json_dumps(data)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def build_evidence_chain(
    input_bytes: bytes,
    decision: str,
    score: float,
    rule_fired: str,
    threshold: float,
    temperature: float,
    started_at: str,
    finished_at: str,
    model_name: str = "KryptosIDNet",
    model_version: str = "1.0.0",
    weights_sha256: str = "unspecified",
    config_sha256: str = "unspecified",
    explanation_hashes: Optional[Dict[str, str]] = None,
    symmetry_matches: Optional[list] = None,
) -> Dict[str, Any]:
    """Assemble the complete forensic evidence chain according to Section 9.1."""
    evidence = {
        "input_sha256": hash_raw_bytes(input_bytes),
        "model": {
            "name": model_name,
            "version": model_version,
            "weights_sha256": weights_sha256,
        },
        "config_sha256": config_sha256,
        "operating_point": {
            "decision_threshold": float(round(threshold, 4)),
            "temperature": float(round(temperature, 4)),
        },
        "timestamps": {
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "verdict": {
            "decision": decision,  # 'genuine', 'manipulated', or 'inconclusive'
            "calibrated_score": float(round(score, 4)),
            "rule_fired": rule_fired,
        },
        "explanations": explanation_hashes or {},
        "symmetry_copy_move_evidence": symmetry_matches or [],
    }

    # Hash canonical evidence representation
    evidence["report_sha256"] = canonical_json_hash(evidence)
    return evidence
