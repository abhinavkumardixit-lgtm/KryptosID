"""Tests for Phase 6: D4 Symmetry Checks, Ed25519 Signing, PDF Reports, and SQLite Store."""

import os
import tempfile
import numpy as np
import pytest

from core.symmetry import (
    D4_TRANSFORMS,
    apply_d4_transform,
    detect_copy_move_d4,
)
from forensics.hashing import (
    build_evidence_chain,
    canonical_json_dumps,
    canonical_json_hash,
    hash_raw_bytes,
)
from forensics.report import generate_forensic_report_pdf
from forensics.signing import (
    Ed25519Signer,
    SignatureVerificationError,
    generate_keypair,
    sign_payload,
    verify_payload,
)
from forensics.store import ForensicReportStore


def test_d4_group_actions_and_properties():
    """Verify algebraic properties and closure of the D4 square symmetry group (Section 4.5)."""
    assert len(D4_TRANSFORMS) == 8

    # Create an asymmetric test image [32, 32]
    img = np.zeros((32, 32), dtype=np.uint8)
    img[5:15, 10:20] = 200
    img[2, 3] = 120

    # 1. Four 90-degree rotations must equal identity (order 4)
    r1 = apply_d4_transform(img, "rot90")
    r2 = apply_d4_transform(r1, "rot90")
    r3 = apply_d4_transform(r2, "rot90")
    r4 = apply_d4_transform(r3, "rot90")
    assert np.array_equal(r4, img), "rot90^4 must equal identity"

    # 2. Horizontal and vertical flips are involutions (order 2)
    fh2 = apply_d4_transform(apply_d4_transform(img, "flip_horizontal"), "flip_horizontal")
    assert np.array_equal(fh2, img), "flip_horizontal^2 must equal identity"

    fv2 = apply_d4_transform(apply_d4_transform(img, "flip_vertical"), "flip_vertical")
    assert np.array_equal(fv2, img), "flip_vertical^2 must equal identity"

    # 3. Transpose reflections are involutions
    fd2 = apply_d4_transform(apply_d4_transform(img, "diag_reflection"), "diag_reflection")
    assert np.array_equal(fd2, img), "diag_reflection^2 must equal identity"


def test_copy_move_detection_d4():
    """Verify detection of forged document patches matched under D4 transforms."""
    doc = np.random.randint(50, 150, (128, 128), dtype=np.uint8)

    # Insert a distinct textured patch at (16, 16) - grid aligned with stride=16
    patch = np.zeros((32, 32), dtype=np.uint8)
    patch[4:28, 4:28] = 250
    patch[10:14, 10:14] = 30
    doc[16:48, 16:48] = patch

    # Copy the patch with a rot90 transformation to (64, 64)
    patch_rot = apply_d4_transform(patch, "rot90")
    doc[64:96, 64:96] = patch_rot

    matches = detect_copy_move_d4(doc, patch_size=32, stride=16, similarity_threshold=0.90)
    assert len(matches) > 0, "D4 copy-move detector should find the forged patch"
    top_match = matches[0]
    assert top_match.similarity >= 0.90
    assert top_match.transform in D4_TRANSFORMS



def test_ed25519_signing_and_tamper_detection():
    """Verify Ed25519 cryptographic signing and rejection of tampered evidence."""
    signer = generate_keypair(key_id="test-key-2026")
    raw_media = b"FAKE_VIDEO_PAYLOAD_RAW_BYTES_12345"

    evidence = build_evidence_chain(
        input_bytes=raw_media,
        decision="genuine",
        score=0.034,
        rule_fired="low_artifact_score",
        threshold=0.452,
        temperature=1.0,
        started_at="2026-10-11T07:00:00Z",
        finished_at="2026-10-11T07:00:02Z",
    )

    signed_record = sign_payload(signer, evidence)
    assert "signature_block" in signed_record

    # 1. Verification succeeds on intact record
    assert verify_payload(signed_record) is True

    # 2. Tampering test: Attacker changes decision to 'manipulated'
    tampered_record = dict(signed_record)
    tampered_record["verdict"] = dict(tampered_record["verdict"])
    tampered_record["verdict"]["decision"] = "manipulated"

    # Must raise SignatureVerificationError
    with pytest.raises(SignatureVerificationError):
        verify_payload(tampered_record)


def test_forensic_pdf_generation_and_validity():
    """Verify ReportLab A4 forensic PDF report generation."""
    signer = generate_keypair()
    evidence = build_evidence_chain(
        input_bytes=b"SAMPLE_IMAGE",
        decision="manipulated",
        score=0.912,
        rule_fired="high_frequency_spectral_artifact",
        threshold=0.45,
        temperature=1.1,
        started_at="2026-10-11T07:00:00Z",
        finished_at="2026-10-11T07:00:01Z",
    )
    signed_rec = sign_payload(signer, evidence)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "forensic_report.pdf")
        out = generate_forensic_report_pdf(
            signed_evidence=signed_rec,
            output_pdf_path=pdf_path,
        )

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 2000

        # Check PDF header
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"


def test_forensic_report_store():
    """Verify SQLite persistence, record retrieval, and re-verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_reports.db")
        store = ForensicReportStore(db_path=db_path)

        signer = generate_keypair()
        evidence = build_evidence_chain(
            input_bytes=b"TEST_DOCUMENT",
            decision="inconclusive",
            score=0.46,
            rule_fired="near_operating_threshold_margin",
            threshold=0.45,
            temperature=1.0,
            started_at="2026-10-11T07:00:00Z",
            finished_at="2026-10-11T07:00:01Z",
        )
        signed_rec = sign_payload(signer, evidence)

        report_id = "rep_2026_test_001"
        store.save_report(report_id, signed_rec, pdf_path="/tmp/fake.pdf")

        # Retrieve
        fetched = store.get_report(report_id)
        assert fetched is not None
        assert fetched["report_id"] == report_id
        assert fetched["decision"] == "inconclusive"

        # Verify
        v_res = store.verify_report(report_id)
        assert v_res["valid"] is True
        assert v_res["report_sha256"] == evidence["report_sha256"]
