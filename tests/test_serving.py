"""Tests for Phase 7: Video Pipeline, Document Rectification, ONNX Parity, and FastAPI Service."""

import io
import os
import tempfile
import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from core.model import KryptosIDNet
from document.locate import detect_and_rectify_document
from document.regions import extract_id_regions
from serving.api import app, get_engine
from serving.export_onnx import export_to_onnx
from serving.inference import ForensicInferenceEngine
from video.aggregate import aggregate_video_frame_scores, decide_video_verdict
from video.liveness import verify_liveness_challenge
from video.sampler import extract_face_crops


def test_video_liveness_and_aggregation():
    """Verify video liveness challenge verification and temporal score aggregation."""
    # 1. Liveness check with motion
    frames_moving = [np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8) for _ in range(6)]
    passed, details = verify_liveness_challenge(frames_moving, challenge_type="blink")
    assert passed is True
    assert details["passed"] is True

    # 2. Static replay attack (zero motion)
    static_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
    frames_static = [static_frame.copy() for _ in range(6)]
    passed_static, _ = verify_liveness_challenge(frames_static, min_variation_threshold=3.0)
    assert passed_static is False

    # 3. Aggregation of scores
    scores = [0.1, 0.12, 0.15, 0.85, 0.90, 0.12]
    agg_score, top_k, penalty = aggregate_video_frame_scores(scores, top_k_ratio=0.2)
    assert 0.0 <= agg_score <= 1.0
    assert top_k > 0.8  # Top-k should capture high spike

    # 4. Verdict decision
    verdict, rule = decide_video_verdict(agg_score, threshold=0.45, liveness_passed=True, face_count=6)
    assert verdict == "manipulated"
    assert "aggregated_score_exceeds_threshold" in rule


def test_document_rectification_and_regions():
    """Verify 4-point document perspective transform and sub-region extraction."""
    # Create white document inside dark background
    canvas_img = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas_img[50:350, 100:500] = 240

    rectified, detected = detect_and_rectify_document(canvas_img, canonical_size=(856, 540))
    assert rectified.shape == (540, 856, 3)

    regions = extract_id_regions(rectified)
    assert "photo_region" in regions
    assert "mrz_region" in regions
    assert regions["photo_region"].shape == (224, 224, 3)


def test_onnx_export_and_strict_parity():
    """Verify ONNX export (opset 17) and parity check (|PyTorch - ONNX| < 1e-4)."""
    model = KryptosIDNet(use_mock_backbones=True)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "kryptosid_test.onnx")
        path, max_diff = export_to_onnx(
            model=model,
            output_path=onnx_file,
            opset_version=17,
            tolerance=1e-4,
        )

        assert os.path.exists(path)
        assert max_diff < 1e-4, f"Parity difference {max_diff:.6e} must be < 1e-4"


def test_fastapi_endpoints():
    """Verify complete FastAPI endpoint workflow using TestClient."""
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/v1/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. Model Card
    res_model = client.get("/v1/model")
    assert res_model.status_code == 200
    assert res_model.json()["name"] == "KryptosIDNet"

    # 3. Analyze Image
    img = Image.new("RGB", (224, 224), color=(180, 120, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res_analyze = client.post(
        "/v1/analyze",
        files={"file": ("test_face.jpg", img_bytes, "image/jpeg")},
        data={"media_type": "image"},
    )
    assert res_analyze.status_code == 200
    data = res_analyze.json()
    assert "decision" in data
    assert "report_id" in data
    assert "report_sha256" in data
    assert "explanations" in data

    report_id = data["report_id"]

    # 4. Get Report Metadata
    res_get = client.get(f"/v1/reports/{report_id}")
    assert res_get.status_code == 200
    assert res_get.json()["report_id"] == report_id

    # 5. Download Report PDF
    res_pdf = client.get(f"/v1/reports/{report_id}/pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-")

    # 6. Verify Report Signature
    res_verify = client.post(f"/v1/reports/{report_id}/verify")
    assert res_verify.status_code == 200
    assert res_verify.json()["valid"] is True

    # 7. Oversized rejection check (HTTP 413)
    oversized_bytes = b"0" * (52428800 + 1024)
    res_oversized = client.post(
        "/v1/analyze",
        files={"file": ("bomb.bin", oversized_bytes, "application/octet-stream")},
    )
    assert res_oversized.status_code == 413
