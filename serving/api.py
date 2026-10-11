"""FastAPI Forensic Inference Service.

Implements Section 10.1 of DESIGN.md:
- POST /v1/analyze: Analyzes image, video, or document and produces signed report.
- GET  /v1/reports/{id}: Report metadata retrieval.
- GET  /v1/reports/{id}/pdf: Download signed forensic PDF report.
- POST /v1/reports/{id}/verify: Cryptographically verifies report hashes and Ed25519 signature.
- GET  /v1/health: Liveness probe and model load status.
- GET  /v1/model: Model card metadata, weights hash, and dataset licenses.
- Request size limits enforced before decoding (HTTP 413).
"""

from typing import Optional
import json
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from serving.inference import ForensicInferenceEngine
from serving.schemas import (
    AnalyzeResponse,
    HealthResponse,
    ModelCardResponse,
    VerifyRequest,
    VerifyResponse,
)

app = FastAPI(
    title="KryptosID Forensic Gateway API",
    description="Explainable forensic gateway and anti-spoofing engine for Video KYC (V-CIP) and identity verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
engine: Optional[ForensicInferenceEngine] = None


@app.on_event("startup")
def startup_event():
    global engine
    engine = ForensicInferenceEngine()


def get_engine() -> ForensicInferenceEngine:
    global engine
    if engine is None:
        engine = ForensicInferenceEngine()
    return engine


@app.get("/v1/health", response_model=HealthResponse)
def health():
    """Liveness probe and model load status."""
    eng = get_engine()
    return HealthResponse(
        status="healthy",
        model_loaded=(eng.model is not None),
        device=str(eng.device),
        version="1.0.0",
    )


@app.get("/v1/model", response_model=ModelCardResponse)
def model_card():
    """Model card fields, operating points, and training dataset licenses."""
    eng = get_engine()
    return ModelCardResponse(
        name="KryptosIDNet",
        version="1.0.0",
        weights_sha256="kryptosid_v1_weights_sealed",
        threshold=eng.threshold,
        temperature=eng.temperature,
        operating_point={
            "standard": "ISO/IEC 30107-3",
            "target_apcer": "<= 1.0%",
            "decision_threshold": eng.threshold,
            "inconclusive_margin": eng.inconclusive_margin,
        },
        datasets_used=[
            "Celeb-DF v2 (Academic)",
            "FaceForensics++ (Research)",
            "GenImage (CC BY-NC 4.0)",
            "Synthetic V-CIP Identity Documents (DPDP Compliant)",
        ],
        limitations=[
            "Estimates statistical manipulation evidence, does not establish human identity.",
            "Extreme low-resolution compression or unseen generators may reduce accuracy.",
            "Constitutes evidence for human review, not an autonomous legal determination.",
        ],
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_media(
    file: UploadFile = File(...),
    media_type: str = Form("image"),  # 'image', 'video', or 'document'
):
    """Analyze uploaded media, returns decision, scores, explanation IDs, and report ID."""
    eng = get_engine()

    # Read raw bytes
    raw_bytes = await file.read()
    max_size = eng.config.get("limits", {}).get("max_file_size_bytes", 52428800)

    # Size limit check before any decoding (Section 3.1 & 10.4)
    if len(raw_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({len(raw_bytes)} bytes) exceeds maximum limit ({max_size} bytes)",
        )


    # Auto-detect media type if unspecified
    content_type = file.content_type or ""
    if "video" in content_type or file.filename.lower().endswith((".mp4", ".webm", ".mov", ".avi")):
        media_type = "video"
    elif file.filename.lower().endswith((".pdf", ".tiff")):
        media_type = "document"

    try:
        if media_type == "video":
            result = eng.analyze_video(raw_bytes)
        elif media_type == "document":
            result = eng.analyze_document(raw_bytes)
        else:
            result = eng.analyze_image(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")

    return AnalyzeResponse(
        decision=result["decision"],
        score=result["score"],
        threshold=result["threshold"],
        rule_fired=result["rule_fired"],
        report_id=result["report_id"],
        report_sha256=result["report_sha256"],
        input_sha256=result["input_sha256"],
        media_type=media_type,
        quality_indicators=result.get("quality_indicators"),
        explanations=result["explanations"],
        download_pdf_url=result["download_pdf_url"],
    )


@app.get("/v1/reports/{report_id}")
def get_report(report_id: str):
    """Retrieve report metadata."""
    eng = get_engine()
    rec = eng.store.get_report(report_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return rec


@app.get("/v1/reports/{report_id}/pdf")
def download_report_pdf(report_id: str):
    """Download signed forensic PDF report."""
    eng = get_engine()
    rec = eng.store.get_report(report_id)
    if not rec or not rec.get("pdf_path") or not os.path.exists(rec["pdf_path"]):
        raise HTTPException(status_code=404, detail="PDF report file not found on server")

    return FileResponse(
        rec["pdf_path"],
        media_type="application/pdf",
        filename=f"KryptosID_Report_{report_id}.pdf",
    )


@app.post("/v1/reports/{report_id}/verify", response_model=VerifyResponse)
def verify_report(report_id: str, request: Optional[VerifyRequest] = None):
    """Recomputes hashes and verifies the Ed25519 signature (Section 9.2 & 10.1)."""
    eng = get_engine()
    res = eng.store.verify_report(report_id)

    if not res.get("valid"):
        return VerifyResponse(
            valid=False,
            report_id=report_id,
            report_sha256="",
            error=res.get("error", "Verification failed"),
        )

    return VerifyResponse(
        valid=True,
        report_id=report_id,
        report_sha256=res.get("report_sha256", ""),
        error=None,
    )


@app.get("/v1/benchmark")
def get_benchmark():
    """Retrieve raw benchmark comparison metrics JSON."""
    json_path = "reports/benchmark_vit_vs_baseline.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Benchmark metrics not found")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/v1/benchmark/pdf")
def download_benchmark_pdf():
    """Download executive publication-grade benchmark audit PDF report."""
    pdf_path = "reports/KryptosID_Benchmark_Report.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Benchmark PDF not found on server")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="KryptosID_Benchmark_Report.pdf",
    )


@app.get("/v1/benchmark/specs-pdf")
def download_specs_pdf():
    """Download dedicated architectural specifications comparison PDF."""
    pdf_path = "reports/KryptosID_Specification_Comparison.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Specifications PDF not found on server")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="KryptosID_Specification_Comparison.pdf",
    )
