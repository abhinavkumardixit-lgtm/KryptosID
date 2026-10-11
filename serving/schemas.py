"""Pydantic Request and Response Schemas for KryptosID API.

Implements Section 10.1 of DESIGN.md:
- Schemas for /v1/analyze, /v1/reports/{id}, /v1/reports/{id}/verify, /v1/health, /v1/model
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ExplanationArtifact(BaseModel):
    file_path: str
    sha256: str


class AnalyzeResponse(BaseModel):
    decision: str = Field(..., description="Verdict: genuine, manipulated, or inconclusive")
    score: float = Field(..., description="Calibrated manipulation probability in [0, 1]")
    threshold: float = Field(..., description="Operating threshold used (APCER <= 1.0%)")
    rule_fired: str = Field(..., description="Explicit rule that produced the decision")
    report_id: str = Field(..., description="Unique report identifier")
    report_sha256: str = Field(..., description="SHA-256 hash of canonical evidence chain")
    input_sha256: str = Field(..., description="SHA-256 of raw uploaded media bytes")
    media_type: str = Field(..., description="image, video, or document")
    quality_indicators: Optional[Dict[str, Any]] = Field(default=None, description="Image blur, quality, and adaptive threshold markers")
    explanations: Dict[str, ExplanationArtifact] = Field(default_factory=dict)
    download_pdf_url: str = Field(..., description="URL to download signed forensic PDF")


class VerifyRequest(BaseModel):
    report_id: Optional[str] = None
    expected_report_sha256: Optional[str] = None


class VerifyResponse(BaseModel):
    valid: bool
    report_id: str
    report_sha256: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str = "1.0.0"


class ModelCardResponse(BaseModel):
    name: str
    version: str
    weights_sha256: str
    threshold: float
    temperature: float
    operating_point: Dict[str, Any]
    datasets_used: List[str]
    limitations: List[str]
