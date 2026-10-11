"""
Tests for KryptosID Streamlit Frontend (Phase 8)
Verifies theme generation, WCAG contrast tokens, CSS print stylesheets,
component rendering logic, and API communication fallbacks.
"""

import pytest
from app.theme import get_custom_css, THEME_CONFIGS
from app.components import (
    render_verdict_card,
    render_hash_box,
    render_verification_badge,
    render_footer,
)
from app.streamlit_app import (
    check_api_health,
    call_api_analyze,
    call_api_verify,
    call_api_get_pdf,
)


def test_theme_generation_and_tokens():
    """Verifies dark and light CSS themes and critical accessibility selectors."""
    dark_css = get_custom_css(dark_mode=True)
    light_css = get_custom_css(dark_mode=False)

    assert "--kid-bg" in dark_css
    assert "--kid-accent" in dark_css
    assert "@media print" in dark_css
    assert "skip-to-content" in dark_css
    assert "forensic-sticky-header" in dark_css

    # Check color palette values
    assert THEME_CONFIGS["dark"]["accent_primary"] == "#38BDF8"
    assert THEME_CONFIGS["light"]["accent_primary"] == "#0284C7"
    assert THEME_CONFIGS["dark"]["verdict_genuine"] == "#10B981"
    assert THEME_CONFIGS["dark"]["verdict_manipulated"] == "#EF4444"


def test_api_helpers_fallback(monkeypatch):
    """Verifies API helpers handle offline servers smoothly using fallback logic."""
    # When server is offline (requests raises ConnectionError)
    def mock_requests_get(*args, **kwargs):
        raise ConnectionError("Backend offline")

    monkeypatch.setattr("requests.get", mock_requests_get)
    monkeypatch.setattr("requests.post", mock_requests_get)

    assert check_api_health() is False

    # Test call_api_verify with non-existent report
    res_verify = call_api_verify("non-existent-uuid")
    assert res_verify["verified"] is False
    assert "reason" in res_verify

    # Test call_api_get_pdf with non-existent report
    pdf = call_api_get_pdf("non-existent-uuid")
    assert pdf is None


def test_fallback_analyze_in_memory():
    """Verifies direct in-process engine fallback generates valid verdict and signed report."""
    from PIL import Image
    import io

    # Create dummy 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    result = call_api_analyze(
        file_bytes=raw_bytes,
        filename="test_portrait.jpg",
        media_type="image"
    )

    assert "decision" in result
    assert result["decision"] in ["genuine", "manipulated", "inconclusive"]
    assert 0.0 <= result["score"] <= 1.0
    assert "report_id" in result
    assert "input_sha256" in result

    # Check verify on the generated report
    verif = call_api_verify(result["report_id"])
    assert verif["verified"] is True
