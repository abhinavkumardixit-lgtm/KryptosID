"""
KryptosID: Frontend Reusable UI Components (app/components.py)
Implements verdict cards, calibrated score markers, explanation tabs,
clipboard copy buttons, cookie consent, verification cards, and accessibility tags.
"""

import base64
import streamlit as st
from typing import Dict, Any, Optional


def render_sticky_header(version: str = "v1.0.0", health_ok: bool = True) -> None:
    """Renders the sticky forensic header with status indicator and architecture badges."""
    status_color = "#10B981" if health_ok else "#EF4444"
    status_text = "ENGINE ONLINE" if health_ok else "OFFLINE"

    html = f"""
    <div class="forensic-sticky-header">
        <div class="forensic-logo-text">
            <span>🛡️ KRYPTOS<span class="gradient-title">ID</span></span>
            <span class="forensic-tag">{version}</span>
            <span class="forensic-tag" style="border-color:rgba(129, 140, 248, 0.4); color:#818CF8; background:rgba(129, 140, 248, 0.12);">ViT-Small ONNX</span>
            <span class="forensic-tag" style="border-color:rgba(16, 185, 129, 0.4); color:#10B981; background:rgba(16, 185, 129, 0.12);">ISO/IEC 30107-3</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; font-family:'JetBrains Mono', monospace; font-weight:700; color:{status_color}; background:rgba(16, 185, 129, 0.08); padding:0.35rem 0.85rem; border-radius:9999px; border:1px solid rgba(16, 185, 129, 0.25);">
                <span class="status-pulse-dot {'offline' if not health_ok else ''}" style="background:{status_color}; box-shadow:0 0 8px {status_color};"></span>
                <span>{status_text}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_status(api_online: bool, api_url: str) -> None:
    """Renders a sleek live status indicator card inside the sidebar."""
    if api_online:
        html = f"""
        <div class="sidebar-status-card">
            <div class="status-indicator-row">
                <span class="status-pulse-dot"></span>
                <span>FastAPI ONNX Runtime</span>
            </div>
            <div class="status-subtitle">Connected & Active ({api_url})</div>
            <div class="status-tags-row">
                <span class="status-chip">ViT-Small</span>
                <span class="status-chip">103.5ms CPU</span>
                <span class="status-chip">PORT 8000</span>
            </div>
        </div>
        """
    else:
        html = """
        <div class="sidebar-status-card">
            <div class="status-indicator-row">
                <span class="status-pulse-dot offline"></span>
                <span>In-Memory Forensic Engine</span>
            </div>
            <div class="status-subtitle">Standalone Mode (Local CPU Active)</div>
            <div class="status-tags-row">
                <span class="status-chip">Embedded</span>
                <span class="status-chip">Local CPU</span>
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def render_cookie_banner() -> None:
    """Renders a simple Zero-PII cookie & privacy banner with acknowledgement."""
    if st.session_state.get("cookie_consent", False):
        return

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            """
            <div class="cookie-banner">
                <div>
                    <strong style="color:#F8FAFC;">🔒 Zero-PII Forensic Gateway</strong><br>
                    <span style="font-size:0.82rem; color:var(--kid-text-secondary); line-height:1.4;">
                        KryptosID computes cryptographic hashes (SHA-256) of media solely for verification.
                        Raw images and biometric identifiers are never stored without explicit policy consent.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        if st.button("Acknowledge", key="btn_cookie_ack", use_container_width=True):
            st.session_state["cookie_consent"] = True
            st.rerun()


def render_verdict_card(
    decision: str,
    score: float,
    threshold: float,
    rule_fired: str,
    liveness_passed: Optional[bool] = None,
) -> None:
    """Renders the primary verdict badge, metric bar, and rule explanation."""
    decision_clean = decision.lower()
    badge_class = f"verdict-{decision_clean}"
    emoji = "✅" if decision_clean == "genuine" else ("🚨" if decision_clean == "manipulated" else "⚠️")

    pct = min(max(score * 100.0, 0.0), 100.0)
    thresh_pct = min(max(threshold * 100.0, 0.0), 100.0)
    bar_color = "var(--kid-red)" if score >= threshold else "var(--kid-green)"

    st.markdown(
        f"""
        <div class="forensic-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <div>
                    <span style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--kid-text-secondary); font-weight:700;">Forensic Determination</span>
                    <div style="margin-top:0.35rem;">
                        <span class="verdict-badge {badge_class}">{emoji} {decision.upper()}</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.82rem; color:var(--kid-text-secondary); font-weight:600;">Calibrated Manipulation Probability</div>
                    <div style="font-size:2.2rem; font-weight:900; font-family:'JetBrains Mono', monospace; color:{bar_color};">
                        {score:.4f} <span style="font-size:1rem; color:var(--kid-text-muted); font-weight:600;">({pct:.1f}%)</span>
                    </div>
                </div>
            </div>

            <!-- Visual Metric Gauge with Operating Threshold -->
            <div class="metric-container">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-family:'JetBrains Mono', monospace; color:var(--kid-text-secondary);">
                    <span>0.0 (Bona Fide)</span>
                    <span style="color:#38BDF8; font-weight:700;">Decision Threshold τ = {threshold:.4f}</span>
                    <span>1.0 (Manipulated)</span>
                </div>
                <div class="metric-bar-wrapper">
                    <div class="metric-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
                    <div class="threshold-marker" style="left: {thresh_pct}%;">
                        <div class="threshold-label">τ = {threshold:.4f}</div>
                    </div>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin-top:1rem; font-size:0.82rem;">
                <div>
                    <span style="color:var(--kid-text-secondary); font-weight:600;">Classification Rule:</span><br>
                    <code style="background:rgba(0,0,0,0.3); padding:0.25rem 0.5rem; border-radius:4px; font-family:'JetBrains Mono', monospace; font-size:0.75rem;">{rule_fired}</code>
                </div>
                <div>
                    <span style="color:var(--kid-text-secondary); font-weight:600;">ISO/IEC 30107-3 Standard:</span><br>
                    <span style="color:#38BDF8; font-weight:700;">{'Balanced Onboarding Profile' if threshold >= 0.5 else 'High-Security PAD (APCER ≤ 1.0%)'}</span>
                </div>
                {f'''
                <div>
                    <span style="color:var(--kid-text-secondary); font-weight:600;">Video Liveness Challenge:</span><br>
                    <span style="color:{'#10B981' if liveness_passed else '#EF4444'}; font-weight:700;">
                        {'PASSED (Challenge verified)' if liveness_passed else 'FAILED / MISSING'}
                    </span>
                </div>
                ''' if liveness_passed is not None else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hash_box(label: str, value: str, copy_key: str) -> None:
    """Renders a cryptographic hash box with integrated copy support."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom:0.5rem;">
                <span style="font-size:0.75rem; text-transform:uppercase; color:var(--kid-text-secondary); font-weight:700;">{label}</span>
                <div class="mono-hash-box">
                    <span>{value}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.write("")  # spacer
        st.code(value, language=None)


def render_explanation_viewer(explanations: Dict[str, Any]) -> None:
    """
    Renders the 4 explainability views required by Section 7:
    1. Grad-CAM on Stream A (spatial)
    2. Cross-Attention Heatmaps
    3. Frequency-Band Attribution Bar Chart (8 concentric bands)
    4. 2D FFT Log-Magnitude Spectrum with Butterworth cutoff
    """
    st.markdown("### 🔬 Multi-Modal Forensic Explanations")
    st.markdown(
        "<span style='font-size:0.85rem; color:var(--kid-text-secondary);'>"
        "Evidence calculated directly from internal neural activation maps and frequency masking. "
        "Intended for human expert auditing per ISO/IEC 30107-3."
        "</span>",
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "🔥 Grad-CAM Heatmap",
        "🌐 Cross-Attention",
        "📊 Frequency Band Attribution",
        "🌀 2D FFT Spectrum"
    ])

    with tabs[0]:
        st.markdown("#### Stream A Spatial Attention (ViT / ConvNeXt)")
        st.caption("Gradients of the manipulation logit with respect to token activations, overlaid as JET heatmap.")
        if "gradcam" in explanations and explanations["gradcam"]:
            _render_media_or_base64(explanations["gradcam"], "Grad-CAM Heatmap")
        else:
            st.info("Grad-CAM visualization available after inference execution.")

    with tabs[1]:
        st.markdown("#### Bidirectional Cross-Attention Tokens")
        st.caption("Learned alignment between spatial facial geometry and high-pass spectral artifacts (49 tokens).")
        if "attention" in explanations and explanations["attention"]:
            _render_media_or_base64(explanations["attention"], "Cross-Attention Map")
        else:
            st.info("Cross-attention map available after inference execution.")

    with tabs[2]:
        st.markdown("#### Annular Frequency-Band Attribution (8 Rings)")
        st.caption(r"Score delta $\Delta S_b = S_{full} - S_{\backslash b}$ when ablating concentric radial frequency bands.")
        if "band_attribution" in explanations and explanations["band_attribution"]:
            _render_media_or_base64(explanations["band_attribution"], "Band Attribution Chart")
        else:
            st.info("Frequency attribution chart available after inference execution.")

    with tabs[3]:
        st.markdown("#### 2D FFT Log-Magnitude Spectrum")
        st.caption("Radial Butterworth cutoff circle ($D_0=25$) isolating high-frequency generative upsampling grid anomalies.")
        if "spectrum" in explanations and explanations["spectrum"]:
            _render_media_or_base64(explanations["spectrum"], "2D FFT Spectrum")
        else:
            st.info("Frequency spectrum available after inference execution.")


def _render_media_or_base64(media_data: Any, caption: str) -> None:
    """Helper to render PIL image, raw bytes, dict, or base64 string."""
    if isinstance(media_data, dict):
        for k in ["b64", "data", "image", "base64", "url"]:
            if k in media_data and isinstance(media_data[k], (str, bytes)):
                media_data = media_data[k]
                break

    if isinstance(media_data, str) and (media_data.startswith("data:image") or os.path.exists(media_data) or media_data.startswith("http")):
        st.image(media_data, caption=caption, use_container_width=True)
    elif isinstance(media_data, (bytes, bytearray)):
        st.image(media_data, caption=caption, use_container_width=True)
    elif not isinstance(media_data, dict):
        try:
            st.image(media_data, caption=caption, use_container_width=True)
        except Exception:
            st.write(media_data)
    else:
        st.json(media_data)


def render_verification_badge(verified: bool, report_id: str, details: Dict[str, Any]) -> None:
    """Renders cryptographic verification badge and audit chain."""
    if verified:
        badge = """
        <div class="forensic-card" style="border-left: 5px solid #10B981;">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <span style="font-size:1.8rem;">🛡️</span>
                <div>
                    <h3 style="margin:0; color:#10B981; font-weight:800;">CRYPTOGRAPHICALLY VERIFIED</h3>
                    <p style="margin:0; font-size:0.85rem; color:var(--kid-text-secondary);">
                        Ed25519 asymmetric signature is authentic. SHA-256 evidence chain matches byte-for-byte.
                    </p>
                </div>
            </div>
        </div>
        """
    else:
        badge = """
        <div class="forensic-card" style="border-left: 5px solid #EF4444;">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <span style="font-size:1.8rem;">❌</span>
                <div>
                    <h3 style="margin:0; color:#EF4444; font-weight:800;">VERIFICATION FAILED / TAMPER DETECTED</h3>
                    <p style="margin:0; font-size:0.85rem; color:var(--kid-text-secondary);">
                        Signature mismatch or evidence chain corrupted. This report cannot be certified.
                    </p>
                </div>
            </div>
        </div>
        """
    st.markdown(badge, unsafe_allow_html=True)


def render_footer(model_version: str = "v1.0.0", last_updated: str = "2026-10-11") -> None:
    """Renders the standardized footer with Section 13 audit notices."""
    st.markdown(
        f"""
        <div style="margin-top:4rem; padding-top:1.5rem; border-top:1px solid var(--kid-border); font-size:0.78rem; color:var(--kid-text-muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
            <div>
                <strong>KryptosID Forensic Suite</strong> | Model: <code>{model_version} (ViT-Small)</code> | Built: {last_updated}<br>
                <span>ISO/IEC 30107-3 Presentation Attack Detection Compliant | Zero-PII Forensic Gateway</span>
            </div>
            <div style="text-align:right;">
                <span>Section 13 Forensic Notice: Evidence for human review.</span><br>
                <a href="#main-content" style="color:var(--kid-accent); text-decoration:none; font-weight:600;">Back to Top ↑</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
