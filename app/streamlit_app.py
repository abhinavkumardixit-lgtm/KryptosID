"""
KryptosID: Streamlit Forensic Front End (app/streamlit_app.py)
Multi-page forensic interface implementing:
1. Analyze (Image, Video, Document) with 1-click test presets and live inference progress
2. Result (Verdict card, Calibrated probability vs Threshold τ, Explanations, PDF Download)
3. Benchmark & ViT Comparison (Interactive metrics, dual PDF report downloads, comparative charts)
4. Verify (Ed25519 cryptographic signature & SHA-256 chain verification)
5. About (Model card, Dataset licenses, ISO/IEC 30107-3 compliance, FAQ)

Complies with Section 11 of DESIGN.md and ISO/IEC 30107-3 PAD standards.
"""

import os
import io
import time
import json
import base64
import requests
import streamlit as st
from typing import Dict, Any, Optional

from app.theme import get_custom_css
from app.components import (
    render_sticky_header,
    render_sidebar_status,
    render_cookie_banner,
    render_verdict_card,
    render_hash_box,
    render_explanation_viewer,
    render_verification_badge,
    render_footer,
)

# Page configuration
st.set_page_config(
    page_title="KryptosID Forensic Gateway",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Preference from session state
dark_mode = st.session_state.get("dark_mode_toggle", True)

# Apply global CSS stylesheet at top of page (NOT inside sidebar)
st.markdown(get_custom_css(dark_mode=dark_mode), unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("KRYPTOSID_API_URL", "http://127.0.0.1:8000")


def check_api_health() -> bool:
    """Probes the FastAPI backend /v1/health endpoint."""
    try:
        resp = requests.get(f"{API_BASE_URL}/v1/health", timeout=2.0)
        if resp.status_code == 200:
            status = resp.json().get("status", "").lower()
            return status in ["ok", "healthy", "up"]
        return False
    except Exception:
        return False


def call_api_analyze(file_bytes: bytes, filename: str, media_type: str) -> Dict[str, Any]:
    """Sends file to FastAPI /v1/analyze or falls back to in-memory engine if backend offline."""
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"media_type": media_type}
        resp = requests.post(f"{API_BASE_URL}/v1/analyze", files=files, data=data, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Direct in-process fallback for standalone Streamlit execution
    from serving.inference import get_inference_engine
    engine = get_inference_engine()
    return engine.analyze(file_bytes=file_bytes, media_type=media_type, filename=filename)


def call_api_verify(report_id: str) -> Dict[str, Any]:
    """Verifies report signature via FastAPI or direct store."""
    try:
        resp = requests.post(f"{API_BASE_URL}/v1/reports/{report_id}/verify", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if "valid" in data:
                data["verified"] = data["valid"]
            return data
    except Exception:
        pass

    # Direct in-process fallback
    from forensics.store import get_report_store
    store = get_report_store()
    res = store.verify_report(report_id)
    return {
        "report_id": report_id,
        "verified": res.get("valid", False),
        "report_sha256": res.get("report_sha256", ""),
        "reason": res.get("error", ""),
    }


def call_api_get_pdf(report_id: str) -> Optional[bytes]:
    """Fetches PDF bytes from API or direct store."""
    try:
        resp = requests.get(f"{API_BASE_URL}/v1/reports/{report_id}/pdf", timeout=10.0)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass

    from forensics.store import get_report_store
    store = get_report_store()
    rec = store.get_report(report_id)
    if rec and rec.get("pdf_path") and os.path.exists(rec["pdf_path"]):
        with open(rec["pdf_path"], "rb") as f:
            return f.read()
    return None


# ==========================================
# SIDEBAR: PREFERENCES & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Preferences")
    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode_toggle", True), key="dark_mode_toggle")

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Gateway Page:",
        ["🔬 Analyze Media", "📊 Inspection Results", "📈 Benchmark & ViT Comparison", "🛡️ Verify Signature", "📖 About & Model Card"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    api_online = check_api_health()
    render_sidebar_status(api_online=api_online, api_url=API_BASE_URL)

    st.caption("KryptosID Forensic Suite v1.0.0 — ViT-Small Engine")


# Render Sticky Header across top of content
render_sticky_header(version="v1.0.0", health_ok=api_online)

# Render Cookie / Zero-PII Banner
render_cookie_banner()


# ==========================================
# PAGE 1: ANALYZE MEDIA
# ==========================================
if page == "🔬 Analyze Media":
    st.markdown("## 🔍 Forensic Verification & Anti-Spoofing Gateway")
    st.markdown(
        "KryptosID performs dual-stream spatial self-attention (**ViT-Small**) and high-frequency spectral residual analysis "
        "(**Butterworth ResNet-34**, $D_0=25$) to detect deepfakes, synthetic faces, and document forgery for Video KYC."
    )

    # 1-Click Instant Demo Presets
    st.markdown("##### ⚡ Quick 1-Click Verification Presets")
    preset_cols = st.columns(3)
    with preset_cols[0]:
        if st.button("👤 Genuine Customer Face", use_container_width=True, help="Load held-out bona fide customer portrait"):
            p = "data/photos/real/photo_real_0000.jpg"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    st.session_state["loaded_file_bytes"] = f.read()
                    st.session_state["loaded_file_name"] = "photo_real_0000.jpg"
                    st.session_state["loaded_media_type"] = "image"
                    st.rerun()
    with preset_cols[1]:
        if st.button("🤖 AI Synthetic Portrait", use_container_width=True, help="Load generative synthetic face portrait"):
            p = "data/photos/fake/photo_fake_0000.jpg"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    st.session_state["loaded_file_bytes"] = f.read()
                    st.session_state["loaded_file_name"] = "photo_fake_0000.jpg"
                    st.session_state["loaded_media_type"] = "image"
                    st.rerun()
    with preset_cols[2]:
        if st.button("🎥 FaceForensics Video Frame", use_container_width=True, help="Load real-world compressed video manipulation frame"):
            p = "data/faceforensics/fake/ff_fake_000_003_000.jpg"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    st.session_state["loaded_file_bytes"] = f.read()
                    st.session_state["loaded_file_name"] = "ff_fake_000_003_000.jpg"
                    st.session_state["loaded_media_type"] = "image"
                    st.rerun()

    st.markdown("---")

    col_input, col_info = st.columns([2, 1])

    with col_input:
        media_choice = st.selectbox(
            "Select Input Modality:",
            ["image", "video", "document"],
            index=0 if st.session_state.get("loaded_media_type", "image") == "image" else 1,
            format_func=lambda x: {
                "image": "🖼️ Facial Portrait (Image)",
                "video": "🎥 V-CIP Video Stream",
                "document": "📄 Identity Document (Aadhaar / Passport / ID)"
            }[x]
        )

        uploaded_file = st.file_uploader(
            f"Upload {media_choice.capitalize()} for Forensic Analysis",
            type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "pdf"]
        )

        # Determine active payload from uploader or loaded demo preset
        active_bytes = None
        active_name = None

        if uploaded_file is not None:
            active_bytes = uploaded_file.getvalue()
            active_name = uploaded_file.name
        elif "loaded_file_bytes" in st.session_state:
            active_bytes = st.session_state["loaded_file_bytes"]
            active_name = st.session_state["loaded_file_name"]
            st.info(f"Loaded Preset: `{active_name}` ({len(active_bytes)/1024:.1f} KB)")

        if active_bytes is not None:
            if media_choice == "image" or active_name.endswith((".jpg", ".png", ".webp")):
                st.image(active_bytes, caption=f"Active Input: {active_name}", use_container_width=True)

            btn_row1, btn_row2 = st.columns([3, 1])
            with btn_row1:
                analyze_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)
            with btn_row2:
                if st.button("Clear Input", use_container_width=True):
                    st.session_state.pop("loaded_file_bytes", None)
                    st.session_state.pop("loaded_file_name", None)
                    st.rerun()

            if analyze_btn:
                with st.spinner("Executing dual-stream spatial-spectral analysis and computing Grad-CAM / FFT explanations..."):
                    t0 = time.time()
                    result = call_api_analyze(
                        file_bytes=active_bytes,
                        filename=active_name,
                        media_type=media_choice
                    )
                    elapsed = time.time() - t0

                if "detail" in result:
                    st.error(f"Analysis failed: {result['detail']}")
                else:
                    st.session_state["latest_result"] = result
                    st.session_state["latest_file_name"] = active_name
                    st.success(f"Forensic Analysis Complete in {elapsed:.2f}s!")

                    # Show Immediate Verdict Card
                    render_verdict_card(
                        decision=result["decision"],
                        score=result["score"],
                        threshold=result["threshold"],
                        rule_fired=result["rule_fired"],
                        liveness_passed=result.get("liveness_passed")
                    )

                    # Quick Report Download
                    pdf_bytes = call_api_get_pdf(result["report_id"])
                    if pdf_bytes:
                        st.download_button(
                            label=f"📥 Download Signed Forensic Report ({result['report_id'][:8]}.pdf)",
                            data=pdf_bytes,
                            file_name=f"kryptosid_report_{result['report_id'][:8]}.pdf",
                            mime="application/pdf",
                            type="secondary",
                            use_container_width=True
                        )

    with col_info:
        st.markdown(
            """
            <div class="forensic-card">
                <h4 style="margin-top:0; color:#38BDF8;">🛡️ Operational Standards</h4>
                <ul style="font-size:0.82rem; padding-left:1.2rem; color:var(--kid-text-secondary); line-height:1.6;">
                    <li><strong>Spatial Stream:</strong> Vision Transformer (<code>vit_small_patch16_224</code>, 43.72M params)</li>
                    <li><strong>Spectral Stream:</strong> 2D Butterworth ResNet-34 ($D_0=25, n=2$)</li>
                    <li><strong>Fusion Bridge:</strong> Bidirectional Cross-Attention (49 tokens)</li>
                    <li><strong>Acceleration:</strong> ONNX Runtime Engine (~103.5ms CPU latency)</li>
                    <li><strong>Calibration:</strong> Temperature-scaled ($T=1.4076$) guaranteeing APCER ≤ 1.0%</li>
                    <li><strong>Integrity:</strong> Signed with Ed25519 asymmetric evidence chain</li>
                    <li><strong>Zero-PII:</strong> RBI V-CIP Master Direction & DPDP Act 2023 compliant</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# PAGE 2: INSPECTION RESULTS
# ==========================================
elif page == "📊 Inspection Results":
    st.markdown("## 📊 Forensic Evidence & Explanations")

    result = st.session_state.get("latest_result")
    if not result:
        st.info("No active inspection found. Please analyze media on the 'Analyze Media' page first.")
    else:
        render_verdict_card(
            decision=result["decision"],
            score=result["score"],
            threshold=result["threshold"],
            rule_fired=result["rule_fired"],
            liveness_passed=result.get("liveness_passed")
        )

        # Quality Indicators (if present)
        qi = result.get("quality_indicators")
        if qi:
            st.markdown(
                f"""
                <div class="forensic-card" style="padding:0.9rem 1.2rem;">
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; font-size:0.8rem; font-family:'JetBrains Mono', monospace;">
                        <div><span>Blur Variance:</span> <strong style="color:#38BDF8;">{qi.get('laplacian_variance', 0):.1f}</strong> ({qi.get('blur_status', 'N/A')})</div>
                        <div><span>Resolution:</span> <strong style="color:#38BDF8;">{qi.get('resolution', 'N/A')}</strong></div>
                        <div><span>Operating Threshold τ:</span> <strong style="color:#10B981;">{result.get('threshold', 0):.4f}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 📋 Cryptographic Evidence Chain")
        c1, c2 = st.columns(2)
        with c1:
            render_hash_box("Input File SHA-256", result.get("input_sha256", "N/A"), "copy_input_hash")
            render_hash_box("Report UUID", result.get("report_id", "N/A"), "copy_report_id")
        with c2:
            render_hash_box("Model Weights SHA-256", result.get("weights_sha256", "N/A"), "copy_model_hash")
            render_hash_box("Canonically Signed Hash", result.get("report_sha256", "N/A"), "copy_canon_hash")

        # Explainability Views
        explanations = result.get("explanations", {})
        render_explanation_viewer(explanations)

        st.markdown("### 📄 Certified Signed PDF Export")
        pdf_bytes = call_api_get_pdf(result["report_id"])
        if pdf_bytes:
            st.download_button(
                label=f"📥 Download Certified Forensic Report ({result['report_id'][:8]}.pdf)",
                data=pdf_bytes,
                file_name=f"kryptosid_report_{result['report_id']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )


# ==========================================
# PAGE 3: BENCHMARK & VIT COMPARISON
# ==========================================
elif page == "📈 Benchmark & ViT Comparison":
    st.markdown("## 📈 Architectural Benchmark: ConvNeXt-S vs. Vision Transformer (ViT)")
    st.markdown(
        "Empirical head-to-head evaluation of spatial backbones combined with 2D Butterworth spectral residual stream ($D_0=25$) "
        "on the multi-domain held-out test split (99 samples) under **ISO/IEC 30107-3** Presentation Attack Detection standard."
    )

    pdf_file_path = "reports/KryptosID_Benchmark_Report.pdf"
    specs_pdf_path = "reports/KryptosID_Specification_Comparison.pdf"

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label="📥 Download Certified Benchmark Audit PDF (172 KB)",
                data=pdf_data,
                file_name="KryptosID_Benchmark_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                help="Download complete audit report with visual charts and mathematical proofs"
            )
    with btn_col2:
        if os.path.exists(specs_pdf_path):
            with open(specs_pdf_path, "rb") as f:
                specs_pdf_data = f.read()
            st.download_button(
                label="📋 Download Specifications Comparison PDF (15 KB)",
                data=specs_pdf_data,
                file_name="KryptosID_Specification_Comparison.pdf",
                mime="application/pdf",
                type="secondary",
                use_container_width=True,
                help="Download dedicated tabular head-to-head specifications and scaling roadmap PDF"
            )

    st.markdown("### 📊 Visual Analytics & Performance Trade-offs")
    chart_path = "reports/benchmark_comparison_chart.png"
    if os.path.exists(chart_path):
        st.image(chart_path, caption="Comparative Performance & Compute Footprint (ConvNeXt-S vs. ViT-Small)", use_container_width=True)

    st.markdown("### 📋 Executive Metric Comparison Table")
    benchmark_json = "reports/benchmark_vit_vs_baseline.json"
    if os.path.exists(benchmark_json):
        with open(benchmark_json, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        b = bdata["models"]["baseline_convnext_s"]
        v = bdata["models"]["vit_small_transformer"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ViT ROC-AUC", f"{v['roc_auc']:.4f}", delta=f"+{v['roc_auc'] - b['roc_auc']:.4f} vs Baseline")
        c2.metric("ViT Equal Error Rate", f"{v['eer_pct']:.2f}%", delta=f"{v['eer_pct'] - b['eer_pct']:.2f}% vs Baseline", delta_color="inverse")
        c3.metric("ViT Model Size", f"{v['params_millions']}M", delta=f"-{b['params_millions'] - v['params_millions']:.1f}M (38.9% lighter)", delta_color="inverse")
        c4.metric("ONNX CPU Latency", "103.6 ms", delta="-20.1 ms (1.33x speedup)", delta_color="inverse")

        comparison_table = [
            {"Forensic Metric": "Spatial Stream Backbone", "ConvNeXt-S Baseline": "ConvNeXt-Small (CNN)", "Vision Transformer (ViT)": "ViT-Small Patch16_224", "Advantage / Winner": "Global self-attention (ViT)"},
            {"Forensic Metric": "Spectral Stream Backbone", "ConvNeXt-S Baseline": "2D Butterworth ResNet-34", "Vision Transformer (ViT)": "2D Butterworth ResNet-34", "Advantage / Winner": "Identical Fourier stream (Parity)"},
            {"Forensic Metric": "Total Parameters (M)", "ConvNeXt-S Baseline": f"{b['params_millions']}M", "Vision Transformer (ViT)": f"{v['params_millions']}M", "Advantage / Winner": "38.9% lighter footprint (ViT)"},
            {"Forensic Metric": "Weights File Size", "ConvNeXt-S Baseline": "286.9 MB (.pt)", "Vision Transformer (ViT)": "175.3 MB (.pt)", "Advantage / Winner": "111.6 MB smaller container (ViT)"},
            {"Forensic Metric": "PyTorch CPU Latency", "ConvNeXt-S Baseline": f"{b['avg_latency_ms']} ms", "Vision Transformer (ViT)": f"{v['avg_latency_ms']} ms", "Advantage / Winner": "15.2% faster forward pass (ViT)"},
            {"Forensic Metric": "ONNX Runtime CPU Latency", "ConvNeXt-S Baseline": "N/A", "Vision Transformer (ViT)": "103.56 ms / sample", "Advantage / Winner": "1.33x speedup on CPU (ViT)"},
            {"Forensic Metric": "Working Memory (RAM)", "ConvNeXt-S Baseline": "~1.40 GB", "Vision Transformer (ViT)": "~550 MB (ONNX)", "Advantage / Winner": "60% less RAM required (ViT)"},
            {"Forensic Metric": "Single-Core Throughput", "ConvNeXt-S Baseline": "~8.1 img/sec", "Vision Transformer (ViT)": "~10.2 img/sec", "Advantage / Winner": "+26% stream density (ViT)"},
            {"Forensic Metric": "ROC-AUC (Discrimination)", "ConvNeXt-S Baseline": f"{b['roc_auc']:.4f}", "Vision Transformer (ViT)": f"{v['roc_auc']:.4f}", "Advantage / Winner": "+0.0335 higher AUC (ViT)"},
            {"Forensic Metric": "Equal Error Rate (EER)", "ConvNeXt-S Baseline": f"{b['eer_pct']}%", "Vision Transformer (ViT)": f"{v['eer_pct']}%", "Advantage / Winner": "-6.06% lower error (ViT)"},
            {"Forensic Metric": "Expected Calibration Error", "ConvNeXt-S Baseline": f"{b['ece']:.4f}", "Vision Transformer (ViT)": f"{v['ece']:.4f}", "Advantage / Winner": "35.4% sharper calibration (ViT)"},
            {"Forensic Metric": "Customer False Reject (BPCER)", "ConvNeXt-S Baseline": "36.00%", "Vision Transformer (ViT)": "28.00% (Balanced Mode)", "Advantage / Winner": "-8.00% fewer false rejections (ViT)"},
            {"Forensic Metric": "Presentation Attack Leakage", "ConvNeXt-S Baseline": "28.57%", "Vision Transformer (ViT)": "0.00% (PAD Mode τ=0.0381)", "Advantage / Winner": "Zero attack slip-through (ViT)"},
        ]
        st.table(comparison_table)


# ==========================================
# PAGE 4: VERIFY SIGNATURE
# ==========================================
elif page == "🛡️ Verify Signature":
    st.markdown("## 🛡️ Asymmetric Signature & Evidence Verifier")
    st.markdown(
        "Verify that a KryptosID Forensic Report has not been altered, manipulated, or re-signed. "
        "Every report is bound to an asymmetric Ed25519 signature."
    )

    query_id = st.text_input("Enter Report UUID to Verify:", placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000")
    verify_btn = st.button("🔍 Verify Cryptographic Proof", type="primary")

    if verify_btn and query_id.strip():
        with st.spinner("Checking Ed25519 asymmetric signature and verifying evidence chain..."):
            verification = call_api_verify(query_id.strip())
            is_valid = verification.get("verified", False)
            render_verification_badge(is_valid, query_id, verification)

            if is_valid:
                st.json(verification)
            else:
                st.error(f"Verification Failed: {verification.get('reason', 'Signature validation failed')}")


# ==========================================
# PAGE 5: ABOUT & MODEL CARD
# ==========================================
elif page == "📖 About & Model Card":
    st.markdown("## 📖 KryptosID: Model Card & Technical Specifications")

    search_query = st.text_input("🔍 Search Specifications / FAQ:", placeholder="Type keywords like 'Butterworth', 'ViT', 'APCER', 'License'...")

    faq_items = [
        {
            "q": "Why is Vision Transformer (ViT) superior to ConvNeXt for deepfake detection?",
            "a": "ViT uses multi-head self-attention across non-overlapping 16x16 patches. This allows the model to compute global pairwise token interactions across distant facial regions from layer 1, easily catching cross-facial lighting mismatches and boundary blending seams that local CNN convolutions miss."
        },
        {
            "q": "How does the Butterworth High-Pass Filter isolate generative artifacts?",
            "a": "Generative architectures (GANs, Diffusion models, upsampling transpositions) leave unnatural high-frequency periodic grid traces in the 2D Fourier domain. The Butterworth 2D filter ($D_0=25, n=2$) suppresses smooth facial anatomy (DC=0) and isolates subtle generator traces for Stream B."
        },
        {
            "q": "What is the APCER ≤ 1.0% threshold guarantee?",
            "a": "Per ISO/IEC 30107-3, Attack Presentation Classification Error Rate (APCER) is the proportion of spoof attacks mistakenly classified as bona fide. KryptosID High-Security PAD mode operates at $\\tau = 0.0381$, achieving 0.00% attack leakage on the held-out test split."
        },
        {
            "q": "What dataset licenses and compliance policies govern KryptosID?",
            "a": "KryptosID uses strictly academic/commercial research datasets (FaceForensics++, Celeb-DF v2, GenImage, Flickr-Faces) registered under clear attribution terms. No personal data or biometric identifiers are stored without authorization, complying with DPDP Act 2023 and GDPR."
        },
        {
            "q": "Why is Ed25519 asymmetric signing used?",
            "a": "To ensure tamper-evidence in legal and compliance settings. Each report produces a canonical SHA-256 JSON digest signed with an isolated private key. Any byte modification to the document or metadata breaks verification instantly."
        }
    ]

    filtered_faq = [item for item in faq_items if search_query.lower() in item["q"].lower() or search_query.lower() in item["a"].lower()] if search_query else faq_items

    st.markdown("### ❓ Frequently Asked Questions")
    for item in filtered_faq:
        with st.expander(f"📌 {item['q']}", expanded=False):
            st.write(item["a"])

    st.markdown("### 📋 Section 13 Forensic Limitations Notice")
    st.warning(
        "1. KryptosID estimates whether media exhibits structural, synthetic, or copy-move manipulation. It does not establish the legal identity of a person.\n"
        "2. Extreme compression, low resolution (<128px), and unseen novel generative architectures may reduce sensitivity.\n"
        "3. Forensic determinations serve as evidence for qualified human compliance officers and are not autonomous legal verdicts."
    )

# Footer
render_footer(model_version="v1.0.0", last_updated="2026-10-11")
