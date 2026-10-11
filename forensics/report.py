"""Forensic PDF Report Generation with Cryptographic Evidence and Embedded Visuals.

Implements Section 9.3 of DESIGN.md:
- Built with reportlab.pdfgen.canvas on standard A4 pages.
- Sections:
  1. Header & Summary Verdict Badge (GENUINE / MANIPULATED / INCONCLUSIVE).
  2. Cryptographic Evidence Chain Table (SHA-256 hashes of input, weights, config, and report).
  3. Embedded Explanation Figures (Grad-CAM, Attention maps, Band attribution, Spectrum).
  4. Ed25519 Cryptographic Signature Block & Public Key ID.
  5. Regulatory Limitations Statement (Section 13).
  6. Independent Verification Command and API endpoint.
"""

from typing import Dict, List, Optional, Any
import os
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_forensic_report_pdf(
    signed_evidence: Dict[str, Any],
    output_pdf_path: str,
    explanation_images: Optional[Dict[str, str]] = None,
) -> str:
    """Generate an A4 forensic PDF report with cryptographic signature and explanation visuals.
    
    Args:
        signed_evidence: Evidence chain dictionary including 'signature_block'.
        output_pdf_path: Destination path for the PDF file.
        explanation_images: Optional dictionary of {name: file_path} for PNG visuals.
        
    Returns:
        The generated PDF file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4  # 595.27 x 841.89 points

    margin_x = 40.0
    cursor_y = height - 40.0

    # -------------------------------------------------------------
    # PAGE 1: Executive Summary & Evidence Chain
    # -------------------------------------------------------------

    # Header Bar
    c.setFillColor(colors.HexColor("#0f172a"))  # Slate 900
    c.rect(margin_x, cursor_y - 35, width - 2 * margin_x, 45, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x + 15, cursor_y - 12, "KryptosID Forensic Analysis Report")
    c.setFont("Helvetica", 9)
    c.drawString(margin_x + 15, cursor_y - 28, "Explainable Deepfake & Identity Manipulation Evidence Chain")

    cursor_y -= 55

    # Verdict Badge
    verdict = signed_evidence.get("verdict", {})
    decision = verdict.get("decision", "inconclusive").upper()
    score = verdict.get("calibrated_score", 0.0)
    rule = verdict.get("rule_fired", "default")
    thresh = signed_evidence.get("operating_point", {}).get("decision_threshold", 0.5)

    badge_color = colors.HexColor("#dc2626") if decision == "MANIPULATED" else (
        colors.HexColor("#16a34a") if decision == "GENUINE" else colors.HexColor("#ca8a04")
    )

    c.setFillColor(badge_color)
    c.roundRect(margin_x, cursor_y - 45, 140, 40, radius=4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(margin_x + 70, cursor_y - 25, decision)
    c.setFont("Helvetica", 8)
    c.drawCentredString(margin_x + 70, cursor_y - 38, f"Score: {score:.3f} | Thresh: {thresh:.3f}")

    # Verdict Details Box
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(margin_x + 150, cursor_y - 45, width - 2 * margin_x - 150, 40, radius=4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin_x + 160, cursor_y - 18, f"Primary Trigger: {rule}")
    c.setFont("Helvetica", 8)
    c.drawString(margin_x + 160, cursor_y - 32, f"Calibrated Score: {score:.4f}  |  Operating Threshold: {thresh:.4f} (ISO/IEC 30107-3 APCER <= 1.0%)")

    cursor_y -= 60

    # Section 1: Cryptographic Evidence Chain Table
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, cursor_y, "1. Forensic Evidence Chain & Provenance")
    cursor_y -= 15

    model_info = signed_evidence.get("model", {})
    timestamps = signed_evidence.get("timestamps", {})
    sig_block = signed_evidence.get("signature_block", {})

    chain_items = [
        ("Input Media SHA-256", signed_evidence.get("input_sha256", "N/A")),
        ("Report Canonical SHA-256", signed_evidence.get("report_sha256", "N/A")),
        ("Model Identifier", f"{model_info.get('name', 'KryptosIDNet')} v{model_info.get('version', '1.0.0')}"),
        ("Model Weights SHA-256", model_info.get("weights_sha256", "N/A")),
        ("Serving Config SHA-256", signed_evidence.get("config_sha256", "N/A")),
        ("Analysis Timestamps", f"Started: {timestamps.get('started_at', 'N/A')} | Finished: {timestamps.get('finished_at', 'N/A')}"),
        ("Signature Algorithm", sig_block.get("algorithm", "Ed25519")),
        ("Signing Key ID", sig_block.get("key_id", "N/A")),
        ("Public Key Hex", sig_block.get("public_key", "N/A")),
        ("Digital Signature (Hex)", sig_block.get("signature", "N/A")),
    ]

    row_h = 16.0
    c.setFont("Helvetica", 7.5)
    for idx, (label, val) in enumerate(chain_items):
        bg = colors.HexColor("#f1f5f9") if idx % 2 == 0 else colors.white
        c.setFillColor(bg)
        c.rect(margin_x, cursor_y - row_h + 3, width - 2 * margin_x, row_h, fill=1, stroke=0)

        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(margin_x + 6, cursor_y - 9, label)

        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Courier", 7.0 if len(str(val)) > 40 else 7.5)
        # Truncate visually if too long but keep legible
        display_val = str(val)[:76] + ("..." if len(str(val)) > 76 else "")
        c.drawString(margin_x + 150, cursor_y - 9, display_val)

        cursor_y -= row_h

    cursor_y -= 15

    # Section 2: D4 Symmetry Checks
    sym_matches = signed_evidence.get("symmetry_copy_move_evidence", [])
    if sym_matches:
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin_x, cursor_y, "2. D4 Dihedral Symmetry Copy-Move Candidates (Section 4.5)")
        cursor_y -= 12

        c.setFont("Helvetica", 7.5)
        for m in sym_matches[:3]:
            c.setFillColor(colors.HexColor("#475569"))
            c.drawString(margin_x + 6, cursor_y, f"• Match Transform: {m.get('transform')} | Similarity: {m.get('similarity')} | Regions: {m.get('region_a')} <-> {m.get('region_b')}")
            cursor_y -= 11
        cursor_y -= 10

    # Section 3: Visual Explanations (First 2 figures on Page 1 if room, otherwise Page 2)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, cursor_y, "3. Forensic Visual Explanations")
    cursor_y -= 18

    fig_w = 240.0
    fig_h = 135.0

    if explanation_images:
        gradcam_img = explanation_images.get("gradcam")
        spectrum_img = explanation_images.get("spectrum_plot")

        if gradcam_img and os.path.exists(gradcam_img):
            c.drawImage(gradcam_img, margin_x, cursor_y - fig_h, width=fig_w, height=fig_h, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(margin_x, cursor_y - fig_h - 10, "Fig 1: Grad-CAM Saliency Heatmap (Stream A)")

        if spectrum_img and os.path.exists(spectrum_img):
            c.drawImage(spectrum_img, margin_x + fig_w + 20, cursor_y - fig_h, width=fig_w, height=fig_h, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(margin_x + fig_w + 20, cursor_y - fig_h - 10, "Fig 2: 2D FFT Spectrum & D0=25 Cutoff")

        cursor_y -= (fig_h + 30)

    # -------------------------------------------------------------
    # PAGE 2: Band Attribution, Limitations & Verification Instructions
    # -------------------------------------------------------------
    c.showPage()
    cursor_y = height - 40.0

    # Header Page 2
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, cursor_y, "KryptosID: Detailed Attribution & Verification Chain")
    cursor_y -= 25

    if explanation_images:
        band_img = explanation_images.get("band_attribution")
        attn_img = explanation_images.get("attn_spatial")

        if band_img and os.path.exists(band_img):
            c.drawImage(band_img, margin_x, cursor_y - 150, width=500, height=140, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(margin_x, cursor_y - 160, "Fig 3: Frequency-Band Attribution Across 8 Concentric Annular Rings")
            cursor_y -= 185

        if attn_img and os.path.exists(attn_img):
            c.drawImage(attn_img, margin_x, cursor_y - 120, width=fig_w, height=110, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(margin_x, cursor_y - 130, "Fig 4: Cross-Attention Activation Map (Spatial-queries-Spectral)")
            cursor_y -= 150

    # Limitations Statement (Section 13)
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(margin_x, cursor_y - 75, width - 2 * margin_x, 70, radius=4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin_x + 10, cursor_y - 16, "Limitations & Regulatory Scope (DESIGN.md Section 13):")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#334155"))
    c.drawString(margin_x + 10, cursor_y - 30, "• The model estimates statistical evidence of manipulation or generative synthesis; it does not establish human identity.")
    c.drawString(margin_x + 10, cursor_y - 42, "• Adversarial attacks, extreme low-resolution compression, or unseen generators may impair classification accuracy.")
    c.drawString(margin_x + 10, cursor_y - 54, "• This report constitutes evidentiary support for human-in-the-loop compliance review, not an autonomous legal determination.")
    cursor_y -= 95

    # Independent Cryptographic Verification Box
    c.setFillColor(colors.HexColor("#0f172a"))
    c.roundRect(margin_x, cursor_y - 65, width - 2 * margin_x, 60, radius=4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin_x + 15, cursor_y - 18, "Cryptographic Verification Instructions:")
    c.setFont("Helvetica", 8)
    c.drawString(margin_x + 15, cursor_y - 32, "To independently verify this report, execute the following CLI command or send a POST request:")
    c.setFont("Courier-Bold", 8)
    c.drawString(margin_x + 15, cursor_y - 48, f"python -m kryptosid verify --report-sha256 {signed_evidence.get('report_sha256', '')}")

    c.save()
    return output_pdf_path
