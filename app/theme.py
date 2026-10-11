"""
KryptosID: Frontend Design System & Theme Engine (app/theme.py)
Provides CSS design tokens, WCAG 4.5:1 compliant dark/light palettes,
modern glassmorphism, glowing status badges, refined sidebar navigation,
and accessible responsive typography.
"""

from typing import Dict, Any

THEME_CONFIGS: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg_primary": "#070B14",
        "bg_secondary": "#0D1424",
        "bg_card": "rgba(17, 24, 39, 0.85)",
        "bg_glass": "rgba(15, 23, 42, 0.82)",
        "border_color": "rgba(56, 189, 248, 0.18)",
        "border_glow": "rgba(56, 189, 248, 0.40)",
        "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_muted": "#64748B",
        "accent_primary": "#38BDF8",       # Sky Cyan
        "accent_secondary": "#818CF8",     # Electric Indigo
        "accent_hover": "#0284C7",
        "verdict_genuine": "#10B981",      # Emerald
        "verdict_manipulated": "#EF4444",  # Crimson
        "verdict_inconclusive": "#F59E0B", # Amber
        "surface_shadow": "0 10px 30px -5px rgba(0, 0, 0, 0.6), 0 8px 16px -6px rgba(0, 0, 0, 0.5)",
    },
    "light": {
        "bg_primary": "#F8FAFC",
        "bg_secondary": "#F1F5F9",
        "bg_card": "rgba(255, 255, 255, 0.95)",
        "bg_glass": "rgba(255, 255, 255, 0.88)",
        "border_color": "#CBD5E1",
        "border_glow": "rgba(14, 165, 233, 0.35)",
        "text_primary": "#0F172A",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "accent_primary": "#0284C7",
        "accent_secondary": "#4F46E5",
        "accent_hover": "#0369A1",
        "verdict_genuine": "#059669",
        "verdict_manipulated": "#DC2626",
        "verdict_inconclusive": "#D97706",
        "surface_shadow": "0 10px 25px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05)",
    }
}


def get_custom_css(dark_mode: bool = True) -> str:
    """
    Generates CSS stylesheet dynamically matching dark/light mode preference,
    complying with WCAG AA/AAA (>= 4.5:1 contrast) and Section 11 specifications.
    """
    mode = "dark" if dark_mode else "light"
    theme = THEME_CONFIGS[mode]

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --kid-bg: {theme['bg_primary']};
        --kid-secondary: {theme['bg_secondary']};
        --kid-surface: {theme['bg_card']};
        --kid-surface-glass: {theme['bg_glass']};
        --kid-border: {theme['border_color']};
        --kid-glow: {theme['border_glow']};
        --kid-text: {theme['text_primary']};
        --kid-text-secondary: {theme['text_secondary']};
        --kid-text-muted: {theme['text_muted']};
        --kid-accent: {theme['accent_primary']};
        --kid-accent-alt: {theme['accent_secondary']};
        --kid-accent-hover: {theme['accent_hover']};
        --kid-shadow: {theme['surface_shadow']};
        --kid-green: {theme['verdict_genuine']};
        --kid-red: {theme['verdict_manipulated']};
        --kid-amber: {theme['verdict_inconclusive']};
    }}

    /* Global Body & Container Overrides */
    .stApp {{
        background: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.07) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(129, 140, 248, 0.06) 0%, transparent 45%),
                    var(--kid-bg);
        color: var(--kid-text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        color: #F8FAFC !important;
    }}

    /* Accessibility Skip Link (Hidden until focused) */
    .skip-to-content {{
        position: fixed;
        top: -120px;
        left: 1rem;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        color: #0F172A;
        padding: 0.5rem 1.25rem;
        z-index: 99999;
        border-radius: 6px;
        font-weight: 700;
        text-decoration: none;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        opacity: 0;
        pointer-events: none;
        transition: top 0.25s ease, opacity 0.25s ease;
    }}
    .skip-to-content:focus {{
        top: 1rem;
        opacity: 1;
        pointer-events: auto;
        outline: 3px solid #38BDF8;
    }}

    /* =======================================================
       SIDEBAR REFINEMENT & NAVIGATION ENHANCEMENT
       ======================================================= */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #060911 0%, #0A0F1D 50%, #0B1224 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.4) !important;
    }}

    /* Sidebar headers and typography */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
    }}

    /* Custom Radio Buttons as Tab Pills in Sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {{
        gap: 0.4rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: 0.55rem 0.85rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        color: #F8FAFC !important;
        transform: translateX(2px) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18) 0%, rgba(129, 140, 248, 0.18) 100%) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.22) !important;
    }}

    /* Sidebar Status Card */
    .sidebar-status-card {{
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    .status-indicator-row {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .status-pulse-dot {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulseGreen 2s infinite ease-in-out;
    }}
    .status-pulse-dot.offline {{
        background: #EF4444;
        box-shadow: 0 0 8px #EF4444;
        animation: none;
    }}
    .status-subtitle {{
        font-size: 0.74rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        line-height: 1.3;
    }}
    .status-tags-row {{
        display: flex;
        gap: 0.4rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }}
    .status-chip {{
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }}

    @keyframes pulseGreen {{
        0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}

    /* =======================================================
       STICKY FORENSIC HEADER
       ======================================================= */
    .forensic-sticky-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(10, 15, 29, 0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(56, 189, 248, 0.20);
        padding: 0.75rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        border-radius: 0 0 14px 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }}
    .forensic-logo-text {{
        font-weight: 900;
        font-size: 1.45rem;
        letter-spacing: -0.03em;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #F8FAFC;
    }}
    .gradient-title {{
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .forensic-tag {{
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-weight: 700;
        letter-spacing: 0.05em;
    }}

    /* =======================================================
       CARDS & INTERACTIVE COMPONENTS
       ======================================================= */
    .forensic-card {{
        background: var(--kid-surface);
        backdrop-filter: blur(10px);
        border: 1px solid var(--kid-border);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--kid-shadow);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .forensic-card:hover {{
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45), 0 0 15px rgba(56, 189, 248, 0.12);
    }}

    /* Decision Verdict Badges */
    .verdict-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.4rem;
        border-radius: 9999px;
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    .verdict-genuine {{
        background: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 2px solid #10B981;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }}
    .verdict-manipulated {{
        background: rgba(239, 68, 68, 0.16);
        color: #EF4444;
        border: 2px solid #EF4444;
        animation: pulseDanger 2s infinite ease-in-out;
    }}
    .verdict-inconclusive {{
        background: rgba(245, 158, 11, 0.16);
        color: #F59E0B;
        border: 2px solid #F59E0B;
    }}

    @keyframes pulseDanger {{
        0%, 100% {{ box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }}
        50% {{ box-shadow: 0 0 25px rgba(239, 68, 68, 0.5); }}
    }}

    /* Metric Bar & Threshold Marker */
    .metric-container {{
        background: rgba(0, 0, 0, 0.35);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border: 1px solid var(--kid-border);
    }}
    .metric-bar-wrapper {{
        position: relative;
        height: 26px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 13px;
        overflow: visible;
        margin: 1.6rem 0 0.6rem 0;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }}
    .metric-bar-fill {{
        height: 100%;
        border-radius: 13px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .threshold-marker {{
        position: absolute;
        top: -7px;
        bottom: -7px;
        width: 3.5px;
        background: #F8FAFC;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.9);
        z-index: 10;
        border-radius: 2px;
    }}
    .threshold-label {{
        position: absolute;
        top: -24px;
        transform: translateX(-50%);
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(15, 23, 42, 0.9);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        white-space: nowrap;
    }}

    /* Monospace Hash Box */
    .mono-hash-box {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: rgba(0, 0, 0, 0.4);
        color: var(--kid-text);
        padding: 0.65rem 0.95rem;
        border-radius: 8px;
        border: 1px solid var(--kid-border);
        word-break: break-all;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* Cookie / Zero-PII Banner */
    .cookie-banner {{
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        box-shadow: var(--kid-shadow);
    }}

    /* Quick Demo Sample Cards */
    .demo-sample-card {{
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .demo-sample-card:hover {{
        border-color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        transform: translateY(-2px);
    }}

    /* Streamlit Button Enhancements */
    button[kind="primary"] {{
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    button[kind="primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }}
    button[kind="secondary"] {{
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    button[kind="secondary"]:hover {{
        border-color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.1) !important;
        transform: translateY(-1px) !important;
    }}

    /* Print Stylesheet (@media print) */
    @media print {{
        .stApp {{
            background: #FFFFFF !important;
            color: #000000 !important;
        }}
        .forensic-sticky-header,
        .cookie-banner,
        [data-testid="stSidebar"],
        button {{
            display: none !important;
        }}
        .forensic-card {{
            border: 1px solid #CCCCCC !important;
            box-shadow: none !important;
            page-break-inside: avoid;
        }}
        .verdict-badge {{
            box-shadow: none !important;
            border: 2px solid #000000 !important;
        }}
    }}
    </style>
    """
