"""
MERIDIAN Dashboard Styles
Custom CSS injection for modern dark theme with emerald accents.
"""

import streamlit as st
import base64
from pathlib import Path


# Color Palette - Emerald Finance Theme
COLORS = {
    # Primary
    'primary': '#10B981',        # Emerald 500
    'primary_light': '#34D399',  # Emerald 400
    'primary_dark': '#059669',   # Emerald 600
    'primary_glow': 'rgba(16, 185, 129, 0.3)',

    # Background
    'bg_dark': '#0F172A',        # Slate 900
    'bg_card': '#1E293B',        # Slate 800
    'bg_elevated': '#334155',    # Slate 700
    'bg_hover': '#475569',       # Slate 600

    # Text
    'text_primary': '#F1F5F9',   # Slate 100
    'text_secondary': '#94A3B8', # Slate 400
    'text_muted': '#64748B',     # Slate 500

    # Semantic
    'success': '#10B981',        # Emerald
    'warning': '#F59E0B',        # Amber
    'danger': '#EF4444',         # Red
    'info': '#3B82F6',           # Blue

    # Chart colors
    'chart_1': '#10B981',        # Emerald
    'chart_2': '#3B82F6',        # Blue
    'chart_3': '#8B5CF6',        # Violet
    'chart_4': '#F59E0B',        # Amber
    'chart_5': '#EC4899',        # Pink
    'chart_6': '#14B8A6',        # Teal
}


def get_base64_image(image_path: str) -> str:
    """Convert image to base64 for embedding in HTML."""
    path = Path(image_path)
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def inject_custom_css():
    """Inject custom CSS styles into the Streamlit app."""

    css = f"""
    <style>
    /* ========== Global Styles ========== */

    .stApp {{
        background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, #0B1120 100%);
    }}

    /* Hide default Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ========== Typography ========== */

    h1, h2, h3 {{
        color: {COLORS['text_primary']} !important;
        font-weight: 600 !important;
    }}

    h1 {{
        background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['primary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* ========== Cards & Containers ========== */

    .stMetric {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['bg_elevated']};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}

    .stMetric:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 20px {COLORS['primary_glow']};
        transition: all 0.3s ease;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS['text_primary']} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    [data-testid="stMetricDelta"] svg {{
        display: none;
    }}

    /* ========== Buttons ========== */

    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 {COLORS['primary_glow']};
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 {COLORS['primary_glow']};
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background: transparent;
        border: 1px solid {COLORS['primary']};
        color: {COLORS['primary']} !important;
    }}

    /* ========== Sidebar ========== */

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['bg_card']} 0%, {COLORS['bg_dark']} 100%);
        border-right: 1px solid {COLORS['bg_elevated']};
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: {COLORS['text_secondary']};
    }}

    /* ========== Tabs ========== */

    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: {COLORS['text_secondary']};
        font-weight: 500;
        padding: 8px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: white !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: {COLORS['bg_elevated']};
        color: {COLORS['text_primary']};
    }}

    /* ========== Data Tables ========== */

    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['bg_elevated']};
    }}

    /* ========== Select Boxes ========== */

    .stSelectbox [data-baseweb="select"] {{
        background: {COLORS['bg_card']};
        border-color: {COLORS['bg_elevated']};
        border-radius: 8px;
    }}

    .stSelectbox [data-baseweb="select"]:hover {{
        border-color: {COLORS['primary']};
    }}

    /* ========== Sliders ========== */

    .stSlider [data-baseweb="slider"] {{
        padding: 0.5rem 0;
    }}

    .stSlider [data-testid="stTickBar"] > div {{
        background: {COLORS['primary']} !important;
    }}

    /* ========== Expanders ========== */

    .streamlit-expanderHeader {{
        background: {COLORS['bg_card']};
        border-radius: 8px;
        border: 1px solid {COLORS['bg_elevated']};
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS['primary']};
    }}

    /* ========== Alerts ========== */

    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid;
    }}

    [data-testid="stAlertContainer"] div[data-testid="stMarkdownContainer"] {{
        color: {COLORS['text_primary']};
    }}

    /* ========== Custom Classes ========== */

    .hero-section {{
        text-align: center;
        padding: 2rem 0;
    }}

    .metric-card {{
        background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_elevated']} 100%);
        border: 1px solid {COLORS['bg_elevated']};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 30px {COLORS['primary_glow']};
        transform: translateY(-2px);
    }}

    .metric-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
        margin: 0.5rem 0;
    }}

    .metric-card .label {{
        color: {COLORS['text_secondary']};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}

    .metric-card .delta {{
        font-size: 0.9rem;
        font-weight: 600;
    }}

    .delta-positive {{
        color: {COLORS['success']};
    }}

    .delta-negative {{
        color: {COLORS['danger']};
    }}

    .feature-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['bg_elevated']};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }}

    .feature-card .icon {{
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }}

    .feature-card h4 {{
        color: {COLORS['text_primary']};
        margin-bottom: 0.5rem;
    }}

    .feature-card p {{
        color: {COLORS['text_secondary']};
        font-size: 0.9rem;
    }}

    .status-indicator {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }}

    .status-active {{
        background: {COLORS['success']};
        box-shadow: 0 0 10px {COLORS['success']};
    }}

    .status-warning {{
        background: {COLORS['warning']};
        box-shadow: 0 0 10px {COLORS['warning']};
    }}

    .status-inactive {{
        background: {COLORS['text_muted']};
    }}

    .glass-card {{
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
    }}

    /* ========== Animations ========== */

    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.5;
        }}
    }}

    .animate-pulse {{
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .animate-fadeIn {{
        animation: fadeIn 0.5s ease-out;
    }}

    /* ========== Scrollbar ========== */

    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_dark']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['bg_elevated']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary']};
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def render_logo_header(logo_path: str = None, title: str = "MERIDIAN"):
    """Render the logo header with gradient styling."""

    if logo_path and Path(logo_path).exists():
        logo_base64 = get_base64_image(logo_path)
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_base64}" alt="MERIDIAN" style="height: 50px; margin-right: 1rem;">'
    else:
        logo_html = ""

    header_html = f"""
    <div class="hero-section animate-fadeIn" style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
        {logo_html}
        <div>
            <h1 style="margin: 0; font-size: 2.5rem;">{title}</h1>
            <p style="color: {COLORS['text_secondary']}; margin: 0.5rem 0 0 0; font-size: 1rem;">
                Portfolio Optimization System
            </p>
        </div>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Render a styled metric card."""

    delta_html = ""
    if delta:
        delta_class = "delta-positive" if delta_positive else "delta-negative"
        delta_symbol = "+" if delta_positive else ""
        delta_html = f'<div class="delta {delta_class}">{delta_symbol}{delta}</div>'

    card_html = f"""
    <div class="metric-card animate-fadeIn">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, description: str):
    """Render a feature highlight card."""

    card_html = f"""
    <div class="feature-card animate-fadeIn">
        <div class="icon">{icon}</div>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_status_badge(status: str, label: str):
    """Render a status indicator badge."""

    status_class = {
        'active': 'status-active',
        'warning': 'status-warning',
        'inactive': 'status-inactive'
    }.get(status, 'status-inactive')

    badge_html = f"""
    <span style="display: inline-flex; align-items: center; padding: 4px 12px;
                 background: {COLORS['bg_card']}; border-radius: 20px; font-size: 0.85rem;">
        <span class="status-indicator {status_class}"></span>
        {label}
    </span>
    """

    st.markdown(badge_html, unsafe_allow_html=True)


def get_chart_colors():
    """Return the chart color sequence for Plotly."""
    return [
        COLORS['chart_1'],
        COLORS['chart_2'],
        COLORS['chart_3'],
        COLORS['chart_4'],
        COLORS['chart_5'],
        COLORS['chart_6'],
    ]


def get_plotly_template():
    """Return Plotly template for dark theme."""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'color': COLORS['text_secondary'],
                'family': 'Inter, sans-serif'
            },
            'title': {
                'font': {
                    'color': COLORS['text_primary'],
                    'size': 18
                }
            },
            'xaxis': {
                'gridcolor': COLORS['bg_elevated'],
                'linecolor': COLORS['bg_elevated'],
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'yaxis': {
                'gridcolor': COLORS['bg_elevated'],
                'linecolor': COLORS['bg_elevated'],
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': COLORS['text_secondary']}
            },
            'colorway': get_chart_colors(),
            'hoverlabel': {
                'bgcolor': COLORS['bg_card'],
                'font': {'color': COLORS['text_primary']}
            }
        }
    }
