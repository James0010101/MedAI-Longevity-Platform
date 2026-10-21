"""
streamlit_app.py — Main Entry Point for MedAI Longevity Platform.
"""
from pathlib import Path
import sys

# Ensure root workspace path is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from core.i18n import SUPPORTED_LANGUAGES, get_current_language, set_current_language, t
from core.branding import BRAND_CSS, FAVICON_PATH, logo_header_html

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="MedAI Longevity Platform",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(BRAND_CSS, unsafe_allow_html=True)

# 2. Session State Initialization
if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"

if "units" not in st.session_state:
    st.session_state["units"] = "mmol"

# 3. Sidebar Controls: Language & Units
with st.sidebar:
    st.markdown(logo_header_html("MedAI Longevity", size=40, title_tag="h3"), unsafe_allow_html=True)
    st.caption("Cardiovascular Screening & Longevity Platform")
    st.divider()

    # Language Selector
    lang_labels = {
        "ru": "Русский",
        "uz": "O'zbekcha (Lotin)",
        "en": "English",
    }
    current_lang = get_current_language()
    selected_lang = st.selectbox(
        t("lang_select"),
        options=SUPPORTED_LANGUAGES,
        format_func=lambda x: lang_labels.get(x, x),
        index=SUPPORTED_LANGUAGES.index(current_lang) if current_lang in SUPPORTED_LANGUAGES else 0,
        key="lang_selector",
    )
    if selected_lang != st.session_state["lang"]:
        set_current_language(selected_lang)
        st.rerun()

    # Measurement Units
    unit_labels = {
        "mmol": t("unit_mmol"),
        "mgdl": t("unit_mgdl"),
    }
    selected_unit = st.radio(
        t("unit_select"),
        options=["mmol", "mgdl"],
        format_func=lambda x: unit_labels.get(x, x),
        index=0 if st.session_state["units"] == "mmol" else 1,
        key="unit_selector",
    )
    st.session_state["units"] = selected_unit

    st.divider()
    st.caption(t("footer_disclaimer"))

# 4. Multi-Page Navigation
pages = [
    st.Page(
        "app_pages/screening.py",
        title=t("nav_screening"),
        icon="🩺",
        default=True,
    ),
    st.Page(
        "app_pages/methodology.py",
        title=t("nav_methodology"),
        icon="📊",
    ),
    st.Page(
        "app_pages/about.py",
        title=t("nav_about"),
        icon="ℹ️",
    ),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
