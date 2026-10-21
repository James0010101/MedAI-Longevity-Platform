"""
core/branding.py — Logo and shared visual identity of MedAI Longevity Platform.
"""
import base64
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_SVG_PATH = ASSETS_DIR / "logo.svg"
LOGO_PNG_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

BRAND_BLUE = "#2F6BFF"
BRAND_CYAN = "#3CC8F0"
BRAND_NAVY = "#1E2F52"


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """Logo as an inline data URI, usable inside unsafe_allow_html markup."""
    encoded = base64.b64encode(LOGO_SVG_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def logo_header_html(title: str, size: int = 56, title_tag: str = "h1") -> str:
    """Logo + page title in one flex row."""
    return (
        '<div class="medai-brand-row">'
        f'<img src="{logo_data_uri()}" width="{size}" height="{size}" alt="MedAI Longevity logo"/>'
        f"<{title_tag}>{title}</{title_tag}>"
        "</div>"
    )


BRAND_CSS = f"""
<style>
.medai-brand-row {{ display:flex; align-items:center; gap:18px; margin: 0 0 4px 0; }}
.medai-brand-row h1 {{ margin:0; padding:0; line-height:1.1; color:{BRAND_NAVY}; }}
.medai-brand-row h3 {{ margin:0; padding:0; color:{BRAND_NAVY}; }}
.medai-brand-row img {{ filter: drop-shadow(0 6px 14px rgba(47,107,255,.28)); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; box-shadow: 0 8px 26px rgba(30,47,82,.06); }}
[data-testid="stMetricValue"] {{ font-weight: 800; letter-spacing: -0.5px; }}
.stButton > button {{ border-radius: 14px; }}
</style>
"""
