"""
core/i18n.py — Trilingual internationalization engine (RU, UZ-Latin, EN).
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

I18N_DIR = Path(__file__).resolve().parent.parent / "i18n"
SUPPORTED_LANGUAGES = ["ru", "uz", "en"]
DEFAULT_LANGUAGE = "ru"

# In-memory translations cache
_TRANSLATIONS_CACHE: Dict[str, Dict[str, str]] = {}


def load_translations(lang: str) -> Dict[str, str]:
    """
    Loads JSON translation dictionary for a given language code.
    """
    lang_code = lang.lower().strip()
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = DEFAULT_LANGUAGE

    if lang_code in _TRANSLATIONS_CACHE:
        return _TRANSLATIONS_CACHE[lang_code]

    file_path = I18N_DIR / f"{lang_code}.json"
    if not file_path.exists():
        logger.warning(f"Translation file not found at {file_path}, falling back to empty dict.")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _TRANSLATIONS_CACHE[lang_code] = data
    return data


def get_current_language() -> str:
    """
    Determines active language from Streamlit session_state or default.
    """
    try:
        import streamlit as st
        return st.session_state.get("lang", DEFAULT_LANGUAGE)
    except Exception:
        return DEFAULT_LANGUAGE


def set_current_language(lang: str) -> None:
    """
    Sets active language in Streamlit session_state.
    """
    lang_code = lang.lower().strip()
    if lang_code in SUPPORTED_LANGUAGES:
        try:
            import streamlit as st
            st.session_state["lang"] = lang_code
        except Exception:
            pass


def t(key: str, lang: Optional[str] = None, **kwargs: Any) -> str:
    """
    Translates key into active language with fallback to Russian, then to key itself.
    Supports str.format(**kwargs) interpolation.
    """
    active_lang = lang if lang else get_current_language()
    translations = load_translations(active_lang)

    text = translations.get(key)
    if text is None:
        # Fallback to default language (Russian)
        if active_lang != DEFAULT_LANGUAGE:
            ru_translations = load_translations(DEFAULT_LANGUAGE)
            text = ru_translations.get(key)

        if text is None:
            # Fallback to key itself
            text = key

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception as e:
            logger.debug(f"Failed to format i18n string '{key}' with kwargs {kwargs}: {e}")
            return text

    return text
