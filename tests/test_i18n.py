"""
Unit tests for core/i18n.py
"""
import json
from pathlib import Path
import pytest
from core.i18n import I18N_DIR, SUPPORTED_LANGUAGES, load_translations, t


def test_i18n_files_exist_and_valid_json():
    assert I18N_DIR.exists()
    for lang in SUPPORTED_LANGUAGES:
        file_path = I18N_DIR / f"{lang}.json"
        assert file_path.exists(), f"Missing translation file: {file_path}"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) > 0


def test_key_parity_across_all_languages():
    ru_data = load_translations("ru")
    uz_data = load_translations("uz")
    en_data = load_translations("en")

    ru_keys = set(ru_data.keys())
    uz_keys = set(uz_data.keys())
    en_keys = set(en_data.keys())

    # Ensure exact key parity
    assert ru_keys == uz_keys, f"UZ keys differ from RU: {ru_keys ^ uz_keys}"
    assert ru_keys == en_keys, f"EN keys differ from RU: {ru_keys ^ en_keys}"

    # Ensure no empty translations
    for k, v in ru_data.items():
        assert isinstance(v, str) and len(v.strip()) > 0, f"Empty RU value for {k}"
    for k, v in uz_data.items():
        assert isinstance(v, str) and len(v.strip()) > 0, f"Empty UZ value for {k}"
    for k, v in en_data.items():
        assert isinstance(v, str) and len(v.strip()) > 0, f"Empty EN value for {k}"


def test_translation_function_and_kwargs():
    # Direct language translation
    assert t("field_age", lang="ru") == "Возраст (лет)"
    assert t("field_age", lang="uz") == "Yosh (yillar)"
    assert t("field_age", lang="en") == "Age (years)"

    # Kwargs interpolation
    interpolated = t("card_chd_uncertainty", lang="ru", val="2.5")
    assert "±2.5%" in interpolated

    interpolated_uz = t("card_chd_uncertainty", lang="uz", val="2.5")
    assert "±2.5%" in interpolated_uz


def test_fallback_behavior():
    # Unknown language code should fallback to Russian
    assert t("field_sex", lang="de") == "Биологический пол"

    # Non-existent key should return key itself
    assert t("completely_nonexistent_key_12345", lang="en") == "completely_nonexistent_key_12345"
