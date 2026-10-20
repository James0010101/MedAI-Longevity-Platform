"""
Unit tests for core/units.py
"""
import pytest
from core.units import to_mgdl, from_mgdl, format_value, MGDL_PER_MMOL_CHOL, MGDL_PER_MMOL_GLUC

def test_cholesterol_conversion():
    # 5.0 mmol/L should be ~193.35 mg/dL
    val_mgdl = to_mgdl(5.0, "cholesterol", "mmol/l")
    assert abs(val_mgdl - (5.0 * MGDL_PER_MMOL_CHOL)) < 1e-6
    # Round-trip
    val_roundtrip = from_mgdl(val_mgdl, "cholesterol", "mmol/l")
    assert abs(val_roundtrip - 5.0) < 1e-9

def test_glucose_conversion():
    # 5.5 mmol/L should be ~99.088 mg/dL
    val_mgdl = to_mgdl(5.5, "glucose", "mmol/l")
    assert abs(val_mgdl - (5.5 * MGDL_PER_MMOL_GLUC)) < 1e-6
    # Round-trip
    val_roundtrip = from_mgdl(val_mgdl, "glucose", "mmol/l")
    assert abs(val_roundtrip - 5.5) < 1e-9

def test_identity_when_mgdl():
    assert to_mgdl(200.0, "cholesterol", "mg/dl") == 200.0
    assert from_mgdl(200.0, "cholesterol", "mg/dl") == 200.0

def test_format_value():
    assert format_value(5.234, "cholesterol", "mmol/l") == "5.2"
    assert format_value(200.4, "cholesterol", "mg/dl") == "200"

def test_invalid_analyte():
    with pytest.raises(ValueError):
        to_mgdl(100.0, "unknown_analyte", "mmol/l")
