"""
Unit tests for core/thresholds.py
"""
import pytest
from core.thresholds import classify

def test_blood_pressure_thresholds():
    # Optimal < 120 / < 80
    r_opt = classify("bp", 119, secondary_value=79)
    assert r_opt["level"] == "ok"
    assert "optimal" in r_opt["label_key"]

    # Normal 120-129 / 80-84
    r_norm = classify("bp", 120, secondary_value=80)
    assert r_norm["level"] == "ok"
    assert "normal" in r_norm["label_key"]

    # High normal 130-139 / 85-89
    r_hn = classify("bp", 130, secondary_value=80)
    assert r_hn["level"] == "warn"
    assert "high_normal" in r_hn["label_key"]

    # Hypertension >= 140 / >= 90
    r_hyp1 = classify("bp", 140, secondary_value=80)
    assert r_hyp1["level"] == "high"
    assert "hypertension" in r_hyp1["label_key"]

    r_hyp2 = classify("bp", 125, secondary_value=90)
    assert r_hyp2["level"] == "high"
    assert "hypertension" in r_hyp2["label_key"]

def test_cholesterol_thresholds():
    # Desirable < 200
    assert classify("totchol", 199)["level"] == "ok"
    # Borderline 200-239
    assert classify("totchol", 200)["level"] == "warn"
    assert classify("totchol", 239)["level"] == "warn"
    # High >= 240
    assert classify("totchol", 240)["level"] == "high"

def test_hdl_thresholds():
    # Male cutoff 40
    assert classify("hdl", 39.9, sex="male")["level"] == "warn"
    assert classify("hdl", 40.0, sex="male")["level"] == "ok"
    # Female cutoff 50
    assert classify("hdl", 49.9, sex="female")["level"] == "warn"
    assert classify("hdl", 50.0, sex="female")["level"] == "ok"

def test_glucose_thresholds():
    # Normal < 110
    assert classify("glucose", 109)["level"] == "ok"
    # Impaired 110-125
    assert classify("glucose", 110)["level"] == "warn"
    assert classify("glucose", 125)["level"] == "warn"
    # Suspect >= 126
    assert classify("glucose", 126)["level"] == "high"

def test_bmi_thresholds():
    assert classify("bmi", 18.4)["level"] == "warn"
    assert classify("bmi", 18.5)["level"] == "ok"
    assert classify("bmi", 24.9)["level"] == "ok"
    assert classify("bmi", 25.0)["level"] == "warn"
    assert classify("bmi", 29.9)["level"] == "warn"
    assert classify("bmi", 30.0)["level"] == "high"
