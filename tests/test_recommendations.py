"""
Unit tests for core/recommendations.py
"""
import pytest
from core.recommendations import generate_recommendations, get_candidate_rules


@pytest.fixture
def optimal_patient():
    return {
        "male": 0,
        "age": 35.0,
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
        "totChol": 170.0,
        "sysBP": 115.0,
        "diaBP": 75.0,
        "BMI": 21.5,
        "heartRate": 68.0,
        "glucose": 82.0,
    }


@pytest.fixture
def smoker_hypertensive_patient():
    return {
        "male": 1,
        "age": 52.0,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "totChol": 235.0,
        "sysBP": 160.0,
        "diaBP": 100.0,
        "BMI": 29.0,
        "heartRate": 78.0,
        "glucose": 110.0,
    }


def test_optimal_patient_recommendations(optimal_patient):
    candidates = get_candidate_rules(optimal_patient)
    assert len(candidates) == 0

    recs = generate_recommendations(optimal_patient)
    assert recs["has_interventions"] is False
    assert len(recs["top_interventions"]) == 0
    assert recs["maintenance_key"] == "rec_maintenance_optimal"


def test_smoker_hypertensive_recommendations(smoker_hypertensive_patient):
    recs = generate_recommendations(smoker_hypertensive_patient)

    assert recs["has_interventions"] is True
    top = recs["top_interventions"]
    assert len(top) <= 3
    assert len(top) > 0

    top_ids = [item["id"] for item in top]
    # Smoking and BP must be identified
    assert "smoking_cessation" in top_ids or "blood_pressure_control" in top_ids

    # Check that interventions yield risk reduction
    for item in top:
        assert item["delta_chd_pp"] <= 0.0
        assert item["delta_heart_age"] <= 0.0
        assert item["delta_life_exp"] >= 0.0
        assert "source" in item
        assert "rationale_key" in item
        assert "action_key" in item

    # Check combined impact
    comb = recs["combined_impact"]
    assert comb is not None
    assert comb["deltas"]["delta_chd_pp"] < 0.0
    assert comb["deltas"]["delta_heart_age"] < 0.0
    assert comb["deltas"]["delta_life_exp"] > 0.0
