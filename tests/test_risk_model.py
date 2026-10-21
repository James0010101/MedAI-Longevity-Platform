"""
Unit tests for core/risk_model.py
"""
import numpy as np
import pytest

from core.risk_model import (
    compute_risk_trajectory,
    explain_prediction,
    get_optimal_healthy_profile,
    load_risk_model,
    patient_dict_to_raw_vector,
    predict_patient,
    simulate_intervention,
)


@pytest.fixture
def sample_high_risk_patient():
    return {
        "male": 1,
        "age": 55.0,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "totChol": 240.0,
        "sysBP": 150.0,
        "diaBP": 95.0,
        "BMI": 28.5,
        "heartRate": 80.0,
        "glucose": 115.0,
    }


def test_patient_dict_vectorization(sample_high_risk_patient):
    vec = patient_dict_to_raw_vector(sample_high_risk_patient)
    assert vec.shape == (13,)
    assert vec[0] == 1.0  # male
    assert vec[1] == 55.0  # age
    assert vec[7] == 240.0  # totChol
    assert vec[8] == 150.0  # sysBP


def test_predict_patient_output_structure(sample_high_risk_patient):
    res = predict_patient(sample_high_risk_patient)

    assert "p_chd" in res
    assert "p_chd_std" in res
    assert "p_diabetes" in res
    assert "heart_age" in res
    assert "heart_age_delta" in res
    assert "life_expectancy" in res
    assert "dagostino_cvd_risk" in res

    # Probability bounds
    assert 0.0 <= res["p_chd"] <= 1.0
    assert 0.0 <= res["p_diabetes"] <= 1.0
    assert res["p_chd_std"] >= 0.0

    # Heart age should be accelerated for a smoker with 150/95 BP
    assert res["heart_age"] > sample_high_risk_patient["age"]
    assert res["heart_age_delta"] > 0.0

    # Life expectancy should be realistic
    assert 55.0 <= res["life_expectancy"] <= 95.0


def test_explain_prediction_completeness(sample_high_risk_patient):
    xai = explain_prediction(sample_high_risk_patient, head="chd", m_steps=50)

    assert "attributions_pp" in xai
    assert "sum_attributions_pp" in xai
    assert "target_diff_pp" in xai

    # Completeness check: sum(attributions_pp) should approximate target_diff_pp
    sum_attr = xai["sum_attributions_pp"]
    target_diff = xai["target_diff_pp"]

    # Difference between sum of IG attributions and actual model difference should be <= 0.5 pp
    assert abs(sum_attr - target_diff) < 0.5

    # Check that high risk factors (sysBP, glucose, tobacco) have positive contributions
    attrs = xai["attributions_pp"]
    assert attrs["sysBP"] > 0
    assert attrs["glucose"] > 0
    # Combined tobacco attribution (currentSmoker + cigsPerDay) must be positive
    assert (attrs["currentSmoker"] + attrs["cigsPerDay"]) > 0



def test_simulate_intervention(sample_high_risk_patient):
    # Simulate smoking cessation and blood pressure control to 120
    changes = {
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "sysBP": 120.0,
        "diaBP": 80.0,
    }

    sim = simulate_intervention(sample_high_risk_patient, changes)

    assert "base" in sim
    assert "simulated" in sim
    assert "deltas" in sim

    deltas = sim["deltas"]
    # Quitting smoking and dropping BP from 150 to 120 must reduce CHD risk
    assert deltas["delta_chd_pp"] < 0.0
    # Must decrease heart age
    assert deltas["delta_heart_age"] < 0.0
    # Must increase life expectancy
    assert deltas["delta_life_exp"] > 0.0


def test_compute_risk_trajectory(sample_high_risk_patient):
    years = (0, 5, 10)
    traj = compute_risk_trajectory(
        sample_high_risk_patient,
        years_ahead=years,
        improved_changes={"currentSmoker": 0, "sysBP": 120.0}
    )

    assert len(traj) == len(years) * 2  # current and improved

    current_records = [r for r in traj if r["scenario"] == "current"]
    improved_records = [r for r in traj if r["scenario"] == "improved"]

    assert len(current_records) == 3
    assert len(improved_records) == 3

    # Natural aging should increase CHD risk over time
    assert current_records[2]["p_chd_percent"] > current_records[0]["p_chd_percent"]

    # Improved scenario should have lower risk than current at each point
    for curr, imp in zip(current_records, improved_records):
        assert imp["p_chd_percent"] < curr["p_chd_percent"]
        assert imp["heart_age"] < curr["heart_age"]


def test_group_attributions_merges_smoking_features():
    from core.risk_model import group_attributions

    attrs = {"currentSmoker": -4.2, "cigsPerDay": 6.3, "sysBP": 8.5}
    grouped = group_attributions(attrs)
    assert "currentSmoker" not in grouped and "cigsPerDay" not in grouped
    assert grouped["smoking"] == 2.1
    assert grouped["sysBP"] == 8.5
    # additive: totals are preserved
    assert round(sum(attrs.values()), 2) == round(sum(grouped.values()), 2)
