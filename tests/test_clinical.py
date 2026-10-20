"""
Unit tests for core/clinical.py
"""
import math
import pytest
from core.clinical import (
    calculate_dagostino_risk,
    calculate_dagostino_linear_predictor,
    calculate_heart_age,
    get_who_remaining_life,
    calculate_life_expectancy_estimate,
    DAGOSTINO_COEFFS
)

def test_optimal_profile_heart_age_equals_chronological_age():
    # An individual with optimal values (SBP 125, TC 180, HDL 45, no meds, no smoker, no dm)
    # MUST have heart_age == chronological age
    for sex in ("male", "female"):
        for age in (35.0, 48.0, 60.0):
            ha = calculate_heart_age(
                age=age,
                sex=sex,
                sbp=125.0,
                bp_meds=0,
                tot_chol=180.0,
                hdl=45.0,
                smoker=0,
                diabetes=0
            )
            assert abs(ha - age) < 0.05, f"Failed for sex={sex}, age={age}: got ha={ha}"

def test_heart_age_monotonicity():
    base_kwargs = dict(
        age=50.0,
        sex="male",
        sbp=130.0,
        bp_meds=0,
        tot_chol=200.0,
        hdl=45.0,
        smoker=0,
        diabetes=0
    )
    base_ha = calculate_heart_age(**base_kwargs)

    # 1. Higher SBP -> higher heart age
    ha_sbp = calculate_heart_age(**{**base_kwargs, "sbp": 160.0})
    assert ha_sbp > base_ha

    # 2. Higher TC -> higher heart age
    ha_tc = calculate_heart_age(**{**base_kwargs, "tot_chol": 260.0})
    assert ha_tc > base_ha

    # 3. Higher HDL -> lower heart age (protective factor)
    ha_hdl = calculate_heart_age(**{**base_kwargs, "hdl": 65.0})
    assert ha_hdl < base_ha

    # 4. Smoking -> higher heart age
    ha_smk = calculate_heart_age(**{**base_kwargs, "smoker": 1})
    assert ha_smk > base_ha

    # 5. Diabetes -> higher heart age
    ha_dm = calculate_heart_age(**{**base_kwargs, "diabetes": 1})
    assert ha_dm > base_ha

def test_women_untreated_bp_coefficient():
    # Verify that women with bp_meds=0 use 2.76157 rather than 2.82263
    c = DAGOSTINO_COEFFS["female"]
    assert c["beta_sbp_untreated"] == 2.76157
    assert c["beta_sbp_treated"] == 2.82263

    sbp = 150.0
    L_untreated = calculate_dagostino_linear_predictor(
        age=50, sex="female", sbp=sbp, bp_meds=0, tot_chol=180, hdl=45, smoker=0, diabetes=0
    )
    L_treated = calculate_dagostino_linear_predictor(
        age=50, sex="female", sbp=sbp, bp_meds=1, tot_chol=180, hdl=45, smoker=0, diabetes=0
    )
    diff = L_treated - L_untreated
    expected_diff = (c["beta_sbp_treated"] - c["beta_sbp_untreated"]) * math.log(sbp)
    assert abs(diff - expected_diff) < 1e-9

def test_dagostino_risk_properties():
    # Risk in (0, 1) and monotonically increases with age
    r30 = calculate_dagostino_risk(30, "male", 130, 0, 200, 45, 0, 0)
    r50 = calculate_dagostino_risk(50, "male", 130, 0, 200, 45, 0, 0)
    r70 = calculate_dagostino_risk(70, "male", 130, 0, 200, 45, 0, 0)

    assert 0.0 < r30 < r50 < r70 < 1.0

def test_who_life_table_interpolation():
    # Value at exact boundary
    ex_0_m = get_who_remaining_life(0, "male")
    assert ex_0_m > 70.0  # Uzbekistan male life expectancy at birth ~71.9

    ex_30_m = get_who_remaining_life(30, "male")
    ex_50_m = get_who_remaining_life(50, "male")
    ex_70_m = get_who_remaining_life(70, "male")

    # Remaining years monotonically decrease with age
    assert ex_0_m > ex_30_m > ex_50_m > ex_70_m > 0

    # Total estimated lifespan age + ex is monotonic or stable
    total_30 = 30 + ex_30_m
    total_70 = 70 + ex_70_m
    assert total_70 >= total_30

def test_four_clinical_profiles():
    # Profile A: Healthy Woman, 35
    ha_a = calculate_heart_age(35, "female", 115, 0, 170, 60, 0, 0)
    assert ha_a <= 35.0  # optimal parameters -> heart age younger or equal to chronological
    le_a = calculate_life_expectancy_estimate(35, "female", ha_a)
    assert le_a["delta"] >= 0.0

    # Profile B: Smoker Man, 52
    ha_b = calculate_heart_age(52, "male", 150, 0, 250, 38, 1, 0)
    assert ha_b > 52.0  # multiple risk factors -> advanced heart age
    le_b = calculate_life_expectancy_estimate(52, "male", ha_b)
    assert le_b["delta"] < 0.0  # accelerated vascular aging reduces projected life

    # Profile C: Woman on BP meds, 60
    ha_c = calculate_heart_age(60, "female", 135, 1, 220, 50, 0, 0)
    assert ha_c > 60.0

    # Profile D: Diabetic Man, 66
    ha_d = calculate_heart_age(66, "male", 160, 1, 260, 35, 1, 1)
    assert ha_d > 66.0
    risk_b = calculate_dagostino_risk(52, "male", 150, 0, 250, 38, 1, 0)
    risk_d = calculate_dagostino_risk(66, "male", 160, 1, 260, 35, 1, 1)
    assert risk_d > risk_b
