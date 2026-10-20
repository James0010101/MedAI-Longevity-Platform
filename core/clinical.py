"""
Clinical risk formulas and longevity calculations:
1. D'Agostino et al. (Circulation 2008) 10-Year General CVD Risk
2. Analytical closed-form Heart/Vascular Age
3. WHO Life Table interpolation for Uzbekistan
"""
import math
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from core.config import DATA_DIR, DEFAULT_HDL_MGDL

WHO_CSV_PATH = DATA_DIR / "who_life_table_uzb.csv"

# D'Agostino 2008 General CVD Laboratory Model Coefficients
DAGOSTINO_COEFFS = {
    "male": {
        "beta_age": 3.06117,
        "beta_tc": 1.12370,
        "beta_hdl": -0.93263,
        "beta_sbp_untreated": 1.93303,
        "beta_sbp_treated": 1.99881,
        "beta_smk": 0.65451,
        "beta_dm": 0.57367,
        "s0_10": 0.88936,
        "l_mean": 23.9388,
    },
    "female": {
        "beta_age": 2.32888,
        "beta_tc": 1.20904,
        "beta_hdl": -0.70833,
        "beta_sbp_untreated": 2.76157,
        "beta_sbp_treated": 2.82263,
        "beta_smk": 0.52873,
        "beta_dm": 0.69154,
        "s0_10": 0.95012,
        "l_mean": 26.1931,
    }
}

# Optimal reference profile values for vascular age
OPTIMAL_TC = 180.0
OPTIMAL_HDL = 45.0
OPTIMAL_SBP = 125.0

def _get_sex_key(sex: Union[int, str, float]) -> str:
    s = str(sex).strip().lower()
    if s in ("1", "1.0", "male", "m", "мужчина", "мужской"):
        return "male"
    return "female"

def calculate_dagostino_linear_predictor(
    age: float,
    sex: Union[int, str],
    sbp: float,
    bp_meds: Union[int, bool],
    tot_chol: float,
    hdl: float = DEFAULT_HDL_MGDL,
    smoker: Union[int, bool] = 0,
    diabetes: Union[int, bool] = 0
) -> float:
    """
    Computes linear predictor L for D'Agostino 2008 model.
    """
    sex_key = _get_sex_key(sex)
    c = DAGOSTINO_COEFFS[sex_key]

    age_val = max(1.0, float(age))
    tc_val = max(1.0, float(tot_chol))
    hdl_val = max(1.0, float(hdl))
    sbp_val = max(1.0, float(sbp))
    is_treated = bool(bp_meds)
    is_smoker = 1.0 if smoker else 0.0
    has_dm = 1.0 if diabetes else 0.0

    beta_sbp = c["beta_sbp_treated"] if is_treated else c["beta_sbp_untreated"]

    L = (
        c["beta_age"] * math.log(age_val) +
        c["beta_tc"] * math.log(tc_val) +
        c["beta_hdl"] * math.log(hdl_val) +
        beta_sbp * math.log(sbp_val) +
        c["beta_smk"] * is_smoker +
        c["beta_dm"] * has_dm
    )
    return L

def calculate_dagostino_risk(
    age: float,
    sex: Union[int, str],
    sbp: float,
    bp_meds: Union[int, bool],
    tot_chol: float,
    hdl: float = DEFAULT_HDL_MGDL,
    smoker: Union[int, bool] = 0,
    diabetes: Union[int, bool] = 0
) -> float:
    """
    Computes 10-year general CVD risk using D'Agostino et al. (Circulation 2008).
    Risk = 1 - S0^exp(L - L_mean)
    """
    sex_key = _get_sex_key(sex)
    c = DAGOSTINO_COEFFS[sex_key]
    L = calculate_dagostino_linear_predictor(
        age, sex, sbp, bp_meds, tot_chol, hdl, smoker, diabetes
    )
    risk = 1.0 - math.pow(c["s0_10"], math.exp(L - c["l_mean"]))
    return min(1.0, max(0.0, risk))

def calculate_heart_age(
    age: float,
    sex: Union[int, str],
    sbp: float,
    bp_meds: Union[int, bool],
    tot_chol: float,
    hdl: float = DEFAULT_HDL_MGDL,
    smoker: Union[int, bool] = 0,
    diabetes: Union[int, bool] = 0
) -> float:
    """
    Computes cardiovascular (vascular) age using the closed-form analytical equation:
    heart_age = exp((L_patient - C_ref) / beta_age)
    where C_ref is the non-age linear predictor terms for an optimal reference profile
    (SBP 125 untreated, TC 180, HDL 45, non-smoker, non-diabetic).
    """
    sex_key = _get_sex_key(sex)
    c = DAGOSTINO_COEFFS[sex_key]

    # Non-age terms of reference optimal profile
    C_ref = (
        c["beta_tc"] * math.log(OPTIMAL_TC) +
        c["beta_hdl"] * math.log(OPTIMAL_HDL) +
        c["beta_sbp_untreated"] * math.log(OPTIMAL_SBP)
    )

    L_patient = calculate_dagostino_linear_predictor(
        age, sex, sbp, bp_meds, tot_chol, hdl, smoker, diabetes
    )

    heart_age_raw = math.exp((L_patient - C_ref) / c["beta_age"])
    # Return numerical heart age clipped to physiological limits
    return min(95.0, max(18.0, heart_age_raw))

# Convenient alias
calculate_framingham_heart_age = calculate_heart_age

# Cache WHO life table in memory

_WHO_LIFE_TABLE = None

def _load_who_table() -> pd.DataFrame:
    global _WHO_LIFE_TABLE
    if _WHO_LIFE_TABLE is None:
        if not WHO_CSV_PATH.exists():
            raise FileNotFoundError(f"WHO life table file not found at {WHO_CSV_PATH}")
        _WHO_LIFE_TABLE = pd.read_csv(WHO_CSV_PATH)
    return _WHO_LIFE_TABLE

def get_who_remaining_life(age: float, sex: Union[int, str]) -> float:
    """
    Returns the remaining life expectancy e_x at age x via linear interpolation
    from official WHO life tables for Uzbekistan.
    """
    df = _load_who_table()
    sex_str = _get_sex_key(sex)
    sub = df[df["sex"] == sex_str].sort_values(by="age_start").reset_index(drop=True)

    if sub.empty:
        raise ValueError(f"No records found for sex '{sex_str}' in WHO life table.")

    age_val = float(age)
    ages = sub["age_start"].values
    exs = sub["ex"].values

    if age_val <= ages[0]:
        return float(exs[0])
    if age_val >= ages[-1]:
        return float(exs[-1])

    # Linear interpolation between nearest age brackets
    return float(np.interp(age_val, ages, exs))

def calculate_life_expectancy_estimate(
    age: float,
    sex: Union[int, str],
    heart_age: float
) -> Dict[str, float]:
    """
    Computes life expectancy estimate based on WHO Uzbekistan mortality tables:
    life_baseline = age + ex(age, sex)
    life_estimate = age + ex(heart_age, sex)
    delta = life_estimate - life_baseline
    """
    age_val = float(age)
    baseline_remaining = get_who_remaining_life(age_val, sex)
    adjusted_remaining = get_who_remaining_life(heart_age, sex)

    life_baseline = age_val + baseline_remaining
    life_estimate = age_val + adjusted_remaining
    delta = life_estimate - life_baseline

    return {
        "life_baseline": round(life_baseline, 1),
        "life_estimate": round(life_estimate, 1),
        "delta": round(delta, 1),
        "baseline_remaining": round(baseline_remaining, 1),
        "adjusted_remaining": round(adjusted_remaining, 1),
    }

def calculate_uzbekistan_life_expectancy(
    age: float,
    sex: Union[int, str],
    heart_age: float
) -> float:
    """Convenience helper returning life_estimate scalar."""
    return calculate_life_expectancy_estimate(age, sex, heart_age)["life_estimate"]

