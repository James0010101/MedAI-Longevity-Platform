"""
core/risk_model.py — Risk calculation, uncertainty, Integrated Gradients XAI,
and intervention simulation engine.
"""
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.clinical import (
    calculate_dagostino_risk,
    calculate_framingham_heart_age,
    calculate_uzbekistan_life_expectancy,
)
from core.config import FEATURE_NAMES, MODELS_DIR
from neural_network import MultiTaskNeuralNetwork

logger = logging.getLogger(__name__)

# Default weights path
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "model_weights.npz"

# Cache resource wrapper for Streamlit if available
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func


@cache_decorator
def load_risk_model(weights_path: Optional[Union[str, Path]] = None) -> MultiTaskNeuralNetwork:
    """
    Loads and caches the trained MultiTaskNeuralNetwork.
    """
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model weights file not found: {path}")
    logger.info(f"Loading risk model from {path}...")
    model = MultiTaskNeuralNetwork.load(path)
    return model


def patient_dict_to_raw_vector(patient: Dict[str, Any]) -> np.ndarray:
    """
    Converts a patient dictionary into a 1D NumPy array strictly matching FEATURE_NAMES.
    """
    raw_values = [
        float(patient.get("male", 1)),
        float(patient.get("age", 50.0)),
        float(patient.get("currentSmoker", 0)),
        float(patient.get("cigsPerDay", 0.0)),
        float(patient.get("BPMeds", 0)),
        float(patient.get("prevalentStroke", 0)),
        float(patient.get("prevalentHyp", 0)),
        float(patient.get("totChol", 200.0)),
        float(patient.get("sysBP", 120.0)),
        float(patient.get("diaBP", 80.0)),
        float(patient.get("BMI", 25.0)),
        float(patient.get("heartRate", 72.0)),
        float(patient.get("glucose", 90.0)),
    ]
    return np.array(raw_values, dtype=np.float64)


def get_optimal_healthy_profile(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs an age- and sex-matched reference profile with optimal clinical markers.
    Used as the baseline for Integrated Gradients and risk gap comparisons.
    """
    return {
        "male": int(patient.get("male", 1)),
        "age": float(patient.get("age", 50.0)),
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
        "totChol": 180.0,
        "sysBP": 115.0,
        "diaBP": 75.0,
        "BMI": 22.0,
        "heartRate": 70.0,
        "glucose": 85.0,
    }


def normalize_features(
    raw_vector: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Applies z-score standardization using training set statistics.
    """
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (raw_vector - means) / stds_safe


def predict_patient(
    patient: Dict[str, Any],
    model: Optional[MultiTaskNeuralNetwork] = None,
    n_mc: int = 30
) -> Dict[str, Any]:
    """
    Full clinical evaluation:
    - 10-year CHD risk (NN output with MC Dropout uncertainty)
    - Diabetes screening probability (NN output with MC Dropout uncertainty)
    - Heart age (D'Agostino 2008 closed-form formula)
    - Life expectancy (WHO Uzbekistan life table with heart age acceleration)
    - General CVD risk (D'Agostino 2008 benchmark)
    """
    if model is None:
        model = load_risk_model()

    meta = model.metadata or {}
    means = np.array(meta.get("means", [0.0] * len(FEATURE_NAMES)), dtype=np.float64)
    stds = np.array(meta.get("stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float64)

    raw_vec = patient_dict_to_raw_vector(patient)
    norm_vec = normalize_features(raw_vec, means, stds).reshape(1, -1)

    # Deterministic inference
    probs = model.predict(norm_vec)
    p_chd = float(probs["chd"][0, 0])
    p_dm = float(probs["diabetes"][0, 0])

    # Epistemic uncertainty via MC Dropout
    unc = model.predict_with_uncertainty(norm_vec, n_mc=n_mc, seed=42)
    chd_std = float(unc["chd"]["std"][0, 0])
    dm_std = float(unc["diabetes"]["std"][0, 0])

    # Clinical formulas
    age = float(patient.get("age", 50.0))
    sex = int(patient.get("male", 1))
    sbp = float(patient.get("sysBP", 120.0))
    bp_meds = int(patient.get("BPMeds", 0))
    tot_chol = float(patient.get("totChol", 200.0))
    smoker = int(patient.get("currentSmoker", 0))

    # Known diabetes flag or model screened diabetes > 0.5
    diabetes_flag = int(patient.get("diabetes", 1 if p_dm >= 0.5 else 0))

    heart_age = calculate_framingham_heart_age(
        age=age,
        sex=sex,
        sbp=sbp,
        bp_meds=bp_meds,
        tot_chol=tot_chol,
        hdl=45.0,
        smoker=smoker,
        diabetes=diabetes_flag,
    )

    life_exp = calculate_uzbekistan_life_expectancy(
        age=age,
        sex=sex,
        heart_age=heart_age,
    )

    dagostino_cvd = calculate_dagostino_risk(
        age=age,
        sex=sex,
        sbp=sbp,
        bp_meds=bp_meds,
        tot_chol=tot_chol,
        hdl=45.0,
        smoker=smoker,
        diabetes=diabetes_flag,
    )

    return {
        "p_chd": round(p_chd, 4),
        "p_chd_std": round(chd_std, 4),
        "p_diabetes": round(p_dm, 4),
        "p_diabetes_std": round(dm_std, 4),
        "heart_age": round(heart_age, 1),
        "heart_age_delta": round(heart_age - age, 1),
        "life_expectancy": round(life_exp, 1),
        "dagostino_cvd_risk": round(dagostino_cvd, 4),
        "age": age,
        "sex": sex,
    }


def explain_prediction(
    patient: Dict[str, Any],
    head: str = "chd",
    m_steps: int = 50,
    model: Optional[MultiTaskNeuralNetwork] = None
) -> Dict[str, Any]:
    """
    Computes feature attributions via Integrated Gradients starting from an
    age- and sex-matched healthy baseline profile.
    Attributions are scaled to percentage points (pp).
    """
    if model is None:
        model = load_risk_model()

    meta = model.metadata or {}
    means = np.array(meta.get("means", [0.0] * len(FEATURE_NAMES)), dtype=np.float64)
    stds = np.array(meta.get("stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float64)

    raw_patient = patient_dict_to_raw_vector(patient)
    norm_patient = normalize_features(raw_patient, means, stds)

    healthy_ref = get_optimal_healthy_profile(patient)
    raw_base = patient_dict_to_raw_vector(healthy_ref)
    norm_base = normalize_features(raw_base, means, stds)

    # Integrated Gradients
    raw_attributions = model.integrated_gradients(
        norm_patient,
        norm_base,
        head=head,
        m_steps=m_steps,
    )

    p_patient = float(model.predict(norm_patient.reshape(1, -1))[head][0, 0])
    p_base = float(model.predict(norm_base.reshape(1, -1))[head][0, 0])

    attributions_pp = {}
    for i, name in enumerate(FEATURE_NAMES):
        attributions_pp[name] = round(float(raw_attributions[i] * 100.0), 2)

    sum_pp = round(float(np.sum(raw_attributions) * 100.0), 2)
    diff_pp = round(float((p_patient - p_base) * 100.0), 2)

    return {
        "head": head,
        "attributions_pp": attributions_pp,
        "patient_prob": round(p_patient, 4),
        "baseline_prob": round(p_base, 4),
        "sum_attributions_pp": sum_pp,
        "target_diff_pp": diff_pp,
    }


# Correlated inputs describing one clinical factor are shown as a single bar.
# Integrated Gradients is additive, so summing their attributions is exact.
ATTRIBUTION_GROUPS: Dict[str, tuple] = {"smoking": ("currentSmoker", "cigsPerDay")}


def group_attributions(attributions_pp: Dict[str, float]) -> Dict[str, float]:
    """Merge correlated features (smoker flag + cigarettes/day) into one factor."""
    grouped = dict(attributions_pp)
    for group, members in ATTRIBUTION_GROUPS.items():
        present = [m for m in members if m in grouped]
        if present:
            grouped[group] = round(sum(grouped.pop(m) for m in present), 2)
    return grouped


def simulate_intervention(
    patient: Dict[str, Any],
    changes: Dict[str, Any],
    model: Optional[MultiTaskNeuralNetwork] = None
) -> Dict[str, Any]:
    """
    Simulates the quantitative impact of clinical interventions on CHD risk,
    heart age, life expectancy, and diabetes screening probability.
    """
    if model is None:
        model = load_risk_model()

    base_eval = predict_patient(patient, model=model)

    sim_patient = deepcopy(patient)
    sim_patient.update(changes)

    # If currentSmoker set to 0, automatically zero out cigsPerDay unless specified
    if changes.get("currentSmoker") == 0 and "cigsPerDay" not in changes:
        sim_patient["cigsPerDay"] = 0.0

    sim_eval = predict_patient(sim_patient, model=model)

    delta_chd_pp = round((sim_eval["p_chd"] - base_eval["p_chd"]) * 100.0, 2)
    delta_heart_age = round(sim_eval["heart_age"] - base_eval["heart_age"], 1)
    delta_life_exp = round(sim_eval["life_expectancy"] - base_eval["life_expectancy"], 1)
    delta_dm_pp = round((sim_eval["p_diabetes"] - base_eval["p_diabetes"]) * 100.0, 2)

    return {
        "base": base_eval,
        "simulated": sim_eval,
        "deltas": {
            "delta_chd_pp": delta_chd_pp,
            "delta_heart_age": delta_heart_age,
            "delta_life_exp": delta_life_exp,
            "delta_diabetes_pp": delta_dm_pp,
        },
    }


def compute_risk_trajectory(
    patient: Dict[str, Any],
    years_ahead: Tuple[int, ...] = (0, 5, 10, 15, 20),
    improved_changes: Optional[Dict[str, Any]] = None,
    model: Optional[MultiTaskNeuralNetwork] = None
) -> List[Dict[str, Any]]:
    """
    Computes risk trajectories across future years under:
    1) Current trajectory (aging naturally with current lifestyle and risk factors)
    2) Optimized trajectory (with beneficial interventions applied)
    """
    if model is None:
        model = load_risk_model()

    base_age = float(patient.get("age", 50.0))
    trajectory = []

    for y in years_ahead:
        sim_age = min(85.0, base_age + y)

        # 1. Current scenario
        curr_p = deepcopy(patient)
        curr_p["age"] = sim_age
        # Physiological drift: untreated SBP increases ~0.4 mmHg per year of aging
        if not curr_p.get("BPMeds", 0):
            curr_p["sysBP"] = min(220.0, float(patient.get("sysBP", 120.0)) + 0.4 * y)

        res_curr = predict_patient(curr_p, model=model)
        trajectory.append({
            "years_ahead": y,
            "age": sim_age,
            "scenario": "current",
            "p_chd_percent": round(res_curr["p_chd"] * 100.0, 1),
            "heart_age": res_curr["heart_age"],
            "life_expectancy": res_curr["life_expectancy"],
        })

        # 2. Optimized scenario (if changes provided)
        if improved_changes:
            opt_p = deepcopy(patient)
            opt_p.update(improved_changes)
            opt_p["age"] = sim_age
            if opt_p.get("currentSmoker") == 0:
                opt_p["cigsPerDay"] = 0.0

            res_opt = predict_patient(opt_p, model=model)
            trajectory.append({
                "years_ahead": y,
                "age": sim_age,
                "scenario": "improved",
                "p_chd_percent": round(res_opt["p_chd"] * 100.0, 1),
                "heart_age": res_opt["heart_age"],
                "life_expectancy": res_opt["life_expectancy"],
            })

    return trajectory
