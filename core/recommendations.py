"""
core/recommendations.py — Clinical guidelines-driven rules linked to AI risk simulation
and personalized prioritization.
"""
from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional

from core.risk_model import load_risk_model, simulate_intervention

logger = logging.getLogger(__name__)


@dataclass
class ClinicalIntervention:
    id: str
    title_key: str
    category: str
    source: str
    rationale_key: str
    action_key: str
    changes: Dict[str, Any]
    delta_chd_pp: float = 0.0
    delta_heart_age: float = 0.0
    delta_life_exp: float = 0.0
    delta_diabetes_pp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_candidate_rules(patient: Dict[str, Any]) -> List[ClinicalIntervention]:
    """
    Identifies applicable clinical interventions according to official guideline thresholds:
    - ESC/ESH 2018, WHO 2021 (Hypertension)
    - WHO Framework Convention on Tobacco Control (Smoking)
    - NCEP ATP III, ESC 2019 (Lipids)
    - WHO 2020 (Overweight/Obesity)
    - ADA Standards of Care 2024 / WHO (Prediabetes/Hyperglycemia)
    """
    candidates = []

    # 1. Tobacco Cessation
    is_smoker = int(patient.get("currentSmoker", 0)) == 1 or float(patient.get("cigsPerDay", 0.0)) > 0
    if is_smoker:
        candidates.append(ClinicalIntervention(
            id="smoking_cessation",
            title_key="rec_smoking_title",
            category="lifestyle",
            source="WHO / ESC 2021 Prevention Guidelines",
            rationale_key="rec_smoking_rationale",
            action_key="rec_smoking_action",
            changes={"currentSmoker": 0, "cigsPerDay": 0.0},
        ))

    # 2. Blood Pressure Control
    sbp = float(patient.get("sysBP", 120.0))
    dbp = float(patient.get("diaBP", 80.0))
    bp_meds = int(patient.get("BPMeds", 0)) == 1
    if sbp >= 135.0 or dbp >= 85.0 or bp_meds:
        target_sbp = min(sbp, 125.0)
        target_dbp = min(dbp, 78.0)
        candidates.append(ClinicalIntervention(
            id="blood_pressure_control",
            title_key="rec_bp_title",
            category="medical",
            source="ESC/ESH 2018 / WHO 2021 Hypertension Guidelines",
            rationale_key="rec_bp_rationale",
            action_key="rec_bp_action",
            changes={"sysBP": target_sbp, "diaBP": target_dbp, "prevalentHyp": 0},
        ))

    # 3. Lipid Profile Optimization
    tot_chol = float(patient.get("totChol", 200.0))
    if tot_chol >= 200.0:  # >= 5.2 mmol/L
        target_chol = max(160.0, round(tot_chol * 0.80, 1))  # 20% reduction target
        candidates.append(ClinicalIntervention(
            id="lipid_management",
            title_key="rec_lipids_title",
            category="lifestyle_medical",
            source="NCEP ATP III / ESC 2019 Dyslipidemia Guidelines",
            rationale_key="rec_lipids_rationale",
            action_key="rec_lipids_action",
            changes={"totChol": target_chol},
        ))

    # 4. Weight Management
    bmi = float(patient.get("BMI", 25.0))
    if bmi >= 25.0:
        target_bmi = min(bmi, 24.5)
        candidates.append(ClinicalIntervention(
            id="weight_reduction",
            title_key="rec_weight_title",
            category="lifestyle",
            source="WHO 2020 Healthy Weight Standards",
            rationale_key="rec_weight_rationale",
            action_key="rec_weight_action",
            changes={"BMI": target_bmi},
        ))

    # 5. Glycemic Control
    glucose = float(patient.get("glucose", 90.0))
    has_dm = int(patient.get("diabetes", 0)) == 1
    if glucose >= 100.0 or has_dm:  # >= 5.6 mmol/L impaired fasting glucose
        target_glucose = min(glucose, 95.0)
        candidates.append(ClinicalIntervention(
            id="glycemic_control",
            title_key="rec_glucose_title",
            category="medical",
            source="ADA Standards of Care 2024 / WHO Guidelines",
            rationale_key="rec_glucose_rationale",
            action_key="rec_glucose_action",
            changes={"glucose": target_glucose},
        ))

    return candidates


def generate_recommendations(
    patient: Dict[str, Any],
    model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates guideline rules, runs individual simulations through the neural network
    and clinical longevity formulas, and prioritizes interventions by absolute CHD risk reduction.
    """
    if model is None:
        model = load_risk_model()

    candidates = get_candidate_rules(patient)

    if not candidates:
        # Healthy profile: provide maintenance guidance
        return {
            "has_interventions": False,
            "top_interventions": [],
            "all_interventions": [],
            "combined_impact": None,
            "maintenance_key": "rec_maintenance_optimal",
        }

    # Simulate each candidate intervention
    all_interventions: List[ClinicalIntervention] = []
    combined_changes: Dict[str, Any] = {}

    for cand in candidates:
        sim_res = simulate_intervention(patient, cand.changes, model=model)
        deltas = sim_res["deltas"]

        cand.delta_chd_pp = deltas["delta_chd_pp"]
        cand.delta_heart_age = deltas["delta_heart_age"]
        cand.delta_life_exp = deltas["delta_life_exp"]
        cand.delta_diabetes_pp = deltas["delta_diabetes_pp"]

        all_interventions.append(cand)
        combined_changes.update(cand.changes)

    # Sort interventions: largest CHD reduction first (most negative delta_chd_pp)
    # Secondary sort: largest heart age reduction
    all_interventions.sort(key=lambda x: (x.delta_chd_pp, x.delta_heart_age))

    top_interventions = all_interventions[:3]

    # Simulate combined impact of all triggered interventions
    combined_sim = simulate_intervention(patient, combined_changes, model=model)

    return {
        "has_interventions": True,
        "top_interventions": [item.to_dict() for item in top_interventions],
        "all_interventions": [item.to_dict() for item in all_interventions],
        "combined_impact": {
            "changes": combined_changes,
            "deltas": combined_sim["deltas"],
            "simulated": combined_sim["simulated"],
        },
        "maintenance_key": None,
    }
