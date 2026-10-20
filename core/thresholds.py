"""
Clinical guideline thresholds and biomarker classification.
Sources:
- Blood Pressure: ESC/ESH 2018; WHO 2021
- Lipids: NCEP ATP III
- Glucose: WHO / ADA
- BMI: WHO
- Heart Rate: Clinical consensus
"""
from typing import Dict, Any, Optional

def classify(
    marker: str,
    value: float,
    sex: str = "male",
    secondary_value: Optional[float] = None,
    lang: str = "ru"
) -> Dict[str, Any]:
    """
    Classifies a biomarker value according to official guidelines.
    Returns:
        level: 'ok' | 'warn' | 'high'
        label_key: i18n translation key
        source: citation of clinical guideline
        normal_range: human readable optimal range string localized to lang
    """
    m = marker.strip().lower()
    is_male = (str(sex).strip().lower() in ("1", "male", "m", "erkak", "мужчина", "мужской"))
    lang_code = lang.lower().strip() if lang else "ru"

    if m in ("bp", "blood_pressure", "sysbp"):
        # value = SBP, secondary_value = DBP
        sbp = float(value)
        dbp = float(secondary_value) if secondary_value is not None else 80.0

        if lang_code == "en":
            range_130 = "< 130 / < 85 mmHg"
            range_120 = "< 120 / < 80 mmHg"
        elif lang_code == "uz":
            range_130 = "< 130 / < 85 mm sim.ust."
            range_120 = "< 120 / < 80 mm sim.ust."
        else:
            range_130 = "< 130 / < 85 мм рт. ст."
            range_120 = "< 120 / < 80 мм рт. ст."
        
        if sbp >= 140.0 or dbp >= 90.0:
            return {
                "level": "high",
                "label_key": "thresholds.bp.hypertension",
                "source": "ESC/ESH 2018; WHO 2021",
                "normal_range": range_130
            }
        elif sbp >= 130.0 or dbp >= 85.0:
            return {
                "level": "warn",
                "label_key": "thresholds.bp.high_normal",
                "source": "ESC/ESH 2018",
                "normal_range": range_130
            }
        elif sbp >= 120.0 or dbp >= 80.0:
            return {
                "level": "ok",
                "label_key": "thresholds.bp.normal",
                "source": "ESC/ESH 2018",
                "normal_range": range_130
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.bp.optimal",
                "source": "ESC/ESH 2018",
                "normal_range": range_120
            }

    elif m in ("totchol", "cholesterol"):
        # value in mg/dL
        val = float(value)
        if lang_code == "en":
            chol_norm = "< 200 mg/dL (< 5.2 mmol/L)"
        elif lang_code == "uz":
            chol_norm = "< 200 mg/dl (< 5.2 mmol/l)"
        else:
            chol_norm = "< 200 мг/дл (< 5.2 ммоль/л)"

        if val >= 240.0:
            return {
                "level": "high",
                "label_key": "thresholds.chol.high",
                "source": "NCEP ATP III",
                "normal_range": chol_norm
            }
        elif val >= 200.0:
            return {
                "level": "warn",
                "label_key": "thresholds.chol.borderline",
                "source": "NCEP ATP III",
                "normal_range": chol_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.chol.desirable",
                "source": "NCEP ATP III",
                "normal_range": chol_norm
            }

    elif m == "hdl":
        # value in mg/dL
        val = float(value)
        cutoff = 40.0 if is_male else 50.0
        if lang_code == "en":
            hdl_norm = "≥ 40 mg/dL (M) / ≥ 50 mg/dL (F)"
        elif lang_code == "uz":
            hdl_norm = "≥ 40 mg/dl (Erk.) / ≥ 50 mg/dl (Ayol)"
        else:
            hdl_norm = "≥ 40 мг/дл (муж.) / ≥ 50 мг/дл (жен.)"

        if val < cutoff:
            return {
                "level": "warn",
                "label_key": "thresholds.hdl.low",
                "source": "NCEP ATP III",
                "normal_range": hdl_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.hdl.normal",
                "source": "NCEP ATP III",
                "normal_range": hdl_norm
            }

    elif m in ("glucose", "fasting_glucose"):
        # value in mg/dL
        val = float(value)
        if lang_code == "en":
            glu_norm = "< 110 mg/dL (< 6.1 mmol/L)"
        elif lang_code == "uz":
            glu_norm = "< 110 mg/dl (< 6.1 mmol/l)"
        else:
            glu_norm = "< 110 мг/дл (< 6.1 ммоль/л)"

        if val >= 126.0:
            return {
                "level": "high",
                "label_key": "thresholds.glucose.diabetes_suspect",
                "source": "WHO / ADA Guidelines",
                "normal_range": glu_norm
            }
        elif val >= 110.0:
            return {
                "level": "warn",
                "label_key": "thresholds.glucose.impaired_fasting",
                "source": "WHO Guidelines",
                "normal_range": glu_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.glucose.normal",
                "source": "WHO Guidelines",
                "normal_range": glu_norm
            }

    elif m in ("bmi", "body_mass_index"):
        val = float(value)
        bmi_norm = "18.5 – 24.9 kg/m²" if lang_code in ("en", "uz") else "18.5 – 24.9 кг/м²"

        if val >= 30.0:
            return {
                "level": "high",
                "label_key": "thresholds.bmi.obesity",
                "source": "WHO Classification",
                "normal_range": bmi_norm
            }
        elif val >= 25.0:
            return {
                "level": "warn",
                "label_key": "thresholds.bmi.overweight",
                "source": "WHO Classification",
                "normal_range": bmi_norm
            }
        elif val < 18.5:
            return {
                "level": "warn",
                "label_key": "thresholds.bmi.underweight",
                "source": "WHO Classification",
                "normal_range": bmi_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.bmi.normal",
                "source": "WHO Classification",
                "normal_range": bmi_norm
            }

    elif m in ("heartrate", "resting_heart_rate"):
        val = float(value)
        if lang_code == "en":
            hr_norm = "60 – 100 bpm"
        elif lang_code == "uz":
            hr_norm = "60 – 100 zarba/daq"
        else:
            hr_norm = "60 – 100 уд/мин"

        if val < 60.0:
            return {
                "level": "warn",
                "label_key": "thresholds.hr.bradycardia",
                "source": "Clinical Consensus",
                "normal_range": hr_norm
            }
        elif val > 100.0:
            return {
                "level": "warn",
                "label_key": "thresholds.hr.tachycardia",
                "source": "Clinical Consensus",
                "normal_range": hr_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.hr.normal",
                "source": "Clinical Consensus",
                "normal_range": hr_norm
            }

    elif m in ("smoking", "currentsmoker"):
        val = float(value)
        if lang_code == "en":
            smoke_norm = "Non-smoker (0 cigs)"
        elif lang_code == "uz":
            smoke_norm = "Chekmaydi (0 sigaret)"
        else:
            smoke_norm = "Не курит (0 сигарет)"

        if val >= 0.5:
            return {
                "level": "high",
                "label_key": "thresholds.smoking.smoker",
                "source": "WHO Tobacco Fact Sheet",
                "normal_range": smoke_norm
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.smoking.non_smoker",
                "source": "WHO Tobacco Fact Sheet",
                "normal_range": smoke_norm
            }

    else:
        raise ValueError(f"Unknown biomarker for classification: {marker}")
