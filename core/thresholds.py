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
    secondary_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Classifies a biomarker value according to official guidelines.
    Returns:
        level: 'ok' | 'warn' | 'high'
        label_key: i18n translation key
        source: citation of clinical guideline
        normal_range: human readable optimal range string
    """
    m = marker.strip().lower()
    is_male = (str(sex).strip().lower() in ("1", "male", "m", "мужчина", "мужской"))

    if m in ("bp", "blood_pressure", "sysbp"):
        # value = SBP, secondary_value = DBP
        sbp = float(value)
        dbp = float(secondary_value) if secondary_value is not None else 80.0
        
        if sbp >= 140.0 or dbp >= 90.0:
            return {
                "level": "high",
                "label_key": "thresholds.bp.hypertension",
                "source": "ESC/ESH 2018; WHO 2021",
                "normal_range": "< 130 / < 85 мм рт. ст."
            }
        elif sbp >= 130.0 or dbp >= 85.0:
            return {
                "level": "warn",
                "label_key": "thresholds.bp.high_normal",
                "source": "ESC/ESH 2018",
                "normal_range": "< 130 / < 85 мм рт. ст."
            }
        elif sbp >= 120.0 or dbp >= 80.0:
            return {
                "level": "ok",
                "label_key": "thresholds.bp.normal",
                "source": "ESC/ESH 2018",
                "normal_range": "< 130 / < 85 мм рт. ст."
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.bp.optimal",
                "source": "ESC/ESH 2018",
                "normal_range": "< 120 / < 80 мм рт. ст."
            }

    elif m in ("totchol", "cholesterol"):
        # value in mg/dL
        val = float(value)
        if val >= 240.0:
            return {
                "level": "high",
                "label_key": "thresholds.chol.high",
                "source": "NCEP ATP III",
                "normal_range": "< 200 мг/дл (< 5.2 ммоль/л)"
            }
        elif val >= 200.0:
            return {
                "level": "warn",
                "label_key": "thresholds.chol.borderline",
                "source": "NCEP ATP III",
                "normal_range": "< 200 мг/дл (< 5.2 ммоль/л)"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.chol.desirable",
                "source": "NCEP ATP III",
                "normal_range": "< 200 мг/дл (< 5.2 ммоль/л)"
            }

    elif m == "hdl":
        # value in mg/dL
        val = float(value)
        cutoff = 40.0 if is_male else 50.0
        if val < cutoff:
            return {
                "level": "warn",
                "label_key": "thresholds.hdl.low",
                "source": "NCEP ATP III",
                "normal_range": "≥ 40 мг/дл (муж.) / ≥ 50 мг/дл (жен.)"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.hdl.normal",
                "source": "NCEP ATP III",
                "normal_range": "≥ 40 мг/дл (муж.) / ≥ 50 мг/дл (жен.)"
            }

    elif m in ("glucose", "fasting_glucose"):
        # value in mg/dL
        val = float(value)
        if val >= 126.0:
            return {
                "level": "high",
                "label_key": "thresholds.glucose.diabetes_suspect",
                "source": "WHO / ADA Guidelines",
                "normal_range": "< 110 мг/дл (< 6.1 ммоль/л)"
            }
        elif val >= 110.0:
            return {
                "level": "warn",
                "label_key": "thresholds.glucose.impaired_fasting",
                "source": "WHO Guidelines",
                "normal_range": "< 110 мг/дл (< 6.1 ммоль/л)"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.glucose.normal",
                "source": "WHO Guidelines",
                "normal_range": "< 110 мг/дл (< 6.1 ммоль/л)"
            }

    elif m in ("bmi", "body_mass_index"):
        val = float(value)
        if val >= 30.0:
            return {
                "level": "high",
                "label_key": "thresholds.bmi.obesity",
                "source": "WHO Classification",
                "normal_range": "18.5 – 24.9 кг/м²"
            }
        elif val >= 25.0:
            return {
                "level": "warn",
                "label_key": "thresholds.bmi.overweight",
                "source": "WHO Classification",
                "normal_range": "18.5 – 24.9 кг/м²"
            }
        elif val < 18.5:
            return {
                "level": "warn",
                "label_key": "thresholds.bmi.underweight",
                "source": "WHO Classification",
                "normal_range": "18.5 – 24.9 кг/м²"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.bmi.normal",
                "source": "WHO Classification",
                "normal_range": "18.5 – 24.9 кг/м²"
            }

    elif m in ("heartrate", "resting_heart_rate"):
        val = float(value)
        if val < 60.0:
            return {
                "level": "warn",
                "label_key": "thresholds.hr.bradycardia",
                "source": "Clinical Consensus",
                "normal_range": "60 – 100 уд/мин"
            }
        elif val > 100.0:
            return {
                "level": "warn",
                "label_key": "thresholds.hr.tachycardia",
                "source": "Clinical Consensus",
                "normal_range": "60 – 100 уд/мин"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.hr.normal",
                "source": "Clinical Consensus",
                "normal_range": "60 – 100 уд/мин"
            }

    elif m in ("smoking", "currentsmoker"):
        val = float(value)
        if val >= 0.5:
            return {
                "level": "high",
                "label_key": "thresholds.smoking.smoker",
                "source": "WHO Tobacco Fact Sheet",
                "normal_range": "Не курит (0 сигарет)"
            }
        else:
            return {
                "level": "ok",
                "label_key": "thresholds.smoking.non_smoker",
                "source": "WHO Tobacco Fact Sheet",
                "normal_range": "Не курит"
            }

    else:
        raise ValueError(f"Unknown biomarker for classification: {marker}")
