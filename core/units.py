"""
Unit conversion utilities for laboratory analytes.
Cholesterol: 1 mmol/L = 38.67 mg/dL
Glucose: 1 mmol/L = 18.016 mg/dL
"""

MGDL_PER_MMOL_CHOL = 38.67
MGDL_PER_MMOL_GLUC = 18.016

def to_mgdl(value: float, analyte: str, unit: str) -> float:
    """
    Converts a value from the specified unit ('mmol/l' or 'mg/dl') to mg/dL.
    Analyte can be 'cholesterol', 'hdl', 'totchol', or 'glucose'.
    """
    if value is None:
        return 0.0
    u = str(unit).strip().lower()
    if u in ("mg/dl", "mgdl", "mg"):
        return float(value)
    
    a = str(analyte).strip().lower()
    if a in ("cholesterol", "totchol", "hdl"):
        return float(value) * MGDL_PER_MMOL_CHOL
    elif a in ("glucose", "gluc"):
        return float(value) * MGDL_PER_MMOL_GLUC
    else:
        raise ValueError(f"Unknown analyte: {analyte}")

def from_mgdl(value: float, analyte: str, unit: str) -> float:
    """
    Converts a value in mg/dL to the specified unit ('mmol/l' or 'mg/dl').
    """
    if value is None:
        return 0.0
    u = str(unit).strip().lower()
    if u in ("mg/dl", "mgdl", "mg"):
        return float(value)
    
    a = str(analyte).strip().lower()
    if a in ("cholesterol", "totchol", "hdl"):
        return float(value) / MGDL_PER_MMOL_CHOL
    elif a in ("glucose", "gluc"):
        return float(value) / MGDL_PER_MMOL_GLUC
    else:
        raise ValueError(f"Unknown analyte: {analyte}")

def format_value(value: float, analyte: str, unit: str) -> str:
    """
    Formats value with clinical decimal rounding:
    mmol/L -> 1 decimal place
    mg/dL -> integer
    """
    u = str(unit).strip().lower()
    if u in ("mmol/l", "mmoll", "mmol"):
        val = from_mgdl(value, analyte, unit) if value > 30 else float(value)
        return f"{val:.1f}"
    else:
        val = to_mgdl(value, analyte, unit) if value < 30 else float(value)
        return f"{int(round(val))}"
