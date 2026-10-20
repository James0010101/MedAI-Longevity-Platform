"""
core/report.py — Clinical report generation (Printable HTML) and HL7 FHIR R4 Bundle export.
"""
from datetime import datetime
import json
from typing import Any, Dict
import uuid

from core.i18n import t
from core.units import format_value, from_mgdl


def generate_html_report(
    patient: Dict[str, Any],
    prediction: Dict[str, Any],
    recommendations: Dict[str, Any],
    lang: str = "ru",
    units: str = "mmol"
) -> str:
    """
    Generates a standalone, self-contained printable HTML clinical report.
    Compatible with browser print-to-PDF (@media print).
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_id = str(uuid.uuid4())[:8].upper()

    age = int(patient.get("age", 50))
    sex_label = t("sex_male", lang=lang) if int(patient.get("male", 1)) == 1 else t("sex_female", lang=lang)
    sbp = int(patient.get("sysBP", 120))
    dbp = int(patient.get("diaBP", 80))
    bmi = round(float(patient.get("BMI", 25.0)), 1)
    hr = int(patient.get("heartRate", 72))
    smoker = t("smoker_yes", lang=lang) if int(patient.get("currentSmoker", 0)) == 1 else t("smoker_no", lang=lang)
    bp_meds = t("meds_yes", lang=lang) if int(patient.get("BPMeds", 0)) == 1 else t("meds_no", lang=lang)

    # Lipid & Glucose formatted
    raw_tc = float(patient.get("totChol", 200.0))
    raw_glu = float(patient.get("glucose", 90.0))
    val_tc = from_mgdl(raw_tc, "totChol", units)
    val_glu = from_mgdl(raw_glu, "glucose", units)
    unit_label = "ммоль/л" if units == "mmol" else "mg/dL"

    # Prediction metrics
    p_chd = prediction.get("p_chd", 0.0) * 100.0
    p_chd_std = prediction.get("p_chd_std", 0.0) * 100.0
    heart_age = prediction.get("heart_age", float(age))
    ha_delta = prediction.get("heart_age_delta", 0.0)
    life_exp = prediction.get("life_expectancy", 74.0)
    p_dm = prediction.get("p_diabetes", 0.0) * 100.0

    # Risk category
    if p_chd < 10.0:
        chd_cat = t("card_chd_cat_low", lang=lang)
        chd_badge_color = "#006c49"
    elif p_chd < 20.0:
        chd_cat = t("card_chd_cat_mod", lang=lang)
        chd_badge_color = "#b25e00"
    else:
        chd_cat = t("card_chd_cat_high", lang=lang)
        chd_badge_color = "#ba1a1a"

    # Top recommendations HTML
    top_items = recommendations.get("top_interventions", [])
    recs_rows_html = ""
    for idx, item in enumerate(top_items, 1):
        recs_rows_html += f"""
        <div style="margin-bottom: 12px; padding: 10px; border-left: 4px solid #006c49; background: #f9fbf9;">
            <div style="font-weight: bold; color: #006c49; font-size: 14px;">{idx}. {t(item['title_key'], lang=lang)}</div>
            <div style="font-size: 12px; color: #555; margin: 4px 0;"><strong>Источник:</strong> {item['source']}</div>
            <div style="font-size: 13px; color: #222; margin: 4px 0;">{t(item['rationale_key'], lang=lang)}</div>
            <div style="font-size: 12px; color: #006c49; font-weight: bold; margin-top: 4px;">
                Ожидаемый эффект: {t('rec_chd_reduction', lang=lang, val=abs(item['delta_chd_pp']))} | 
                {t('rec_heart_age_rejuvenation', lang=lang, val=abs(item['delta_heart_age']))} | 
                {t('rec_life_gain', lang=lang, val=abs(item['delta_life_exp']))}
            </div>
            <div style="font-size: 12px; color: #333; margin-top: 4px; font-style: italic;">{t(item['action_key'], lang=lang)}</div>
        </div>
        """

    if not recs_rows_html:
        recs_rows_html = f"<p style='color: #006c49;'>{t('rec_maintenance_optimal', lang=lang)}</p>"

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<title>MedAI Longevity Platform — Клинический отчёт #{report_id}</title>
<style>
    @page {{
        size: A4;
        margin: 15mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        color: #1a1a1a;
        line-height: 1.4;
        font-size: 13px;
        margin: 0;
        padding: 0;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #006c49;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }}
    .title {{
        font-size: 20px;
        font-weight: bold;
        color: #006c49;
    }}
    .meta {{
        font-size: 11px;
        color: #666;
        text-align: right;
    }}
    .section-title {{
        font-size: 14px;
        font-weight: bold;
        color: #006c49;
        border-bottom: 1px solid #ddd;
        padding-bottom: 4px;
        margin-top: 15px;
        margin-bottom: 10px;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }}
    .grid-4 {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }}
    .card {{
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        background: #ffffff;
    }}
    .card-metric {{
        font-size: 20px;
        font-weight: bold;
        color: #006c49;
        margin: 4px 0;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        color: #ffffff;
        background: {chd_badge_color};
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    th, td {{
        padding: 6px 8px;
        border-bottom: 1px solid #eee;
        text-align: left;
    }}
    th {{
        background: #f4fbf4;
        color: #006c49;
    }}
    .disclaimer {{
        margin-top: 25px;
        padding: 10px;
        background: #fdf6ec;
        border: 1px solid #faecd8;
        border-radius: 4px;
        font-size: 11px;
        color: #7a5216;
    }}
    @media print {{
        body {{
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}
        .no-break {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>

<div class="header">
    <div>
        <div class="title">MedAI Longevity Platform</div>
        <div style="font-size: 12px; color: #555;">{t('app_subtitle', lang=lang)}</div>
    </div>
    <div class="meta">
        <div><strong>Отчёт №:</strong> {report_id}</div>
        <div><strong>Дата:</strong> {now_str}</div>
        <div><strong>Версия:</strong> 2.0.0 (Research Prototype)</div>
    </div>
</div>

<div class="section-title">1. Паспорт пациента и исходные биомаркеры</div>
<table>
    <tr>
        <th>Показатель</th><th>Значение</th>
        <th>Показатель</th><th>Значение</th>
    </tr>
    <tr>
        <td><strong>Возраст</strong></td><td>{age} лет</td>
        <td><strong>Артериальное давление</strong></td><td>{sbp}/{dbp} мм рт.ст.</td>
    </tr>
    <tr>
        <td><strong>Биологический пол</strong></td><td>{sex_label}</td>
        <td><strong>Гипотензивная терапия</strong></td><td>{bp_meds}</td>
    </tr>
    <tr>
        <td><strong>Статус курения</strong></td><td>{smoker}</td>
        <td><strong>Индекс массы тела (ИМТ)</strong></td><td>{bmi} кг/м²</td>
    </tr>
    <tr>
        <td><strong>Общий холестерин</strong></td><td>{val_tc:.2f} {unit_label}</td>
        <td><strong>Пульс в покое</strong></td><td>{hr} уд/мин</td>
    </tr>
    <tr>
        <td><strong>Глюкоза натощак</strong></td><td>{val_glu:.2f} {unit_label}</td>
        <td><strong>Ранее диабет</strong></td><td>{"Да" if patient.get("diabetes") else "Нет"}</td>
    </tr>
</table>

<div class="section-title">2. Результаты скрининга сердечно-сосудистого риска и долголетия</div>
<div class="grid-4 no-break">
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_chd_title', lang=lang)}</div>
        <div class="card-metric">{p_chd:.1f}%</div>
        <span class="badge">{chd_cat}</span>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">Неопределённость: ±{p_chd_std:.1f}%</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_heart_age_title', lang=lang)}</div>
        <div class="card-metric">{heart_age:.1f} лет</div>
        <div style="font-size: 11px; color: {'#ba1a1a' if ha_delta > 0 else '#006c49'}; font-weight: bold;">
            {'+' if ha_delta > 0 else ''}{ha_delta:.1f} к паспортному
        </div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">D'Agostino Circulation 2008</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_life_exp_title', lang=lang)}</div>
        <div class="card-metric">{life_exp:.1f} лет</div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">Таблицы ВОЗ для Узбекистана</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_dm_title', lang=lang)}</div>
        <div class="card-metric">{p_dm:.1f}%</div>
        <div style="font-size: 11px; color: {'#ba1a1a' if p_dm >= 50 else '#006c49'}; font-weight: bold;">
            {'Скрининг-сигнал' if p_dm >= 50 else 'Норма'}
        </div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">Текущий статус (Нейросеть)</div>
    </div>
</div>

<div class="section-title">3. Персонализированный план снижения риска (ранжирован нейросетью)</div>
<div class="no-break">
    {recs_rows_html}
</div>

<div class="disclaimer no-break">
    <strong>{t('meth_disclaimer_title', lang=lang)}:</strong> {t('meth_disclaimer_body', lang=lang)}
</div>

</body>
</html>
"""
    return html


def generate_fhir_bundle(
    patient: Dict[str, Any],
    prediction: Dict[str, Any],
    lang: str = "ru"
) -> Dict[str, Any]:
    """
    Exports screening results in standard HL7 FHIR R4 Bundle format (JSON).
    Valid verified LOINC codes are utilized for each clinical observation.
    """
    bundle_id = str(uuid.uuid4())
    patient_id = f"patient-{str(uuid.uuid4())[:8]}"
    now_iso = datetime.now().isoformat()

    p_chd = prediction.get("p_chd", 0.0) * 100.0
    heart_age = prediction.get("heart_age", 50.0)
    life_exp = prediction.get("life_expectancy", 74.0)

    sbp = float(patient.get("sysBP", 120.0))
    dbp = float(patient.get("diaBP", 80.0))
    tot_chol = float(patient.get("totChol", 200.0))
    glucose = float(patient.get("glucose", 90.0))
    bmi = float(patient.get("BMI", 25.0))
    hr = float(patient.get("heartRate", 72.0))

    entries = []

    # 1. Patient Resource
    entries.append({
        "fullUrl": f"urn:uuid:{patient_id}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "gender": "male" if int(patient.get("male", 1)) == 1 else "female",
            "birthDate": f"{datetime.now().year - int(patient.get('age', 50))}-01-01",
        }
    })

    # Observations helper
    def add_obs(loinc_code: str, display_name: str, value: float, unit: str):
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"urn:uuid:{patient_id}"},
                "effectiveDateTime": now_iso,
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc_code,
                        "display": display_name,
                    }]
                },
                "valueQuantity": {
                    "value": round(value, 2),
                    "unit": unit,
                    "system": "http://unitsofmeasure.org",
                }
            }
        })

    # Verified LOINC observations
    add_obs("8480-6", "Systolic blood pressure", sbp, "mm[Hg]")
    add_obs("8462-4", "Diastolic blood pressure", dbp, "mm[Hg]")
    add_obs("2093-3", "Cholesterol in Serum or Plasma", tot_chol, "mg/dL")
    add_obs("1558-6", "Fasting glucose in Serum or Plasma", glucose, "mg/dL")
    add_obs("39156-5", "Body mass index (BMI)", bmi, "kg/m2")
    add_obs("8867-4", "Heart rate", hr, "/min")

    # RiskAssessment Resource for 10-Year CHD Risk
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": f"urn:uuid:{patient_id}"},
            "occurrenceDateTime": now_iso,
            "method": {
                "coding": [{
                    "system": "http://medai.uz/algorithms",
                    "code": "framingham-nn-v2",
                    "display": "MedAI MultiTask Neural Network 10-Year CHD Risk",
                }]
            },
            "prediction": [{
                "outcome": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "53741008",
                        "display": "Coronary heart disease",
                    }]
                },
                "probabilityDecimal": round(prediction.get("p_chd", 0.0), 4),
                "qualitativeRisk": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                        "code": "low" if p_chd < 10 else ("moderate" if p_chd < 20 else "high"),
                    }]
                }
            }],
            "note": [{
                "text": f"Estimated Heart Age: {heart_age:.1f} years. Estimated Life Expectancy (Uzbekistan WHO tables): {life_exp:.1f} years."
            }]
        }
    })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": now_iso,
        "entry": entries,
    }
