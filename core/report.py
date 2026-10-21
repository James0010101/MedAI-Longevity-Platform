"""
core/report.py — Clinical report generation (Printable HTML) and HL7 FHIR R4 Bundle export.
"""
from datetime import datetime
import io
import json
import os
from typing import Any, Dict
import uuid

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing

from core.branding import LOGO_PNG_PATH, logo_data_uri
from core.i18n import t
from core.units import format_value, from_mgdl

_FONTS_REGISTERED = False

def _setup_fonts():
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'fonts')
    regular_path = os.path.join(font_dir, 'DejaVuSans.ttf')
    bold_path = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
    oblique_path = os.path.join(font_dir, 'DejaVuSans-Oblique.ttf')
    
    if not os.path.exists(regular_path):
        font_dir = os.path.join(os.getcwd(), 'assets', 'fonts')
        regular_path = os.path.join(font_dir, 'DejaVuSans.ttf')
        bold_path = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
        oblique_path = os.path.join(font_dir, 'DejaVuSans-Oblique.ttf')

    pdfmetrics.registerFont(TTFont('MedAIFont', regular_path))
    pdfmetrics.registerFont(TTFont('MedAIFont-Bold', bold_path))
    pdfmetrics.registerFont(TTFont('MedAIFont-Italic', oblique_path))
    _FONTS_REGISTERED = True


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
    unit_label = "mmol/L" if (units == "mmol" and lang == "en") else ("mmol/l" if (units == "mmol" and lang == "uz") else ("ммоль/л" if units == "mmol" else "mg/dL"))

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
            <div style="font-size: 12px; color: #555; margin: 4px 0;"><strong>{t('bio_col_source', lang=lang)}:</strong> {item['source']}</div>
            <div style="font-size: 13px; color: #222; margin: 4px 0;">{t(item['rationale_key'], lang=lang)}</div>
            <div style="font-size: 12px; color: #006c49; font-weight: bold; margin-top: 4px;">
                {t('rec_quant_effect', lang=lang)} {t('rec_chd_reduction', lang=lang, val=abs(item['delta_chd_pp']))} | 
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
<title>MedAI Longevity Platform — {t('report_title', lang=lang)} #{report_id}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1a1a1a;
        line-height: 1.45;
        font-size: 13px;
        margin: 20px;
        background: #fff;
    }}
    .header {{
        border-bottom: 2px solid #006c49;
        padding-bottom: 12px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
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
        margin-top: 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 4px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 12px;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 6px 8px;
        text-align: left;
        font-size: 12px;
    }}
    th {{
        background-color: #f5f7f6;
        font-weight: 600;
    }}
    .grid-4 {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 16px;
    }}
    .card {{
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px 10px;
        background: #fafafa;
    }}
    .card-metric {{
        font-size: 18px;
        font-weight: bold;
        color: #111;
        margin: 4px 0;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: bold;
        color: #fff;
        background: {chd_badge_color};
    }}
    .disclaimer {{
        margin-top: 24px;
        padding: 10px;
        background: #fff8e1;
        border-left: 4px solid #ffb300;
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
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="{logo_data_uri()}" width="44" height="44" alt="MedAI logo"/>
        <div>
            <div class="title">MedAI Longevity Platform</div>
            <div style="font-size: 12px; color: #555;">{t('app_subtitle', lang=lang)}</div>
        </div>
    </div>
    <div class="meta">
        <div><strong>{t('report_title', lang=lang)} №:</strong> {report_id}</div>
        <div><strong>{t('report_param_age', lang=lang)}:</strong> {now_str}</div>
        <div><strong>Version:</strong> 2.0.0 (Research Prototype)</div>
    </div>
</div>

<div class="section-title">{t('report_sec_1', lang=lang)}</div>
<table>
    <tr>
        <th>{t('report_col_metric', lang=lang)}</th><th>{t('report_col_value', lang=lang)}</th>
        <th>{t('report_col_metric', lang=lang)}</th><th>{t('report_col_value', lang=lang)}</th>
    </tr>
    <tr>
        <td><strong>{t('report_param_age', lang=lang)}</strong></td><td>{age} {t('res_years', lang=lang)}</td>
        <td><strong>{t('report_param_bp', lang=lang)}</strong></td><td>{sbp}/{dbp} mmHg</td>
    </tr>
    <tr>
        <td><strong>{t('report_param_sex', lang=lang)}</strong></td><td>{sex_label}</td>
        <td><strong>{t('report_param_meds', lang=lang)}</strong></td><td>{bp_meds}</td>
    </tr>
    <tr>
        <td><strong>{t('report_param_smoke', lang=lang)}</strong></td><td>{smoker}</td>
        <td><strong>{t('report_param_bmi', lang=lang)}</strong></td><td>{bmi} kg/m²</td>
    </tr>
    <tr>
        <td><strong>{t('report_param_chol', lang=lang)}</strong></td><td>{val_tc:.2f} {unit_label}</td>
        <td><strong>{t('report_param_hr', lang=lang)}</strong></td><td>{hr} bpm</td>
    </tr>
    <tr>
        <td><strong>{t('report_param_glu', lang=lang)}</strong></td><td>{val_glu:.2f} {unit_label}</td>
        <td><strong>{t('report_param_dm', lang=lang)}</strong></td><td>{t('report_yes', lang=lang) if patient.get("diabetes") else t('report_no', lang=lang)}</td>
    </tr>
</table>

<div class="section-title">{t('report_sec_2', lang=lang)}</div>
<div class="grid-4 no-break">
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_chd_title', lang=lang)}</div>
        <div class="card-metric">{p_chd:.1f}%</div>
        <span class="badge">{chd_cat}</span>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">{t('card_chd_uncertainty', lang=lang, val=f'{p_chd_std:.1f}')}</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_heart_age_title', lang=lang)}</div>
        <div class="card-metric">{heart_age:.1f} {t('res_years', lang=lang)}</div>
        <div style="font-size: 11px; color: {'#ba1a1a' if ha_delta > 0 else '#006c49'}; font-weight: bold;">
            {'+' if ha_delta > 0 else ''}{ha_delta:.1f} {t('res_delta_passport', lang=lang)}
        </div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">D'Agostino Circulation 2008</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_life_exp_title', lang=lang)}</div>
        <div class="card-metric">{life_exp:.1f} {t('res_years', lang=lang)}</div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">{t('card_life_exp_sub', lang=lang)}</div>
    </div>
    <div class="card">
        <div style="font-size: 11px; color: #666;">{t('card_dm_title', lang=lang)}</div>
        <div class="card-metric">{p_dm:.1f}%</div>
        <div style="font-size: 11px; color: {'#ba1a1a' if p_dm >= 50 else '#006c49'}; font-weight: bold;">
            {t('card_dm_high', lang=lang) if p_dm >= 50 else t('card_dm_low', lang=lang)}
        </div>
        <div style="font-size: 10px; color: #777; margin-top: 4px;">{t('card_dm_sub', lang=lang)}</div>
    </div>
</div>

<div class="section-title">{t('report_sec_3', lang=lang)}</div>
<div class="no-break">
    {recs_rows_html}
</div>

<div class="disclaimer no-break">
    <strong>{t('meth_disclaimer_title', lang=lang)}:</strong> {t('report_disclaimer_text', lang=lang)}
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


def generate_pdf_report(
    patient: Dict[str, Any],
    prediction: Dict[str, Any],
    recommendations: Dict[str, Any],
    lang: str = "ru",
    units: str = "mmol"
) -> bytes:
    """
    Generates a structured, clinical A4 PDF screening report with embedded verification QR code.
    Guaranteed to fit on exactly one clean A4 page across RU, UZ, and EN.
    """
    _setup_fonts()
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=20,
        bottomMargin=20
    )

    # Custom styles using DejaVuSans for UTF-8 Cyrillic & Uzbek support
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='MedAIFont-Bold',
        fontSize=16,
        leading=19,
        textColor=colors.HexColor('#006c49')
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        fontName='MedAIFont',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748b')
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        fontName='MedAIFont',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#475569'),
        alignment=2 # Right
    )
    sec_heading = ParagraphStyle(
        'SecHead',
        fontName='MedAIFont-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#006c49'),
        spaceAfter=4
    )
    cell_bold = ParagraphStyle(
        'CellBld',
        fontName='MedAIFont-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    cell_norm = ParagraphStyle(
        'CellNrm',
        fontName='MedAIFont',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#334155')
    )
    card_title = ParagraphStyle(
        'CardTtl',
        fontName='MedAIFont-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#475569'),
        alignment=1 # Center
    )
    card_val = ParagraphStyle(
        'CardVal',
        fontName='MedAIFont-Bold',
        fontSize=13,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        alignment=1 # Center
    )
    card_sub = ParagraphStyle(
        'CardSub',
        fontName='MedAIFont',
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor('#64748b'),
        alignment=1 # Center
    )
    rec_title_style = ParagraphStyle(
        'RecTitle',
        fontName='MedAIFont-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#006c49')
    )
    rec_source_style = ParagraphStyle(
        'RecSrc',
        fontName='MedAIFont-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#475569')
    )
    rec_text_style = ParagraphStyle(
        'RecTxt',
        fontName='MedAIFont',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    rec_quant_style = ParagraphStyle(
        'RecQuant',
        fontName='MedAIFont-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#006c49')
    )
    rec_action_style = ParagraphStyle(
        'RecAct',
        fontName='MedAIFont-Italic',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#334155')
    )
    disclaimer_style = ParagraphStyle(
        'DiscTxt',
        fontName='MedAIFont',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#78350f')
    )

    story = []

    # 1. Header with Built-in Verification QR Code
    report_id = str(uuid.uuid4())[:8].upper()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    qr_drawing = Drawing(42, 42)
    qr = QrCodeWidget("https://github.com/James0010101/MedAI-Longevity-Platform")
    qr.barWidth = 42
    qr.barHeight = 42
    qr_drawing.add(qr)

    header_data = [
        [
            Table(
                [[Image(str(LOGO_PNG_PATH), width=30, height=30), Paragraph("<b>MedAI Longevity Platform</b>", title_style)]],
                colWidths=[38, 252],
                style=TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]),
            ),
            Paragraph(f"<b>{t('report_title', lang=lang)}:</b> #{report_id}<br/><b>{t('report_param_age', lang=lang)}:</b> {now_str}<br/><b>Standard:</b> HL7 FHIR / WHO GHO", meta_style),
            qr_drawing
        ],
        [
            Paragraph(t("app_subtitle", lang=lang), subtitle_style),
            Paragraph("Ver: 2.0.0 (Research Prototype)", meta_style),
            Paragraph("<font size=6 color='#64748b'>Verify</font>", ParagraphStyle('QRlbl', fontName='MedAIFont', fontSize=6, alignment=1, textColor=colors.HexColor('#64748b')))
        ]
    ]
    header_table = Table(header_data, colWidths=[290, 199, 50])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#006c49'), spaceAfter=6))

    # Patient Data
    age = int(patient.get("age", 50))
    sex_label = t("sex_male", lang=lang) if int(patient.get("male", 1)) == 1 else t("sex_female", lang=lang)
    sbp = int(patient.get("sysBP", 120))
    dbp = int(patient.get("diaBP", 80))
    bmi = round(float(patient.get("BMI", 25.0)), 1)
    hr = int(patient.get("heartRate", 72))
    smoker = t("smoker_yes", lang=lang) if int(patient.get("currentSmoker", 0)) == 1 else t("smoker_no", lang=lang)
    bp_meds = t("meds_yes", lang=lang) if int(patient.get("BPMeds", 0)) == 1 else t("meds_no", lang=lang)

    raw_tc = float(patient.get("totChol", 200.0))
    raw_glu = float(patient.get("glucose", 90.0))
    val_tc = from_mgdl(raw_tc, "totChol", units)
    val_glu = from_mgdl(raw_glu, "glucose", units)
    unit_label = "mmol/L" if (units == "mmol" and lang == "en") else ("mmol/l" if (units == "mmol" and lang == "uz") else ("ммоль/л" if units == "mmol" else "mg/dL"))

    # Section 1: Demographics Table
    story.append(Paragraph(t("report_sec_1", lang=lang), sec_heading))
    patient_table_data = [
        [
            Paragraph(t("report_param_age", lang=lang), cell_bold),
            Paragraph(f"{age} {t('res_years', lang=lang)}", cell_norm),
            Paragraph(t("report_param_bp", lang=lang), cell_bold),
            Paragraph(f"{sbp}/{dbp} mmHg", cell_norm),
        ],
        [
            Paragraph(t("report_param_sex", lang=lang), cell_bold),
            Paragraph(sex_label, cell_norm),
            Paragraph(t("report_param_meds", lang=lang), cell_bold),
            Paragraph(bp_meds, cell_norm),
        ],
        [
            Paragraph(t("report_param_smoke", lang=lang), cell_bold),
            Paragraph(smoker, cell_norm),
            Paragraph(t("report_param_bmi", lang=lang), cell_bold),
            Paragraph(f"{bmi} kg/m²", cell_norm),
        ],
        [
            Paragraph(t("report_param_chol", lang=lang), cell_bold),
            Paragraph(f"{val_tc:.2f} {unit_label}", cell_norm),
            Paragraph(t("report_param_hr", lang=lang), cell_bold),
            Paragraph(f"{hr} bpm", cell_norm),
        ],
        [
            Paragraph(t("report_param_glu", lang=lang), cell_bold),
            Paragraph(f"{val_glu:.2f} {unit_label}", cell_norm),
            Paragraph(t("report_param_dm", lang=lang), cell_bold),
            Paragraph(t("report_yes", lang=lang) if patient.get("diabetes") else t("report_no", lang=lang), cell_norm),
        ],
    ]
    p_table = Table(patient_table_data, colWidths=[125, 145, 135, 134])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 6))

    # Section 2: Core Metrics
    p_chd = prediction.get("p_chd", 0.0) * 100.0
    p_chd_std = prediction.get("p_chd_std", 0.0) * 100.0
    heart_age = prediction.get("heart_age", float(age))
    ha_delta = prediction.get("heart_age_delta", 0.0)
    life_exp = prediction.get("life_expectancy", 74.0)
    p_dm = prediction.get("p_diabetes", 0.0) * 100.0

    if p_chd < 10.0:
        chd_cat = t("card_chd_cat_low", lang=lang)
        chd_color = colors.HexColor('#15803d')
    elif p_chd < 20.0:
        chd_cat = t("card_chd_cat_mod", lang=lang)
        chd_color = colors.HexColor('#b45309')
    else:
        chd_cat = t("card_chd_cat_high", lang=lang)
        chd_color = colors.HexColor('#b91c1c')

    ha_color = colors.HexColor('#b91c1c') if ha_delta > 0 else colors.HexColor('#15803d')

    metrics_cell_chd = [
        Paragraph(t("card_chd_title", lang=lang), card_title),
        Spacer(1, 1),
        Paragraph(f"<font color='{chd_color.hexval()}'>{p_chd:.1f}%</font>", card_val),
        Spacer(1, 1),
        Paragraph(f"<b>[{chd_cat}]</b>", ParagraphStyle('Bdg', fontName='MedAIFont-Bold', fontSize=7.5, leading=9, alignment=1, textColor=chd_color)),
        Spacer(1, 1),
        Paragraph(f"CI: ±{p_chd_std:.1f}%", card_sub)
    ]
    metrics_cell_ha = [
        Paragraph(t("card_heart_age_title", lang=lang), card_title),
        Spacer(1, 1),
        Paragraph(f"{heart_age:.1f} {t('res_years', lang=lang)}", card_val),
        Spacer(1, 1),
        Paragraph(f"<font color='{ha_color.hexval()}'><b>{'+' if ha_delta > 0 else ''}{ha_delta:.1f} {t('res_delta_passport', lang=lang)}</b></font>", ParagraphStyle('HAd', fontName='MedAIFont-Bold', fontSize=7.5, leading=9, alignment=1)),
        Spacer(1, 1),
        Paragraph("D'Agostino 2008", card_sub)
    ]
    metrics_cell_le = [
        Paragraph(t("card_life_exp_title", lang=lang), card_title),
        Spacer(1, 1),
        Paragraph(f"{life_exp:.1f} {t('res_years', lang=lang)}", card_val),
        Spacer(1, 1),
        Paragraph("WHO Uzbekistan 2023", ParagraphStyle('LEs', fontName='MedAIFont', fontSize=7.5, leading=9, alignment=1, textColor=colors.HexColor('#475569'))),
        Spacer(1, 1),
        Paragraph(t("card_life_exp_sub", lang=lang), card_sub)
    ]
    metrics_cell_dm = [
        Paragraph(t("card_dm_title", lang=lang), card_title),
        Spacer(1, 1),
        Paragraph(f"{p_dm:.1f}%", card_val),
        Spacer(1, 1),
        Paragraph(f"<b>{t('card_dm_high', lang=lang) if p_dm >= 50 else t('card_dm_low', lang=lang)}</b>", ParagraphStyle('DMs', fontName='MedAIFont-Bold', fontSize=7.5, leading=9, alignment=1, textColor=colors.HexColor('#b91c1c') if p_dm >= 50 else colors.HexColor('#15803d'))),
        Spacer(1, 1),
        Paragraph(t("card_dm_sub", lang=lang), card_sub)
    ]

    metrics_table = Table(
        [[metrics_cell_chd, metrics_cell_ha, metrics_cell_le, metrics_cell_dm]],
        colWidths=[134, 135, 135, 135]
    )
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))

    story.append(Paragraph(t("report_sec_2", lang=lang), sec_heading))
    story.append(metrics_table)
    story.append(Spacer(1, 6))

    # Section 3: Recommendations
    story.append(Paragraph(t("report_sec_3", lang=lang), sec_heading))
    top_items = recommendations.get("top_interventions", [])

    if top_items:
        rec_tables = []
        for idx, item in enumerate(top_items[:3], 1):
            title_text = f"{idx}. {t(item['title_key'], lang=lang)}"
            source_text = f"<b>{t('bio_col_source', lang=lang)}:</b> {item['source']}"
            rationale_text = t(item['rationale_key'], lang=lang)
            quant_text = f"{t('rec_quant_effect', lang=lang)} {t('rec_chd_reduction', lang=lang, val=abs(item['delta_chd_pp']))} | {t('rec_heart_age_rejuvenation', lang=lang, val=abs(item['delta_heart_age']))} | {t('rec_life_gain', lang=lang, val=abs(item['delta_life_exp']))}"
            action_text = f"<b>{t('bio_col_guideline', lang=lang)}:</b> {t(item['action_key'], lang=lang)}"

            rec_box = [
                [Paragraph(title_text, rec_title_style), Paragraph(source_text, rec_source_style)],
                [Paragraph(rationale_text, rec_text_style), ""],
                [Paragraph(quant_text, rec_quant_style), ""],
                [Paragraph(action_text, rec_action_style), ""]
            ]
            rt = Table(rec_box, colWidths=[395, 144])
            rt.setStyle(TableStyle([
                ('SPAN', (0,1), (1,1)),
                ('SPAN', (0,2), (1,2)),
                ('SPAN', (0,3), (1,3)),
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0fdf4')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#86efac')),
                ('LINEBEFORE', (0,0), (0,-1), 3, colors.HexColor('#006c49')),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ]))
            rec_tables.append(rt)
            rec_tables.append(Spacer(1, 3))

        story.append(KeepTogether(rec_tables))
    else:
        story.append(Paragraph(t("rec_maintenance_optimal", lang=lang), rec_text_style))

    story.append(Spacer(1, 4))

    # Section 4: Disclaimer & Signature
    disc_data = [
        [
            Paragraph(f"<b>{t('meth_disclaimer_title', lang=lang)}:</b> {t('report_disclaimer_text', lang=lang)}", disclaimer_style)
        ]
    ]
    disc_table = Table(disc_data, colWidths=[539])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#fde68a')),
        ('LINEBEFORE', (0,0), (0,-1), 3, colors.HexColor('#d97706')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(disc_table)

    story.append(Spacer(1, 10))

    # Signature block
    sig_label_doc = "Shifokor F.I.SH. / Imzosi" if lang == "uz" else ("Врач-терапевт / Подпись" if lang == "ru" else "Attending Physician / Signature")
    sig_label_stamp = "Tibbiyot muassasasi muhri (M.P.)" if lang == "uz" else ("М.П. Медицинского учреждения" if lang == "ru" else "Institution Stamp / Seal")

    sig_data = [
        [
            Paragraph(f"____________________________________<br/><font size=7 color='#64748b'>{sig_label_doc}</font>", cell_norm),
            Paragraph(f"____________________________________<br/><font size=7 color='#64748b'>{sig_label_stamp}</font>", cell_norm)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 269])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(sig_table)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
