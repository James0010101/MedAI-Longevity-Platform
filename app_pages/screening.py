"""
app_pages/screening.py — Core Clinical Risk Screening & Actionable Longevity Interface.
"""
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.branding import logo_header_html
from core.i18n import get_current_language, t
from core.recommendations import generate_recommendations
from core.report import generate_fhir_bundle, generate_html_report, generate_pdf_report
from core.risk_model import (
    compute_risk_trajectory,
    explain_prediction,
    group_attributions,
    predict_patient,
    simulate_intervention,
)
from core.thresholds import classify
from core.units import from_mgdl, to_mgdl

lang = get_current_language()
units = st.session_state.get("units", "mmol")

# Clinical Demo Presets
PRESETS = {
    "preset_a": {
        "male": 0,
        "age": 35.0,
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
        "sysBP": 115.0,
        "diaBP": 75.0,
        "BMI": 21.5,
        "heartRate": 68.0,
        "totChol": 170.0,  # mg/dL
        "glucose": 85.0,   # mg/dL
        "diabetes": 0,
    },
    "preset_b": {
        "male": 1,
        "age": 52.0,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "sysBP": 142.0,
        "diaBP": 88.0,
        "BMI": 27.2,
        "heartRate": 76.0,
        "totChol": 225.0,
        "glucose": 98.0,
        "diabetes": 0,
    },
    "preset_c": {
        "male": 0,
        "age": 60.0,
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 1,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "sysBP": 155.0,
        "diaBP": 92.0,
        "BMI": 29.5,
        "heartRate": 74.0,
        "totChol": 240.0,
        "glucose": 105.0,
        "diabetes": 0,
    },
    "preset_d": {
        "male": 1,
        "age": 66.0,
        "currentSmoker": 1,
        "cigsPerDay": 25.0,
        "BPMeds": 1,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "sysBP": 168.0,
        "diaBP": 98.0,
        "BMI": 31.8,
        "heartRate": 82.0,
        "totChol": 260.0,
        "glucose": 145.0,
        "diabetes": 1,
    },
}

# Initialize active patient in session state if missing
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = PRESETS["preset_b"].copy()

# Header
st.markdown(logo_header_html(t("app_title", lang=lang), size=64), unsafe_allow_html=True)
st.caption(t("app_subtitle", lang=lang))

# Quick Profile Selector
st.markdown(f"##### {t('btn_preset_profile', lang=lang)}")
p_cols = st.columns(4)
for idx, (p_key, p_val) in enumerate(PRESETS.items()):
    with p_cols[idx]:
        if st.button(t(p_key, lang=lang), key=f"btn_{p_key}", width="stretch"):
            st.session_state["patient_data"] = p_val.copy()
            st.rerun()

st.divider()

# Input Form
cur_p = st.session_state["patient_data"]

with st.container(border=True):
    st.subheader(t("section_demographics", lang=lang))
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        age_val = st.slider(
            t("field_age", lang=lang),
            min_value=18,
            max_value=90,
            value=int(cur_p.get("age", 52)),
            step=1,
            help=t("field_age_help", lang=lang),
        )
        cur_p["age"] = float(age_val)
        if age_val < 32 or age_val > 70:
            st.warning(t("age_extrapolation_warning", lang=lang))

    with col_demo2:
        sex_idx = 0 if cur_p.get("male", 1) == 1 else 1
        sex_choice = st.radio(
            t("field_sex", lang=lang),
            options=[t("sex_male", lang=lang), t("sex_female", lang=lang)],
            index=sex_idx,
            horizontal=True,
        )
        cur_p["male"] = 1 if sex_choice == t("sex_male", lang=lang) else 0

with st.container(border=True):
    st.subheader(t("section_lifestyle", lang=lang))
    col_life1, col_life2, col_life3 = st.columns(3)
    with col_life1:
        smoker_idx = 0 if cur_p.get("currentSmoker", 0) == 1 else 1
        smoker_choice = st.radio(
            t("field_smoker", lang=lang),
            options=[t("smoker_yes", lang=lang), t("smoker_no", lang=lang)],
            index=smoker_idx,
            horizontal=True,
        )
        is_smoker = 1 if smoker_choice == t("smoker_yes", lang=lang) else 0
        cur_p["currentSmoker"] = is_smoker

        if is_smoker:
            cigs = st.slider(
                t("field_cigs_per_day", lang=lang),
                min_value=1,
                max_value=60,
                value=max(1, int(cur_p.get("cigsPerDay", 15))),
                step=1,
            )
            cur_p["cigsPerDay"] = float(cigs)
        else:
            cur_p["cigsPerDay"] = 0.0

    with col_life2:
        meds_idx = 0 if cur_p.get("BPMeds", 0) == 1 else 1
        meds_choice = st.radio(
            t("field_bp_meds", lang=lang),
            options=[t("meds_yes", lang=lang), t("meds_no", lang=lang)],
            index=meds_idx,
            horizontal=True,
        )
        cur_p["BPMeds"] = 1 if meds_choice == t("meds_yes", lang=lang) else 0

    with col_life3:
        stroke_idx = 0 if cur_p.get("prevalentStroke", 0) == 1 else 1
        stroke_choice = st.radio(
            t("field_prev_stroke", lang=lang),
            options=[t("meds_yes", lang=lang), t("meds_no", lang=lang)],
            index=stroke_idx,
            horizontal=True,
        )
        cur_p["prevalentStroke"] = 1 if stroke_choice == t("meds_yes", lang=lang) else 0
        # Automatically mark prevalentHyp if SBP >= 140 or on BPMeds
        cur_p["prevalentHyp"] = 1 if cur_p["BPMeds"] or cur_p.get("sysBP", 120) >= 140 else 0

with st.container(border=True):
    st.subheader(t("section_vitals", lang=lang))
    col_vit1, col_vit2, col_vit3, col_vit4 = st.columns(4)
    with col_vit1:
        sys_bp = st.slider(
            t("field_sys_bp", lang=lang),
            min_value=85,
            max_value=220,
            value=int(cur_p.get("sysBP", 140)),
            step=1,
        )
        cur_p["sysBP"] = float(sys_bp)

    with col_vit2:
        dia_bp = st.slider(
            t("field_dia_bp", lang=lang),
            min_value=50,
            max_value=130,
            value=int(cur_p.get("diaBP", 88)),
            step=1,
        )
        cur_p["diaBP"] = float(dia_bp)

    with col_vit3:
        bmi_val = st.slider(
            t("field_bmi", lang=lang),
            min_value=16.0,
            max_value=48.0,
            value=float(cur_p.get("BMI", 27.0)),
            step=0.1,
        )
        cur_p["BMI"] = float(bmi_val)

        with st.expander(t("calc_bmi_expander", lang=lang)):
            h_c, w_c = st.columns(2)
            h_in = h_c.number_input(t("calc_height_label", lang=lang), min_value=120, max_value=220, value=172, step=1)
            w_in = w_c.number_input(t("calc_weight_label", lang=lang), min_value=35.0, max_value=180.0, value=75.0, step=0.5)
            c_bmi = round(w_in / ((h_in / 100.0) ** 2), 1)
            st.caption(t("calc_bmi_result", lang=lang, bmi=c_bmi))
            if st.button(t("calc_bmi_btn", lang=lang), key="btn_apply_bmi", width="stretch"):
                cur_p["BMI"] = float(c_bmi)
                st.session_state["patient_data"] = cur_p
                st.rerun()

    with col_vit4:
        hr_val = st.slider(
            t("field_heart_rate", lang=lang),
            min_value=45,
            max_value=140,
            value=int(cur_p.get("heartRate", 75)),
            step=1,
        )
        cur_p["heartRate"] = float(hr_val)

with st.container(border=True):
    st.subheader(t("section_lab", lang=lang))
    col_lab1, col_lab2, col_lab3 = st.columns(3)

    u_label = "mmol/L" if (units == "mmol" and lang == "en") else ("mmol/l" if (units == "mmol" and lang == "uz") else ("ммоль/л" if units == "mmol" else "mg/dL"))

    with col_lab1:
        if units == "mmol":
            curr_tc_mmol = from_mgdl(cur_p.get("totChol", 200.0), "totChol", "mmol")
            tc_input = st.slider(
                f"{t('field_tot_chol', lang=lang)} ({u_label})",
                min_value=2.5,
                max_value=12.0,
                value=round(curr_tc_mmol, 1),
                step=0.1,
            )
            cur_p["totChol"] = to_mgdl(tc_input, "totChol", "mmol")
        else:
            tc_input = st.slider(
                f"{t('field_tot_chol', lang=lang)} (mg/dL)",
                min_value=100,
                max_value=450,
                value=int(cur_p.get("totChol", 200.0)),
                step=5,
            )
            cur_p["totChol"] = float(tc_input)

    with col_lab2:
        if units == "mmol":
            curr_glu_mmol = from_mgdl(cur_p.get("glucose", 90.0), "glucose", "mmol")
            glu_input = st.slider(
                f"{t('field_glucose', lang=lang)} ({u_label})",
                min_value=3.0,
                max_value=20.0,
                value=round(curr_glu_mmol, 1),
                step=0.1,
            )
            cur_p["glucose"] = to_mgdl(glu_input, "glucose", "mmol")
        else:
            glu_input = st.slider(
                f"{t('field_glucose', lang=lang)} (mg/dL)",
                min_value=60,
                max_value=350,
                value=int(cur_p.get("glucose", 90.0)),
                step=2,
            )
            cur_p["glucose"] = float(glu_input)

    with col_lab3:
        diab_idx = 0 if cur_p.get("diabetes", 0) == 1 else 1
        diab_choice = st.radio(
            t("field_known_diabetes", lang=lang),
            options=[t("meds_yes", lang=lang), t("meds_no", lang=lang)],
            index=diab_idx,
            horizontal=True,
        )
        cur_p["diabetes"] = 1 if diab_choice == t("meds_yes", lang=lang) else 0

st.session_state["patient_data"] = cur_p

# Perform Real-Time Inference
prediction = predict_patient(cur_p)
recs = generate_recommendations(cur_p)

# Display 4 Main Metric Cards
st.markdown(f"### {t('res_screening_title', lang=lang)}")
c_m1, c_m2, c_m3, c_m4 = st.columns(4)

with c_m1:
    with st.container(border=True):
        st.caption(t("card_chd_title", lang=lang))
        p_chd = prediction["p_chd"] * 100.0
        p_chd_std = prediction["p_chd_std"] * 100.0

        if p_chd < 10.0:
            badge_cat = t("card_chd_cat_low", lang=lang)
            badge_color = "#006c49"
        elif p_chd < 20.0:
            badge_cat = t("card_chd_cat_mod", lang=lang)
            badge_color = "#b25e00"
        else:
            badge_cat = t("card_chd_cat_high", lang=lang)
            badge_color = "#ba1a1a"

        st.metric(
            label=t("res_label_chd", lang=lang),
            value=f"{p_chd:.1f}%",
            delta=None,
        )
        st.markdown(
            f"<div style='background: {badge_color}; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; display: inline-block;'>{badge_cat}</div>",
            unsafe_allow_html=True,
        )
        st.caption(t("card_chd_uncertainty", lang=lang, val=f"{p_chd_std:.1f}"))

with c_m2:
    with st.container(border=True):
        st.caption(t("card_heart_age_title", lang=lang))
        heart_age = prediction["heart_age"]
        ha_delta = prediction["heart_age_delta"]

        st.metric(
            label=t("res_label_heart_age", lang=lang),
            value=f"{heart_age:.1f} {t('res_years', lang=lang)}",
            delta=f"{'+' if ha_delta > 0 else ''}{ha_delta:.1f} {t('res_delta_passport', lang=lang)}",
            delta_color="inverse" if ha_delta > 0 else "normal",
        )
        st.caption(t("card_heart_age_sub", lang=lang))

with c_m3:
    with st.container(border=True):
        st.caption(t("card_life_exp_title", lang=lang))
        life_exp = prediction["life_expectancy"]

        st.metric(
            label=t("res_label_life_exp", lang=lang),
            value=f"{life_exp:.1f} {t('res_years', lang=lang)}",
            delta=None,
        )
        st.caption(t("card_life_exp_sub", lang=lang))

with c_m4:
    with st.container(border=True):
        st.caption(t("card_dm_title", lang=lang))
        p_dm = prediction["p_diabetes"] * 100.0
        is_dm_signal = p_dm >= 50.0

        st.metric(
            label=t("res_label_diabetes", lang=lang),
            value=f"{p_dm:.1f}%",
            delta=None,
        )
        dm_color = "#ba1a1a" if is_dm_signal else "#006c49"
        dm_label = t("card_dm_high", lang=lang) if is_dm_signal else t("card_dm_low", lang=lang)
        st.markdown(
            f"<div style='background: {dm_color}; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; display: inline-block;'>{dm_label}</div>",
            unsafe_allow_html=True,
        )
        st.caption(t("card_dm_sub", lang=lang))

st.divider()

# Clinical Verdict Banner
xai_quick = explain_prediction(cur_p, head="chd", m_steps=20)
attrs_sorted = sorted(group_attributions(xai_quick["attributions_pp"]).items(), key=lambda x: x[1], reverse=True)
top_driver_key = attrs_sorted[0][0]
top_driver_val = attrs_sorted[0][1]
top_driver_desc = t(f"feat_{top_driver_key}", lang=lang)

with st.container(border=True):
    v_icon = "🟢" if p_chd < 10.0 else ("🟠" if p_chd < 20.0 else "🔴")
    st.markdown(f"#### {v_icon} {t('verdict_title', lang=lang)}")
    if p_chd < 10.0:
        verdict_msg = t(
            "verdict_msg_low",
            lang=lang,
            badge=badge_cat.lower(),
            chd=f"{p_chd:.1f}",
            ha=f"{heart_age:.1f}",
            delta=f"{'+' if ha_delta > 0 else ''}{ha_delta:.1f}",
            le=f"{life_exp:.1f}",
        )
    else:
        verdict_msg = t(
            "verdict_msg_elevated",
            lang=lang,
            badge=badge_cat.lower(),
            chd=f"{p_chd:.1f}",
            ha=f"{heart_age:.1f}",
            delta=f"{'+' if ha_delta > 0 else ''}{ha_delta:.1f}",
            driver=top_driver_desc,
            contrib=f"{top_driver_val:.1f}",
        )
    st.write(verdict_msg)

st.divider()

# Priority Action Steps (TOP-3)
st.subheader(t("sec_priorities_title", lang=lang))
st.caption(t("sec_priorities_sub", lang=lang))

top_items = recs.get("top_interventions", [])
if top_items:
    step_cols = st.columns(len(top_items))
    for idx, item in enumerate(top_items):
        with step_cols[idx]:
            with st.container(border=True):
                st.markdown(f"#### #{idx+1}. {t(item['title_key'], lang=lang)}")
                st.caption(f"**{t('rec_source', lang=lang)}:** {item['source']}")
                st.write(t(item["rationale_key"], lang=lang))

                st.markdown(f"**{t('rec_quant_effect', lang=lang)}**")
                v_chd = f"{abs(item['delta_chd_pp']):.1f}"
                v_ha = f"{abs(item['delta_heart_age']):.1f}"
                v_le = f"{abs(item['delta_life_exp']):.1f}"
                if abs(item["delta_chd_pp"]) >= 0.05:
                    st.write(f"- {t('rec_chd_reduction', lang=lang, val=v_chd)}")
                if abs(item["delta_heart_age"]) >= 0.1:
                    st.write(f"- {t('rec_heart_age_rejuvenation', lang=lang, val=v_ha)}")
                if abs(item["delta_life_exp"]) >= 0.1:
                    st.write(f"- {t('rec_life_gain', lang=lang, val=v_le)}")

                st.markdown(f"*{t(item['action_key'], lang=lang)}*")

    # Combined Impact Card
    comb = recs.get("combined_impact", {})
    if comb:
        comb_deltas = comb.get("deltas", {})
        st.markdown(f"#### 🌟 {t('rec_combined_title', lang=lang)}")
        st.info(
            t(
                "rec_combined_desc",
                lang=lang,
                chd=f"{abs(comb_deltas.get('delta_chd_pp', 0.0)):.1f}",
                va=f"{abs(comb_deltas.get('delta_heart_age', 0.0)):.1f}",
                le=f"{abs(comb_deltas.get('delta_life_exp', 0.0)):.1f}",
            )
        )
else:
    st.success(f"✅ {t('rec_maintenance_optimal', lang=lang)}")

st.divider()

# Interactive Analytical Tabs
tab_sim, tab_xai, tab_bio, tab_traj, tab_exp = st.tabs([
    f"🎯 {t('tab_what_if', lang=lang)}",
    f"📊 {t('tab_xai', lang=lang)}",
    f"📋 {t('tab_biomarkers', lang=lang)}",
    f"📈 {t('tab_trajectory', lang=lang)}",
    f"📄 {t('tab_export', lang=lang)}",
])

with tab_sim:
    st.subheader(t("what_if_heading", lang=lang))
    st.caption(t("what_if_desc", lang=lang))

    st.markdown(f"##### {t('sim_checkbox_heading', lang=lang)}")
    col_c1, col_c2, col_c3 = st.columns(3)

    sim_changes = {}
    with col_c1:
        if cur_p.get("currentSmoker", 0) == 1:
            q_smoke = st.checkbox(t("sim_smoke_label", lang=lang), value=True, key="sim_smoke")
            if q_smoke:
                sim_changes["currentSmoker"] = 0
                sim_changes["cigsPerDay"] = 0.0

        if cur_p.get("sysBP", 120) >= 130 or cur_p.get("diaBP", 80) >= 85:
            q_bp = st.checkbox(t("sim_bp_label", lang=lang), value=True, key="sim_bp")
            if q_bp:
                sim_changes["sysBP"] = 125.0
                sim_changes["diaBP"] = 78.0
                sim_changes["prevalentHyp"] = 0

    with col_c2:
        if cur_p.get("totChol", 200) >= 200:
            q_chol = st.checkbox(t("sim_chol_label", lang=lang), value=True, key="sim_chol")
            if q_chol:
                sim_changes["totChol"] = max(160.0, round(float(cur_p.get("totChol", 200)) * 0.8, 1))

        if cur_p.get("BMI", 25) >= 25.0:
            q_bmi = st.checkbox(t("sim_bmi_label", lang=lang), value=False, key="sim_bmi")
            if q_bmi:
                sim_changes["BMI"] = 24.5

    with col_c3:
        if cur_p.get("glucose", 90) >= 100 or cur_p.get("diabetes", 0) == 1:
            q_glu = st.checkbox(t("sim_glu_label", lang=lang), value=False, key="sim_glu")
            if q_glu:
                sim_changes["glucose"] = 95.0

    if sim_changes:
        sim_dyn = simulate_intervention(cur_p, sim_changes)
        b_res = sim_dyn["base"]
        s_res = sim_dyn["simulated"]
        d_res = sim_dyn["deltas"]

        st.markdown(f"##### {t('sim_comparison_title', lang=lang)}")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            with st.container(border=True):
                st.caption(t("card_chd_title", lang=lang))
                st.metric(
                    label=t("sim_metric_new_chd", lang=lang),
                    value=f"{s_res['p_chd']*100:.1f}%",
                    delta=f"{d_res['delta_chd_pp']:.1f} {t('sim_pp', lang=lang)}",
                    delta_color="normal",
                )
        with col_s2:
            with st.container(border=True):
                st.caption(t("card_heart_age_title", lang=lang))
                st.metric(
                    label=t("sim_metric_new_heart_age", lang=lang),
                    value=f"{s_res['heart_age']:.1f} {t('res_years', lang=lang)}",
                    delta=f"{d_res['delta_heart_age']:.1f} {t('res_years', lang=lang)}",
                    delta_color="normal",
                )
        with col_s3:
            with st.container(border=True):
                st.caption(t("card_life_exp_title", lang=lang))
                st.metric(
                    label=t("sim_metric_new_life_exp", lang=lang),
                    value=f"{s_res['life_expectancy']:.1f} {t('res_years', lang=lang)}",
                    delta=f"+{d_res['delta_life_exp']:.1f} {t('res_years', lang=lang)}",
                    delta_color="normal",
                )
        with col_s4:
            with st.container(border=True):
                st.caption(t("card_dm_title", lang=lang))
                st.metric(
                    label=t("sim_metric_new_dm", lang=lang),
                    value=f"{s_res['p_diabetes']*100:.1f}%",
                    delta=f"{d_res['delta_diabetes_pp']:.1f} {t('sim_pp', lang=lang)}",
                    delta_color="normal",
                )
    else:
        st.info(t("what_if_desc", lang=lang))

with tab_xai:
    st.subheader(t("xai_heading", lang=lang))
    st.caption(t("xai_desc", lang=lang))

    xai_res = explain_prediction(cur_p, head="chd", m_steps=50)
    attrs = group_attributions(xai_res["attributions_pp"])

    lbl_inc = t("xai_type_increase", lang=lang)
    lbl_dec = t("xai_type_decrease", lang=lang)

    plot_data = []
    for k, v in attrs.items():
        if abs(v) >= 0.05:
            plot_data.append({
                "feature": t(f"feat_{k}", lang=lang),
                "contribution": v,
                "type": lbl_inc if v > 0 else lbl_dec,
            })

    if plot_data:
        df_xai = pd.DataFrame(plot_data).sort_values(by="contribution", ascending=True)

        chart_xai = (
            alt.Chart(df_xai)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("contribution:Q", title=t("xai_axis_title", lang=lang)),
                y=alt.Y("feature:N", sort=None, title=None),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(domain=[lbl_inc, lbl_dec], range=["#ba1a1a", "#006c49"]),
                    title=None,
                ),
                tooltip=["feature", "contribution", "type"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_xai, width="stretch")
    else:
        st.info(t("xai_optimal_notice", lang=lang))

with tab_bio:
    st.subheader(t("biomarkers_heading", lang=lang))
    st.caption(t("biomarkers_desc", lang=lang))

    tc_val_display = from_mgdl(cur_p.get("totChol", 200), "totChol", units)
    glu_val_display = from_mgdl(cur_p.get("glucose", 90), "glucose", units)

    bp_unit = "mmHg" if lang == "en" else ("mm sim.ust." if lang == "uz" else "мм рт.ст.")
    bp_cls = classify("bp", cur_p.get("sysBP", 120), sex=str(cur_p.get("male", 1)), secondary_value=cur_p.get("diaBP", 80), lang=lang)
    tc_cls = classify("totChol", cur_p.get("totChol", 200), sex=str(cur_p.get("male", 1)), lang=lang)
    glu_cls = classify("glucose", cur_p.get("glucose", 90), sex=str(cur_p.get("male", 1)), lang=lang)
    bmi_cls = classify("bmi", cur_p.get("BMI", 25), sex=str(cur_p.get("male", 1)), lang=lang)

    def status_badge(lvl):
        if lvl == "ok":
            return t("status_badge_ok", lang=lang)
        elif lvl == "warn":
            return t("status_badge_warn", lang=lang)
        else:
            return t("status_badge_high", lang=lang)

    col_name = t("bio_col_name", lang=lang)
    col_val = t("bio_col_val", lang=lang)
    col_norm = t("bio_col_norm", lang=lang)
    col_status = t("bio_col_status", lang=lang)
    col_source = t("bio_col_source", lang=lang)

    bio_rows = [
        {
            col_name: t("bio_marker_bp", lang=lang),
            col_val: f"{int(cur_p.get('sysBP', 120))} / {int(cur_p.get('diaBP', 80))} {bp_unit}",
            col_norm: bp_cls.get("normal_range", "< 130/80"),
            col_status: status_badge(bp_cls.get("level", "ok")),
            col_source: bp_cls.get("source", "ESC/ESH 2018"),
        },
        {
            col_name: t("bio_marker_chol", lang=lang, unit=u_label),
            col_val: f"{tc_val_display:.2f} {u_label}",
            col_norm: tc_cls.get("normal_range", "< 200 mg/dL"),
            col_status: status_badge(tc_cls.get("level", "ok")),
            col_source: tc_cls.get("source", "NCEP ATP III"),
        },
        {
            col_name: t("bio_marker_glu", lang=lang, unit=u_label),
            col_val: f"{glu_val_display:.2f} {u_label}",
            col_norm: glu_cls.get("normal_range", "< 110 mg/dL"),
            col_status: status_badge(glu_cls.get("level", "ok")),
            col_source: glu_cls.get("source", "WHO / ADA"),
        },
        {
            col_name: t("bio_marker_bmi", lang=lang),
            col_val: f"{cur_p.get('BMI', 25):.1f} kg/m²",
            col_norm: "18.5 – 24.9 kg/m²",
            col_status: status_badge(bmi_cls.get("level", "ok")),
            col_source: bmi_cls.get("source", "WHO Guidelines"),
        },
        {
            col_name: t("bio_marker_smoker", lang=lang),
            col_val: t("bio_smoker_val", lang=lang, cigs=int(cur_p.get('cigsPerDay', 0))) if cur_p.get("currentSmoker") else t("bio_nonsmoker_val", lang=lang),
            col_norm: t("bio_cessation_norm", lang=lang),
            col_status: t("status_risk_factor", lang=lang) if cur_p.get("currentSmoker") else t("status_normal", lang=lang),
            col_source: "WHO FCTC / ESC 2021",
        },
        {
            col_name: t("bio_marker_hr", lang=lang),
            col_val: f"{int(cur_p.get('heartRate', 72))} bpm",
            col_norm: "60 – 80 bpm",
            col_status: t("status_badge_ok", lang=lang) if 60 <= cur_p.get("heartRate", 72) <= 80 else t("status_out_of_range", lang=lang),
            col_source: "ESC Guidelines",
        },
    ]
    st.table(pd.DataFrame(bio_rows))

with tab_traj:
    st.subheader(t("traj_heading", lang=lang))
    st.caption(t("traj_desc", lang=lang))

    opt_changes = recs.get("combined_impact", {}).get("changes") if recs.get("has_interventions") else None
    traj_records = compute_risk_trajectory(
        cur_p,
        years_ahead=(0, 2, 5, 8, 10, 12, 15, 18, 20),
        improved_changes=opt_changes,
    )

    df_traj = pd.DataFrame(traj_records)
    df_traj["scenario_label"] = df_traj["scenario"].map({
        "current": t("traj_scenario_current", lang=lang),
        "improved": t("traj_scenario_improved", lang=lang),
    })

    chart_traj = (
        alt.Chart(df_traj)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("age:Q", title=t("traj_axis_age", lang=lang)),
            y=alt.Y("p_chd_percent:Q", title=t("traj_axis_risk", lang=lang)),
            color=alt.Color(
                "scenario_label:N",
                scale=alt.Scale(
                    domain=[t("traj_scenario_current", lang=lang), t("traj_scenario_improved", lang=lang)],
                    range=["#ba1a1a", "#006c49"],
                ),
                title=t("tab_trajectory", lang=lang),
            ),
            tooltip=["age", "p_chd_percent", "heart_age", "life_expectancy", "scenario_label"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_traj, width="stretch")

with tab_exp:
    st.subheader(t("export_heading", lang=lang))
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.container(border=True):
            st.markdown(f"#### {t('export_report_title', lang=lang)}")
            st.write(t("export_report_desc", lang=lang))
            pdf_bytes = generate_pdf_report(cur_p, prediction, recs, lang=lang, units=units)
            st.download_button(
                label=t("export_pdf_btn", lang=lang),
                data=pdf_bytes,
                file_name=f"MedAI_Clinical_Report_{int(cur_p.get('age', 50))}_{'M' if cur_p.get('male', 1) else 'F'}.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
            html_content = generate_html_report(cur_p, prediction, recs, lang=lang, units=units)
            st.download_button(
                label=t("export_html_btn", lang=lang),
                data=html_content,
                file_name=f"MedAI_Report_{int(cur_p.get('age', 50))}_{'M' if cur_p.get('male', 1) else 'F'}.html",
                mime="text/html",
                width="stretch",
            )

    with exp_col2:
        with st.container(border=True):
            st.markdown(f"#### {t('export_fhir_title', lang=lang)}")
            st.write(t("export_fhir_desc", lang=lang))
            fhir_bundle = generate_fhir_bundle(cur_p, prediction, lang=lang)
            fhir_json = json.dumps(fhir_bundle, indent=2, ensure_ascii=False)
            st.download_button(
                label=t("export_fhir_btn", lang=lang),
                data=fhir_json,
                file_name="fhir_bundle_screening.json",
                mime="application/json",
                width="stretch",
            )
