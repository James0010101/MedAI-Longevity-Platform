"""
app_pages/screening.py — Core Clinical Risk Screening & Actionable Longevity Interface.
"""
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.i18n import get_current_language, t
from core.recommendations import generate_recommendations
from core.report import generate_fhir_bundle, generate_html_report
from core.risk_model import (
    compute_risk_trajectory,
    explain_prediction,
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
st.title(f"🫀 {t('app_title')}")
st.caption(t("app_subtitle"))

# Quick Profile Selector
st.markdown(f"##### {t('btn_preset_profile')}")
p_cols = st.columns(4)
for idx, (p_key, p_val) in enumerate(PRESETS.items()):
    with p_cols[idx]:
        if st.button(t(p_key), key=f"btn_{p_key}", width="stretch"):
            st.session_state["patient_data"] = p_val.copy()
            st.rerun()

st.divider()

# Input Form
cur_p = st.session_state["patient_data"]

with st.container(border=True):
    st.subheader(t("section_demographics"))
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        age_val = st.slider(
            t("field_age"),
            min_value=18,
            max_value=90,
            value=int(cur_p.get("age", 52)),
            step=1,
            help=t("field_age_help"),
        )
        cur_p["age"] = float(age_val)
        if age_val < 32 or age_val > 70:
            st.warning(t("age_extrapolation_warning"))

    with col_demo2:
        sex_idx = 0 if cur_p.get("male", 1) == 1 else 1
        sex_choice = st.radio(
            t("field_sex"),
            options=[t("sex_male"), t("sex_female")],
            index=sex_idx,
            horizontal=True,
        )
        cur_p["male"] = 1 if sex_choice == t("sex_male") else 0

with st.container(border=True):
    st.subheader(t("section_lifestyle"))
    col_life1, col_life2, col_life3 = st.columns(3)
    with col_life1:
        smoker_idx = 0 if cur_p.get("currentSmoker", 0) == 1 else 1
        smoker_choice = st.radio(
            t("field_smoker"),
            options=[t("smoker_yes"), t("smoker_no")],
            index=smoker_idx,
            horizontal=True,
        )
        is_smoker = 1 if smoker_choice == t("smoker_yes") else 0
        cur_p["currentSmoker"] = is_smoker

        if is_smoker:
            cigs = st.slider(
                t("field_cigs_per_day"),
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
            t("field_bp_meds"),
            options=[t("meds_yes"), t("meds_no")],
            index=meds_idx,
            horizontal=True,
        )
        cur_p["BPMeds"] = 1 if meds_choice == t("meds_yes") else 0

    with col_life3:
        stroke_idx = 0 if cur_p.get("prevalentStroke", 0) == 1 else 1
        stroke_choice = st.radio(
            t("field_prev_stroke"),
            options=[t("meds_yes"), t("meds_no")],
            index=stroke_idx,
            horizontal=True,
        )
        cur_p["prevalentStroke"] = 1 if stroke_choice == t("meds_yes") else 0
        # Automatically mark prevalentHyp if SBP >= 140 or on BPMeds
        cur_p["prevalentHyp"] = 1 if cur_p["BPMeds"] or cur_p.get("sysBP", 120) >= 140 else 0

with st.container(border=True):
    st.subheader(t("section_vitals"))
    col_vit1, col_vit2, col_vit3, col_vit4 = st.columns(4)
    with col_vit1:
        sys_bp = st.slider(
            t("field_sys_bp"),
            min_value=85,
            max_value=220,
            value=int(cur_p.get("sysBP", 140)),
            step=1,
        )
        cur_p["sysBP"] = float(sys_bp)

    with col_vit2:
        dia_bp = st.slider(
            t("field_dia_bp"),
            min_value=50,
            max_value=130,
            value=int(cur_p.get("diaBP", 88)),
            step=1,
        )
        cur_p["diaBP"] = float(dia_bp)

    with col_vit3:
        bmi_val = st.slider(
            t("field_bmi"),
            min_value=16.0,
            max_value=48.0,
            value=float(cur_p.get("BMI", 27.0)),
            step=0.1,
        )
        cur_p["BMI"] = float(bmi_val)

        with st.expander(t("calc_bmi_expander")):
            h_c, w_c = st.columns(2)
            h_in = h_c.number_input("Рост (см)", min_value=120, max_value=220, value=172, step=1)
            w_in = w_c.number_input("Вес (кг)", min_value=35.0, max_value=180.0, value=75.0, step=0.5)
            c_bmi = round(w_in / ((h_in / 100.0) ** 2), 1)
            st.caption(f"Рассчитанный ИМТ: **{c_bmi}** кг/м²")
            if st.button(t("calc_bmi_btn"), key="btn_apply_bmi", width="stretch"):
                cur_p["BMI"] = float(c_bmi)
                st.session_state["patient_data"] = cur_p
                st.rerun()

    with col_vit4:
        hr_val = st.slider(
            t("field_heart_rate"),
            min_value=45,
            max_value=140,
            value=int(cur_p.get("heartRate", 75)),
            step=1,
        )
        cur_p["heartRate"] = float(hr_val)

with st.container(border=True):
    st.subheader(t("section_lab"))
    col_lab1, col_lab2, col_lab3 = st.columns(3)

    with col_lab1:
        if units == "mmol":
            curr_tc_mmol = from_mgdl(cur_p.get("totChol", 200.0), "totChol", "mmol")
            tc_input = st.slider(
                f"{t('field_tot_chol')} (ммоль/л)",
                min_value=2.5,
                max_value=12.0,
                value=round(curr_tc_mmol, 1),
                step=0.1,
            )
            cur_p["totChol"] = to_mgdl(tc_input, "totChol", "mmol")
        else:
            tc_input = st.slider(
                f"{t('field_tot_chol')} (mg/dL)",
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
                f"{t('field_glucose')} (ммоль/л)",
                min_value=3.0,
                max_value=20.0,
                value=round(curr_glu_mmol, 1),
                step=0.1,
            )
            cur_p["glucose"] = to_mgdl(glu_input, "glucose", "mmol")
        else:
            glu_input = st.slider(
                f"{t('field_glucose')} (mg/dL)",
                min_value=60,
                max_value=350,
                value=int(cur_p.get("glucose", 90.0)),
                step=2,
            )
            cur_p["glucose"] = float(glu_input)

    with col_lab3:
        diab_idx = 0 if cur_p.get("diabetes", 0) == 1 else 1
        diab_choice = st.radio(
            t("field_known_diabetes"),
            options=[t("meds_yes"), t("meds_no")],
            index=diab_idx,
            horizontal=True,
        )
        cur_p["diabetes"] = 1 if diab_choice == t("meds_yes") else 0

st.session_state["patient_data"] = cur_p

# Perform Real-Time Inference
prediction = predict_patient(cur_p)
recs = generate_recommendations(cur_p)

# Display 4 Main Metric Cards
st.markdown("### Результаты скрининга")
c_m1, c_m2, c_m3, c_m4 = st.columns(4)

with c_m1:
    with st.container(border=True):
        st.caption(t("card_chd_title"))
        p_chd = prediction["p_chd"] * 100.0
        p_chd_std = prediction["p_chd_std"] * 100.0

        if p_chd < 10.0:
            badge_cat = t("card_chd_cat_low")
            badge_color = "#006c49"
        elif p_chd < 20.0:
            badge_cat = t("card_chd_cat_mod")
            badge_color = "#b25e00"
        else:
            badge_cat = t("card_chd_cat_high")
            badge_color = "#ba1a1a"

        st.metric(
            label="Риск за 10 лет",
            value=f"{p_chd:.1f}%",
            delta=None,
        )
        st.markdown(
            f"<div style='background: {badge_color}; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; display: inline-block;'>{badge_cat}</div>",
            unsafe_allow_html=True,
        )
        st.caption(t("card_chd_uncertainty", val=f"{p_chd_std:.1f}"))

with c_m2:
    with st.container(border=True):
        st.caption(t("card_heart_age_title"))
        heart_age = prediction["heart_age"]
        ha_delta = prediction["heart_age_delta"]

        st.metric(
            label="Биологический возраст",
            value=f"{heart_age:.1f} лет",
            delta=f"{'+' if ha_delta > 0 else ''}{ha_delta:.1f} к паспортному",
            delta_color="inverse" if ha_delta > 0 else "normal",
        )
        st.caption(t("card_heart_age_sub"))

with c_m3:
    with st.container(border=True):
        st.caption(t("card_life_exp_title"))
        life_exp = prediction["life_expectancy"]

        st.metric(
            label="Оценка долголетия",
            value=f"{life_exp:.1f} лет",
            delta=None,
        )
        st.caption(t("card_life_exp_sub"))

with c_m4:
    with st.container(border=True):
        st.caption(t("card_dm_title"))
        p_dm = prediction["p_diabetes"] * 100.0
        is_dm_signal = p_dm >= 50.0

        st.metric(
            label="Вероятность диабета",
            value=f"{p_dm:.1f}%",
            delta=None,
        )
        dm_color = "#ba1a1a" if is_dm_signal else "#006c49"
        dm_label = t("card_dm_high") if is_dm_signal else t("card_dm_low")
        st.markdown(
            f"<div style='background: {dm_color}; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; display: inline-block;'>{dm_label}</div>",
            unsafe_allow_html=True,
        )
        st.caption(t("card_dm_sub"))

# Clinical Verdict Banner
xai_quick = explain_prediction(cur_p, head="chd", m_steps=20)
attrs_sorted = sorted(xai_quick["attributions_pp"].items(), key=lambda x: x[1], reverse=True)
top_driver_key = attrs_sorted[0][0]
top_driver_val = attrs_sorted[0][1]

feat_names_ru = {
    "sysBP": "повышенное систолическое артериальное давление",
    "currentSmoker": "табакокурение",
    "cigsPerDay": "интенсивность курения табака",
    "totChol": "гиперхолестеринемия (высокий уровень холестерина)",
    "glucose": "повышенный уровень глюкозы крови",
    "BMI": "избыточная масса тела / ожирение",
    "diaBP": "повышенное диастолическое давление",
    "prevalentHyp": "артериальная гипертензия",
    "age": "возрастной фактор",
    "male": "биологический мужской пол",
    "heartRate": "тахикардия покоя",
    "prevalentStroke": "сосудистое событие в анамнезе",
    "BPMeds": "недостаточный контроль давления на терапии",
}
top_driver_desc = feat_names_ru.get(top_driver_key, top_driver_key)

with st.container(border=True):
    v_icon = "🟢" if p_chd < 10.0 else ("🟠" if p_chd < 20.0 else "🔴")
    st.markdown(f"#### {v_icon} {t('verdict_title')}")
    if p_chd < 10.0:
        verdict_msg = (
            f"У пациента регистрируется **{badge_cat.lower()}** ({p_chd:.1f}% за 10 лет). "
            f"Биологический сердечно-сосудистый возраст составляет **{heart_age:.1f} лет** "
            f"({'+' if ha_delta > 0 else ''}{ha_delta:.1f} к паспортному возрасту). "
            f"Оценка ожидаемой продолжительности жизни: **{life_exp:.1f} лет**."
        )
    else:
        verdict_msg = (
            f"У пациента регистрируется **{badge_cat.lower()}** ({p_chd:.1f}% за 10 лет). "
            f"Биологический сердечно-сосудистый возраст составляет **{heart_age:.1f} лет** "
            f"({'+' if ha_delta > 0 else ''}{ha_delta:.1f} к паспортному возрасту). "
            f"Ведущим управляемым фактором риска по оценке ИИ является **{top_driver_desc}** "
            f"(индивидуальный вклад: +{top_driver_val:.1f} процентных пунктов)."
        )
    st.write(verdict_msg)

st.divider()

# Priority Action Steps (TOP-3)
st.subheader(t("sec_priorities_title"))
st.caption(t("sec_priorities_sub"))

top_items = recs.get("top_interventions", [])
if top_items:
    step_cols = st.columns(len(top_items))
    for idx, item in enumerate(top_items):
        with step_cols[idx]:
            with st.container(border=True):
                st.markdown(f"#### #{idx+1}. {t(item['title_key'])}")
                st.caption(f"**Источник:** {item['source']}")
                st.write(t(item["rationale_key"]))

                st.markdown(f"**{t('rec_quant_effect')}**")
                st.markdown(
                    f"- 📉 **{t('rec_chd_reduction', val=abs(item['delta_chd_pp']))}**\n"
                    f"- 🫀 **{t('rec_heart_age_rejuvenation', val=abs(item['delta_heart_age']))}**\n"
                    f"- 🌱 **{t('rec_life_gain', val=abs(item['delta_life_exp']))}**"
                )

                st.info(f"💡 {t(item['action_key'])}")

    # Combined Cumulative Impact Container
    comb = recs.get("combined_impact")
    if comb:
        with st.container(border=True):
            st.markdown(f"### 🎯 {t('rec_combined_title')}")
            c_deltas = comb["deltas"]
            st.markdown(
                t(
                    "rec_combined_desc",
                    chd=abs(c_deltas["delta_chd_pp"]),
                    va=abs(c_deltas["delta_heart_age"]),
                    le=abs(c_deltas["delta_life_exp"]),
                )
            )
else:
    with st.container(border=True):
        st.success(f"🌟 {t('rec_maintenance_optimal')}")

st.divider()

# Deep-Dive Analytics Tabs
tab_sim, tab_xai, tab_bio, tab_traj, tab_exp = st.tabs([
    f"🎯 {t('tab_what_if')}",
    f"📊 {t('tab_xai')}",
    f"📋 {t('tab_biomarkers')}",
    f"📈 {t('tab_trajectory')}",
    f"📄 {t('tab_export')}",
])

with tab_sim:
    st.subheader(t("what_if_heading"))
    st.caption(t("what_if_desc"))

    st.markdown("##### Отметьте планируемые вмешательства для мгновенной симуляции:")
    col_c1, col_c2, col_c3 = st.columns(3)

    sim_changes = {}
    with col_c1:
        if cur_p.get("currentSmoker", 0) == 1:
            q_smoke = st.checkbox("🚭 Полный отказ от курения", value=True, key="sim_smoke")
            if q_smoke:
                sim_changes["currentSmoker"] = 0
                sim_changes["cigsPerDay"] = 0.0

        if cur_p.get("sysBP", 120) >= 130 or cur_p.get("diaBP", 80) >= 85:
            q_bp = st.checkbox("🩺 Целевое АД (< 125/78 мм рт.ст.)", value=True, key="sim_bp")
            if q_bp:
                sim_changes["sysBP"] = 125.0
                sim_changes["diaBP"] = 78.0
                sim_changes["prevalentHyp"] = 0

    with col_c2:
        if cur_p.get("totChol", 200) >= 200:
            q_chol = st.checkbox("🥗 Снижение холестерина на 20%", value=True, key="sim_chol")
            if q_chol:
                sim_changes["totChol"] = max(160.0, round(float(cur_p.get("totChol", 200)) * 0.8, 1))

        if cur_p.get("BMI", 25) >= 25.0:
            q_bmi = st.checkbox("⚖️ Оптимизация массы тела (ИМТ < 25)", value=False, key="sim_bmi")
            if q_bmi:
                sim_changes["BMI"] = 24.5

    with col_c3:
        if cur_p.get("glucose", 90) >= 100 or cur_p.get("diabetes", 0) == 1:
            q_glu = st.checkbox("🩸 Нормализация гликемии (< 5.6 ммоль/л)", value=False, key="sim_glu")
            if q_glu:
                sim_changes["glucose"] = 95.0

    if sim_changes:
        sim_dyn = simulate_intervention(cur_p, sim_changes)
        b_res = sim_dyn["base"]
        s_res = sim_dyn["simulated"]
        d_res = sim_dyn["deltas"]

        st.markdown("##### Сравнение: Исходный профиль vs Смоделированный результат")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            with st.container(border=True):
                st.caption(t("card_chd_title"))
                st.metric(
                    label="Новый риск ИБС",
                    value=f"{s_res['p_chd']*100:.1f}%",
                    delta=f"{d_res['delta_chd_pp']:.1f} п.п.",
                    delta_color="normal",
                )
        with col_s2:
            with st.container(border=True):
                st.caption(t("card_heart_age_title"))
                st.metric(
                    label="Новый возраст сердца",
                    value=f"{s_res['heart_age']:.1f} лет",
                    delta=f"{d_res['delta_heart_age']:.1f} лет",
                    delta_color="normal",
                )
        with col_s3:
            with st.container(border=True):
                st.caption(t("card_life_exp_title"))
                st.metric(
                    label="Новая оценка жизни",
                    value=f"{s_res['life_expectancy']:.1f} лет",
                    delta=f"+{d_res['delta_life_exp']:.1f} лет",
                    delta_color="normal",
                )
        with col_s4:
            with st.container(border=True):
                st.caption(t("card_dm_title"))
                st.metric(
                    label="Вероятность диабета",
                    value=f"{s_res['p_diabetes']*100:.1f}%",
                    delta=f"{d_res['delta_diabetes_pp']:.1f} п.п.",
                    delta_color="normal",
                )
    else:
        st.info("Отметьте вмешательства выше, чтобы посмотреть динамический перерасчет риска.")

with tab_xai:
    st.subheader(t("xai_heading"))
    st.caption(t("xai_desc"))

    xai_res = explain_prediction(cur_p, head="chd", m_steps=50)
    attrs = xai_res["attributions_pp"]

    feat_display_names = {
        "male": "Пол (мужской)",
        "age": "Возраст",
        "currentSmoker": "Статус курения",
        "cigsPerDay": "Интенсивность курения (сигарет/день)",
        "BPMeds": "Терапия гипертонии",
        "prevalentStroke": "Инсульт в анамнезе",
        "prevalentHyp": "Артериальная гипертензия",
        "totChol": "Общий холестерин",
        "sysBP": "Систолическое АД",
        "diaBP": "Диастолическое АД",
        "BMI": "Индекс массы тела (ИМТ)",
        "heartRate": "Пульс в покое",
        "glucose": "Глюкоза крови",
    }

    plot_data = []
    for k, v in attrs.items():
        if abs(v) >= 0.05:
            plot_data.append({
                "feature": feat_display_names.get(k, k),
                "contribution": v,
                "type": "Увеличивает риск" if v > 0 else "Снижает риск",
            })

    if plot_data:
        df_xai = pd.DataFrame(plot_data).sort_values(by="contribution", ascending=True)

        chart_xai = (
            alt.Chart(df_xai)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("contribution:Q", title="Вклад в 10-летний риск ИБС (процентные пункты, п.п.)"),
                y=alt.Y("feature:N", sort=None, title=None),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(domain=["Увеличивает риск", "Снижает риск"], range=["#ba1a1a", "#006c49"]),
                    title=None,
                ),
                tooltip=["feature", "contribution", "type"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_xai, width="stretch")
    else:
        st.info("Вклады факторов близки к эталонным оптимальным значениям.")

with tab_bio:
    st.subheader(t("biomarkers_heading"))
    st.caption(t("biomarkers_desc"))

    tc_val_display = from_mgdl(cur_p.get("totChol", 200), "totChol", units)
    glu_val_display = from_mgdl(cur_p.get("glucose", 90), "glucose", units)
    u_label = "ммоль/л" if units == "mmol" else "mg/dL"

    bp_cls = classify("bp", cur_p.get("sysBP", 120), sex=str(cur_p.get("male", 1)), secondary_value=cur_p.get("diaBP", 80))
    tc_cls = classify("totChol", cur_p.get("totChol", 200), sex=str(cur_p.get("male", 1)))
    glu_cls = classify("glucose", cur_p.get("glucose", 90), sex=str(cur_p.get("male", 1)))
    bmi_cls = classify("bmi", cur_p.get("BMI", 25), sex=str(cur_p.get("male", 1)))

    def status_badge(lvl):
        if lvl == "ok":
            return "🟢 Оптимально"
        elif lvl == "warn":
            return "🟡 Погранично"
        else:
            return "🔴 Повышено"

    bio_rows = [
        {
            "Биомаркер": "Артериальное давление (САД / ДАД)",
            "Ваше значение": f"{int(cur_p.get('sysBP', 120))} / {int(cur_p.get('diaBP', 80))} мм рт.ст.",
            "Целевая норма": bp_cls.get("normal_range", "< 130/80"),
            "Клинический статус": status_badge(bp_cls.get("level", "ok")),
            "Рекомендации / Источник": bp_cls.get("source", "ESC/ESH 2018"),
        },
        {
            "Биомаркер": f"Общий холестерин ({u_label})",
            "Ваше значение": f"{tc_val_display:.2f} {u_label}",
            "Целевая норма": "< 5.0 ммоль/л (< 200 мг/дл)",
            "Клинический статус": status_badge(tc_cls.get("level", "ok")),
            "Рекомендации / Источник": tc_cls.get("source", "NCEP ATP III"),
        },
        {
            "Биомаркер": f"Глюкоза натощак ({u_label})",
            "Ваше значение": f"{glu_val_display:.2f} {u_label}",
            "Целевая норма": "< 5.6 ммоль/л (< 100 мг/дл)",
            "Клинический статус": status_badge(glu_cls.get("level", "ok")),
            "Рекомендации / Источник": glu_cls.get("source", "WHO / ADA"),
        },
        {
            "Биомаркер": "Индекс массы тела (ИМТ)",
            "Ваше значение": f"{cur_p.get('BMI', 25):.1f} кг/м²",
            "Целевая норма": "18.5 – 24.9 кг/м²",
            "Клинический статус": status_badge(bmi_cls.get("level", "ok")),
            "Рекомендации / Источник": bmi_cls.get("source", "WHO Guidelines"),
        },
        {
            "Биомаркер": "Статус курения табака",
            "Ваше значение": f"Курит ({int(cur_p.get('cigsPerDay', 0))} сигарет/сут)" if cur_p.get("currentSmoker") else "Не курит",
            "Целевая норма": "Полный отказ от табака",
            "Клинический статус": "🔴 Фактор риска" if cur_p.get("currentSmoker") else "🟢 Норма",
            "Рекомендации / Источник": "WHO FCTC / ESC 2021",
        },
        {
            "Биомаркер": "Пульс в покое (ЧСС)",
            "Ваше значение": f"{int(cur_p.get('heartRate', 72))} уд/мин",
            "Целевая норма": "60 – 80 уд/мин",
            "Клинический статус": "🟢 Оптимально" if 60 <= cur_p.get("heartRate", 72) <= 80 else "🟡 Вне диапазона",
            "Рекомендации / Источник": "ESC Guidelines",
        },
    ]
    st.table(pd.DataFrame(bio_rows))

with tab_traj:
    st.subheader(t("traj_heading"))
    st.caption(t("traj_desc"))

    opt_changes = recs.get("combined_impact", {}).get("changes") if recs.get("has_interventions") else None
    traj_records = compute_risk_trajectory(
        cur_p,
        years_ahead=(0, 2, 5, 8, 10, 12, 15, 18, 20),
        improved_changes=opt_changes,
    )

    df_traj = pd.DataFrame(traj_records)
    df_traj["scenario_label"] = df_traj["scenario"].map({
        "current": t("traj_scenario_current"),
        "improved": t("traj_scenario_improved"),
    })

    chart_traj = (
        alt.Chart(df_traj)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("age:Q", title="Возраст (лет)"),
            y=alt.Y("p_chd_percent:Q", title="10-летний риск ИБС (%)"),
            color=alt.Color(
                "scenario_label:N",
                scale=alt.Scale(
                    domain=[t("traj_scenario_current"), t("traj_scenario_improved")],
                    range=["#ba1a1a", "#006c49"],
                ),
                title="Сценарий",
            ),
            tooltip=["age", "p_chd_percent", "heart_age", "life_expectancy", "scenario_label"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_traj, width="stretch")

with tab_exp:
    st.subheader(t("export_heading"))
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.container(border=True):
            st.markdown("#### 📄 Клинический печатный отчёт")
            st.write("Формирует структурированный медицинский документ с результатами скрининга, графиком приоритетов и врачебным дисклеймером.")
            html_content = generate_html_report(cur_p, prediction, recs, lang=lang, units=units)
            st.download_button(
                label=t("export_html_btn"),
                data=html_content,
                file_name=f"MedAI_Report_{int(cur_p.get('age', 50))}_{'M' if cur_p.get('male', 1) else 'F'}.html",
                mime="text/html",
                width="stretch",
            )

    with exp_col2:
        with st.container(border=True):
            st.markdown("#### 🏥 Экспорт в стандарте HL7 FHIR R4")
            st.write("Интероперабельный пакет для интеграции с медицинскими информационными системами (МИС) с валидными кодами LOINC.")
            fhir_bundle = generate_fhir_bundle(cur_p, prediction, lang=lang)
            fhir_json = json.dumps(fhir_bundle, indent=2, ensure_ascii=False)
            st.download_button(
                label=t("export_fhir_btn"),
                data=fhir_json,
                file_name="fhir_bundle_screening.json",
                mime="application/json",
                width="stretch",
            )
