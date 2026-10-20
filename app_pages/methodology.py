"""
app_pages/methodology.py — Scientific Methodology, Real ML Evaluation Metrics & AI Audit.
"""
import json
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.config import MODELS_DIR
from core.i18n import get_current_language, t

lang = get_current_language()
st.title(f"📊 {t('meth_title', lang=lang)}")

# 1. Medical Disclaimer
with st.container(border=True):
    st.warning(f"⚠️ **{t('meth_disclaimer_title', lang=lang)}:** {t('meth_disclaimer_body', lang=lang)}")

# Load Real Audited Metrics from models/metrics.json
metrics_file = MODELS_DIR / "metrics.json"
metrics = {}
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

# 2. Section: Training Cohort & Data
st.subheader(t("meth_sec_data", lang=lang))
with st.container(border=True):
    st.markdown(t("meth_cohort_desc", lang=lang))

# 3. Section: Neural Network Architecture & Real Metrics
st.subheader(t("meth_sec_model", lang=lang))

tab_metrics1, tab_metrics2, tab_metrics3 = st.tabs([
    t("method_tab_loss", lang=lang),
    t("method_tab_metrics", lang=lang),
    t("method_tab_calibration", lang=lang),
])

with tab_metrics1:
    st.markdown(f"##### {t('method_loss_heading', lang=lang)}")
    hist = metrics.get("training", {}).get("history", {})
    if hist and "epochs" in hist:
        lbl_epoch = t("method_loss_epoch", lang=lang)
        lbl_sample = t("method_loss_sample", lang=lang)
        lbl_train = t("method_loss_train", lang=lang)
        lbl_val = t("method_loss_val", lang=lang)

        df_loss = pd.DataFrame({
            lbl_epoch: hist["epochs"],
            lbl_train: hist["train_loss"],
            lbl_val: hist["val_loss"],
        }).melt(
            id_vars=[lbl_epoch],
            value_vars=[lbl_train, lbl_val],
            var_name=lbl_sample,
            value_name="BCE Loss",
        )

        chart_loss = (
            alt.Chart(df_loss)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X(f"{lbl_epoch}:Q", title=t("method_loss_axis_x", lang=lang)),
                y=alt.Y("BCE Loss:Q", title=t("method_loss_axis_y", lang=lang)),
                color=alt.Color(
                    f"{lbl_sample}:N",
                    scale=alt.Scale(domain=[lbl_train, lbl_val], range=["#006c49", "#ba1a1a"]),
                    title=lbl_sample,
                ),
                tooltip=[lbl_epoch, "BCE Loss", lbl_sample],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_loss, width="stretch")
        best_ep = metrics.get("training", {}).get("best_epoch", "N/A")
        best_vl = metrics.get("training", {}).get("best_val_loss", "N/A")
        st.caption(t("method_loss_early_stop", lang=lang, ep=best_ep, loss=best_vl))
    else:
        st.info(t("method_loss_no_data", lang=lang))

with tab_metrics2:
    st.markdown(f"##### {t('method_metrics_heading', lang=lang)}")

    chd_m = metrics.get("chd", {})
    dm_m = metrics.get("diabetes", {})

    nn_chd_auc = chd_m.get("nn", {}).get("auc", 0.0)
    nn_chd_ci = chd_m.get("nn", {}).get("auc_ci_95", [0.0, 0.0])
    lr_chd_auc = chd_m.get("logistic_regression", {}).get("auc", 0.0)
    dag_chd_auc = chd_m.get("dagostino_reference", {}).get("auc", 0.0)

    nn_dm_auc = dm_m.get("nn", {}).get("auc", 0.0)
    nn_dm_ci = dm_m.get("nn", {}).get("auc_ci_95", [0.0, 0.0])
    lr_dm_auc = dm_m.get("logistic_regression", {}).get("auc", 0.0)

    col_task = t("method_col_task", lang=lang)
    col_nn = t("method_col_nn", lang=lang)
    col_lr = t("method_col_lr", lang=lang)
    col_dag = t("method_col_dag", lang=lang)
    col_brier = t("method_col_brier", lang=lang)

    eval_table = [
        {
            col_task: t("method_task_chd", lang=lang),
            col_nn: f"ROC-AUC = {nn_chd_auc:.3f} [{nn_chd_ci[0]:.3f}, {nn_chd_ci[1]:.3f}]",
            col_lr: f"ROC-AUC = {lr_chd_auc:.3f}",
            col_dag: f"ROC-AUC = {dag_chd_auc:.3f}",
            col_brier: f"{chd_m.get('nn', {}).get('brier', 0.0):.4f}",
        },
        {
            col_task: t("method_task_dm", lang=lang),
            col_nn: f"ROC-AUC = {nn_dm_auc:.3f} [{nn_dm_ci[0]:.3f}, {nn_dm_ci[1]:.3f}]",
            col_lr: f"ROC-AUC = {lr_dm_auc:.3f}",
            col_dag: "—",
            col_brier: f"{dm_m.get('nn', {}).get('brier', 0.0):.4f}",
        },
    ]
    st.table(pd.DataFrame(eval_table))

    st.markdown(f"##### {t('meth_th_heading', lang=lang)}")
    th10 = chd_m.get("nn", {}).get("threshold_10", {})
    th20 = chd_m.get("nn", {}).get("threshold_20", {})

    col_th1, col_th2 = st.columns(2)
    with col_th1:
        with st.container(border=True):
            st.markdown(f"###### {t('meth_th10_title', lang=lang)}")
            st.write(f"- **{t('meth_sens', lang=lang)}:** {th10.get('sensitivity', 0.0)*100:.1f}%")
            st.write(f"- **{t('meth_spec', lang=lang)}:** {th10.get('specificity', 0.0)*100:.1f}%")
            st.write(f"- **{t('meth_ppv', lang=lang)}:** {th10.get('ppv', 0.0)*100:.1f}%")
            st.caption(t("meth_th10_sub", lang=lang))

    with col_th2:
        with st.container(border=True):
            st.markdown(f"###### {t('meth_th20_title', lang=lang)}")
            st.write(f"- **{t('meth_sens', lang=lang)}:** {th20.get('sensitivity', 0.0)*100:.1f}%")
            st.write(f"- **{t('meth_spec', lang=lang)}:** {th20.get('specificity', 0.0)*100:.1f}%")
            st.write(f"- **{t('meth_ppv', lang=lang)}:** {th20.get('ppv', 0.0)*100:.1f}%")
            st.caption(t("meth_th20_sub", lang=lang))

with tab_metrics3:
    st.markdown(f"##### {t('meth_calib_title', lang=lang)}")
    cal = chd_m.get("calibration", {})
    if "prob_pred" in cal and "prob_true" in cal:
        lbl_pred = t("method_calib_pred_prob", lang=lang)
        lbl_true = t("method_calib_true_freq", lang=lang)

        df_cal = pd.DataFrame({
            lbl_pred: cal["prob_pred"],
            lbl_true: cal["prob_true"],
        })
        diag = pd.DataFrame({
            lbl_pred: [0.0, 0.5],
            lbl_true: [0.0, 0.5],
        })

        line_model = (
            alt.Chart(df_cal)
            .mark_line(point=True, color="#006c49", strokeWidth=2.5)
            .encode(
                x=alt.X(f"{lbl_pred}:Q", scale=alt.Scale(domain=[0, 0.5])),
                y=alt.Y(f"{lbl_true}:Q", scale=alt.Scale(domain=[0, 0.5])),
                tooltip=[lbl_pred, lbl_true],
            )
        )
        line_diag = (
            alt.Chart(diag)
            .mark_line(strokeDash=[4, 4], color="#888888")
            .encode(
                x=f"{lbl_pred}:Q",
                y=f"{lbl_true}:Q",
            )
        )
        st.altair_chart((line_diag + line_model).properties(height=320), width="stretch")
        st.caption(t("meth_calib_caption", lang=lang))

# 4. Section: Clinical Formulas
st.subheader(t("meth_sec_clinical", lang=lang))
with st.container(border=True):
    st.markdown(t("meth_clinical_desc", lang=lang))

# 5. Section: Limitations
st.subheader(t("meth_sec_limitations", lang=lang))
with st.container(border=True):
    st.markdown(t("meth_limits_desc", lang=lang))
