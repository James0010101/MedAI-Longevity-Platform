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
st.title(f"📊 {t('meth_title')}")

# 1. Medical Disclaimer
with st.container(border=True):
    st.warning(f"⚠️ **{t('meth_disclaimer_title')}:** {t('meth_disclaimer_body')}")

# Load Real Audited Metrics from models/metrics.json
metrics_file = MODELS_DIR / "metrics.json"
metrics = {}
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

# 2. Section: Training Cohort & Data
st.subheader(t("meth_sec_data"))
with st.container(border=True):
    st.markdown("""
    **Когорта обучения:** Framingham Heart Study (публичная верифицированная учебная когорта Национального института сердца, легких и крови США — NHLBI).
    - **Общий объем выборки:** 4 240 участников, 16 исходных клинических признаков.
    - **Стратифицированное разбиение:** 70% Train (2 968 чел.), 15% Validation (636 чел.), 15% Test (636 чел.).
    - **Распространенность исходов на тесте:** ИБС (10-летний инцидент) — **15.25%**, сахарный диабет — **2.83%**.
    - **Предобработка:** Заполнение пропусков медианами, рассчитанными *исключительно* на обучающей выборке. Стандартизация (Z-score) зафиксирована в метаданных модели.
    - **Важное клиническое ограничение:** Когорта отражает популяцию США второй половины XX века. Для внедрения в клиническую практику здравоохранения Узбекистана необходима локальная калибровка на репрезентативной национальной выборке.
    """)

# 3. Section: Neural Network Architecture & Real Metrics
st.subheader(t("meth_sec_model"))

tab_metrics1, tab_metrics2, tab_metrics3 = st.tabs([
    "📈 Кривые обучения (Loss)",
    "🎯 Метрики качества (ROC-AUC & Brier)",
    "⚖️ Калибровка вероятностей",
])

with tab_metrics1:
    st.markdown("##### Реальная динамика функции потерь при обучении (без хардкода)")
    hist = metrics.get("training", {}).get("history", {})
    if hist and "epochs" in hist:
        df_loss = pd.DataFrame({
            "Эпоха": hist["epochs"],
            "Train Loss": hist["train_loss"],
            "Validation Loss": hist["val_loss"],
        }).melt(id_vars=["Эпоха"], value_vars=["Train Loss", "Validation Loss"], var_name="Выборка", value_name="BCE Loss")

        chart_loss = (
            alt.Chart(df_loss)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("Эпоха:Q", title="Эпоха обучения"),
                y=alt.Y("BCE Loss:Q", title="Функция потерь (Binary Cross-Entropy)"),
                color=alt.Color(
                    "Выборка:N",
                    scale=alt.Scale(domain=["Train Loss", "Validation Loss"], range=["#006c49", "#ba1a1a"]),
                ),
                tooltip=["Эпоха", "BCE Loss", "Выборка"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_loss, width="stretch")
        best_ep = metrics.get("training", {}).get("best_epoch", "N/A")
        best_vl = metrics.get("training", {}).get("best_val_loss", "N/A")
        st.caption(f"Ранняя остановка (Early Stopping): лучший чекпоинт зафиксирован на эпохе {best_ep} с валидационными потерями {best_vl}.")
    else:
        st.info("Данные истории обучения отсутствуют.")

with tab_metrics2:
    st.markdown("##### Сравнение предсказательной способности на отложенной тестовой выборке (636 чел.)")

    chd_m = metrics.get("chd", {})
    dm_m = metrics.get("diabetes", {})

    nn_chd_auc = chd_m.get("nn", {}).get("auc", 0.0)
    nn_chd_ci = chd_m.get("nn", {}).get("auc_ci_95", [0.0, 0.0])
    lr_chd_auc = chd_m.get("logistic_regression", {}).get("auc", 0.0)
    dag_chd_auc = chd_m.get("dagostino_reference", {}).get("auc", 0.0)

    nn_dm_auc = dm_m.get("nn", {}).get("auc", 0.0)
    nn_dm_ci = dm_m.get("nn", {}).get("auc_ci_95", [0.0, 0.0])
    lr_dm_auc = dm_m.get("logistic_regression", {}).get("auc", 0.0)

    eval_table = [
        {
            "Целевая задача": "10-летний риск ИБС (CHD)",
            "Модель MedAI (Нейросеть)": f"ROC-AUC = {nn_chd_auc:.3f} [{nn_chd_ci[0]:.3f}, {nn_chd_ci[1]:.3f}]",
            "Базовый LogReg": f"ROC-AUC = {lr_chd_auc:.3f}",
            "Референс D'Agostino 2008": f"ROC-AUC = {dag_chd_auc:.3f}",
            "Brier Score (калибровка)": f"{chd_m.get('nn', {}).get('brier', 0.0):.4f}",
        },
        {
            "Целевая задача": "Скрининг сахарного диабета",
            "Модель MedAI (Нейросеть)": f"ROC-AUC = {nn_dm_auc:.3f} [{nn_dm_ci[0]:.3f}, {nn_dm_ci[1]:.3f}]",
            "Базовый LogReg": f"ROC-AUC = {lr_dm_auc:.3f}",
            "Референс D'Agostino 2008": "—",
            "Brier Score (калибровка)": f"{dm_m.get('nn', {}).get('brier', 0.0):.4f}",
        },
    ]
    st.table(pd.DataFrame(eval_table))

    st.markdown("##### Клинические операционные характеристики при порогах риска")
    th10 = chd_m.get("nn", {}).get("threshold_10", {})
    th20 = chd_m.get("nn", {}).get("threshold_20", {})

    col_th1, col_th2 = st.columns(2)
    with col_th1:
        with st.container(border=True):
            st.markdown("###### Порог скрининга: 10% (Промежуточный риск)")
            st.write(f"- **Чувствительность (Sensitivity):** {th10.get('sensitivity', 0.0)*100:.1f}%")
            st.write(f"- **Специфичность (Specificity):** {th10.get('specificity', 0.0)*100:.1f}%")
            st.write(f"- **PPV (Прогностичность полож. рез.):** {th10.get('ppv', 0.0)*100:.1f}%")
            st.caption("Обеспечивает высокую чувствительность для первичного скрининга населения.")

    with col_th2:
        with st.container(border=True):
            st.markdown("###### Порог скрининга: 20% (Высокий клинический риск)")
            st.write(f"- **Чувствительность (Sensitivity):** {th20.get('sensitivity', 0.0)*100:.1f}%")
            st.write(f"- **Специфичность (Specificity):** {th20.get('specificity', 0.0)*100:.1f}%")
            st.write(f"- **PPV (Прогностичность полож. рез.):** {th20.get('ppv', 0.0)*100:.1f}%")
            st.caption("Высокая специфичность: отбирает пациентов для интенсивной медикаментозной терапии.")

with tab_metrics3:
    st.markdown("##### Калибровочная кривая модели ИБС")
    cal = chd_m.get("calibration", {})
    if "prob_pred" in cal and "prob_true" in cal:
        df_cal = pd.DataFrame({
            "Предсказанная вероятность": cal["prob_pred"],
            "Наблюдаемая доля событий": cal["prob_true"],
        })
        diag = pd.DataFrame({
            "Предсказанная вероятность": [0.0, 0.5],
            "Наблюдаемая доля событий": [0.0, 0.5],
        })

        line_model = (
            alt.Chart(df_cal)
            .mark_line(point=True, color="#006c49", strokeWidth=2.5)
            .encode(
                x=alt.X("Предсказанная вероятность:Q", scale=alt.Scale(domain=[0, 0.5])),
                y=alt.Y("Наблюдаемая доля событий:Q", scale=alt.Scale(domain=[0, 0.5])),
                tooltip=["Предсказанная вероятность", "Наблюдаемая доля событий"],
            )
        )
        line_diag = (
            alt.Chart(diag)
            .mark_line(strokeDash=[4, 4], color="#888888")
            .encode(
                x="Предсказанная вероятность:Q",
                y="Наблюдаемая доля событий:Q",
            )
        )
        st.altair_chart((line_diag + line_model).properties(height=320), width="stretch")
        st.caption("Пунктирная серая линия — идеальная калибровка. Близость зеленой линии отражает отсутствие систематического завышения или занижения вероятностей.")

# 4. Section: Clinical Formulas
st.subheader(t("meth_sec_clinical"))
with st.container(border=True):
    st.markdown("""
    **1. Формула сердечно-сосудистого возраста (D'Agostino et al., Circulation 2008):**
    В отличие от старых прототипов, где возраст предсказывался нейросетью и «схлопывался» в константу,
    в MedAI внедрена строгая аналитическая замкнутая формула:
    $$\\text{Heart Age} = \\exp\\left(\\frac{L_{\\text{patient}} - C_{\\text{ref}}}{\\beta_{\\text{age}}}\\right)$$
    где $C_{\\text{ref}}$ — не-возрастные коэффициенты для эталонного оптимального профиля (САД 125, ОХС 180 мг/дл, HDL 45 мг/дл, не курит, без диабета).
    Для идеального профиля любого возраста математически доказано: $\\text{Heart Age} \\equiv \\text{Chronological Age}$.

    **2. Оценка продолжительности жизни по таблицам ВОЗ для Узбекистана (2023):**
    Используются официальные данные таблиц смертности ВОЗ (Global Health Estimates):
    $$\\text{Оценка продолжительности жизни} = \\text{Возраст} + e_x(\\text{Heart Age})$$
    где $e_x$ — оставшаяся продолжительность жизни для конкретного пола по официальной национальной демографической сетке.
    """)

# 5. Section: Limitations
st.subheader(t("meth_sec_limitations"))
with st.container(border=True):
    st.markdown("""
    - **Отсутствие местных когортных исследований:** Обучение произведено на когорте Framingham (США). Для сертификации в Минздраве Узбекистана требуется валидация на выборке жителей РУз.
    - **Ограниченность липидной панели:** Учебная выборка содержит только общий холестерин без прямого измерения ЛПНП и триглицеридов.
    - **Статус скрининга:** Система предназначена для стратификации рисков и санитарно-просветительской работы, но не для автономной постановки клинических диагнозов.
    """)
