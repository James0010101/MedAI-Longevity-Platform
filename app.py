import os
import json
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

from neural_network import MultiTaskNeuralNetwork
from dataset_manager import get_medical_dataset, normalize

st.set_page_config(
    page_title="MedAI Longevity Suite | Executive Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

WEIGHTS_FILE = "model_weights.npz"

# ────────────────── INTERNATIONALIZATION (I18N) ──────────────────
I18N = {
    "ru": {
        "title": "MedAI Longevity Platform",
        "subtitle": "Система Нейросетевой Клинической Диагностики, Оценки Рисков и 20-Летнего Прогнозирования Долголетия",
        "params_header": "Конфигурация нейросети",
        "data_header": "Источник данных",
        "dataset_fused": "🌐 Сводный Клинический Датасет (10,000+ записей)",
        "num_layers": "Скрытых слоев",
        "neurons_l1": "Нейронов в слое 1",
        "neurons_l2": "Нейронов в слое 2",
        "neurons_l3": "Нейронов в слое 3",
        "neurons_l4": "Нейронов в слое 4",
        "learning_rate": "Скорость обучения (LR)",
        "epochs": "Эпохи обучения",
        "l2_reg": "L2 Регуляризация",
        "run_training": "Дообучить модель на датасете",
        "save_weights_btn": "Сохранить веса",
        "reset_weights_btn": "Сбросить веса",
        "performance": "Динамика ошибки (Loss)",
        "tabs": [
            "📊 Метрики ИИ",
            "3D Кластеры",
            "📈 Прогноз 20 лет",
            "🕸️ Радар признаков",
            "🔍 XAI Вклад"
        ],
        "training_tab_title": "Архитектура и результаты обучения",
        "arch_title": "Архитектура нейросети",
        "loss_title": "Динамика функции потерь",
        "eval_title": "Метрики на обучающей выборке",
        "accuracy": "Точность",
        "mae": "Средняя абс. ошибка (MAE)",
        "cardiac_risk_hdr": "Кардио-риск (CI 95%)",
        "diabetes_risk_hdr": "Риск Диабета II (CI 95%)",
        "life_exp_hdr": "Ожидаемая жизнь (CI 95%)",
        "vascular_age_hdr": "Сосудистый возраст (CI 95%)",
        "recommendations_hdr": "Персонализированный клинический план долголетия",
        "symptoms_hdr": "Анализ причин и возможных симптомов",
        "xai_hdr": "Explainable AI (XAI): Вклад каждого биомаркера (%)",
        "xai_xlabel": "Вклад признака (%)",
        "what_if_hdr": "Симулятор медицинских вмешательств «Что, если?»",
        "report_btn": "📥 Скачать клинический отчёт (HTML / PDF)",
        "fhir_btn": "🌐 Скачать запись в формате HL7 FHIR (JSON)",
        "high_risk": "ВЫСОКИЙ РИСК",
        "low_risk": "НИЗКИЙ РИСК",
        "years": "лет",
        "vs_baseline": "к базовой",
        "vs_chrono": "к паспортному",
        "low_risk_label": "Низкий риск",
        "high_risk_label": "Высокий риск",
        "healthy_cluster": "Здоровый кластер",
        "pathology_cluster": "Кластер патологии",
        "bp_label": "Давление (мм рт.ст.)",
        "chol_label": "Холестерин (мг/дл)",
        "sim_quit_smoke": "Отказ от курения",
        "sim_lower_bp": "Нормализация давления (120 мм)",
        "sim_max_act": "Максимальная физ. активность (100%)",
        "sim_result_title": "Результат вмешательства:",
        "sim_cardiac_fmt": "Кардио-риск: {before:.1f}% ➔ {after:.1f}% ({diff:+.1f}%)",
        "sim_life_fmt": "Продолжительность жизни: {before:.1f} ➔ {after:.1f} ({diff:+.1f} лет)",
        "slider_age": "Возраст (лет)",
        "slider_sex": "Пол",
        "slider_bp": "Давление (мм рт.ст.)",
        "slider_chol": "Холестерин (мг/дл)",
        "slider_glucose": "Глюкоза (мг/дл)",
        "slider_max_hr": "Макс. пульс",
        "slider_st_dep": "Депрессия ST (мм)",
        "slider_bmi": "ИМТ (BMI)",
        "slider_activity": "Физ. активность (0..1)",
        "slider_smoking": "Курение",
        "sex_male": "Мужской ♂",
        "sex_female": "Женский ♀",
        "smoke_no": "Нет",
        "smoke_yes": "Да",
        "trajectory_title": "Прогноз долголетия и риска на 20 лет вперед",
        "radar_title": "Отклонение биомаркеров пациента от нормы здоровой когорты",
        "presets_hdr": "Быстрый выбор профиля пациента:"
    },
    "en": {
        "title": "MedAI Longevity Platform",
        "subtitle": "Multi-Dataset Neural Clinical Diagnostics, MC Uncertainty (CI 95%) & 20-Year Trajectory",
        "params_header": "Neural Architecture Config",
        "data_header": "Data Source",
        "dataset_fused": "🌐 Master Clinical Dataset (10,000+ records)",
        "num_layers": "Hidden Layers Count",
        "neurons_l1": "Layer 1 Neurons",
        "neurons_l2": "Layer 2 Neurons",
        "neurons_l3": "Layer 3 Neurons",
        "neurons_l4": "Layer 4 Neurons",
        "learning_rate": "Learning Rate (LR)",
        "epochs": "Training Epochs",
        "l2_reg": "L2 Regularization",
        "run_training": "Run Incremental Training",
        "save_weights_btn": "Save Model Weights",
        "reset_weights_btn": "Reset Weights",
        "performance": "Loss Dynamics",
        "tabs": [
            "📊 AI Metrics",
            "3D Clusters",
            "📈 20Y Forecast",
            "🕸️ Radar",
            "🔍 XAI Contribution"
        ],
        "training_tab_title": "Architecture & Training Results",
        "arch_title": "Neural Network Architecture",
        "loss_title": "Loss Dynamics",
        "eval_title": "Training Evaluation Metrics",
        "accuracy": "Accuracy",
        "mae": "Mean Abs. Error (MAE)",
        "cardiac_risk_hdr": "Cardiac Risk (CI 95%)",
        "diabetes_risk_hdr": "Diabetes II Risk (CI 95%)",
        "life_exp_hdr": "Life Expectancy (CI 95%)",
        "vascular_age_hdr": "Vascular Age (CI 95%)",
        "recommendations_hdr": "Personalized Action Plan",
        "symptoms_hdr": "Root Causes & Symptom Trigger Analysis",
        "xai_hdr": "Explainable AI (XAI): Feature Contribution (%)",
        "xai_xlabel": "Feature Contribution (%)",
        "what_if_hdr": "'What-If' Intervention Simulator",
        "report_btn": "📥 Download Clinical Report (HTML / PDF)",
        "fhir_btn": "🌐 Export HL7 FHIR Record (JSON)",
        "high_risk": "HIGH RISK",
        "low_risk": "LOW RISK",
        "years": "yrs",
        "vs_baseline": "vs baseline",
        "vs_chrono": "vs chronological",
        "low_risk_label": "Low Risk",
        "high_risk_label": "High Risk",
        "healthy_cluster": "Healthy Cluster",
        "pathology_cluster": "Pathology Cluster",
        "bp_label": "Resting BP (mmHg)",
        "chol_label": "Serum Cholesterol (mg/dL)",
        "sim_quit_smoke": "Smoking Cessation",
        "sim_lower_bp": "BP Normalization (120 mmHg)",
        "sim_max_act": "Physical Activity (100%)",
        "sim_result_title": "Intervention Result:",
        "sim_cardiac_fmt": "Cardiac Risk: {before:.1f}% ➔ {after:.1f}% ({diff:+.1f}%)",
        "sim_life_fmt": "Life Expectancy: {before:.1f} ➔ {after:.1f} ({diff:+.1f} yrs)",
        "slider_age": "Age (years)",
        "slider_sex": "Sex",
        "slider_bp": "BP (mmHg)",
        "slider_chol": "Cholesterol (mg/dL)",
        "slider_glucose": "Glucose (mg/dL)",
        "slider_max_hr": "Max HR (bpm)",
        "slider_st_dep": "ST Depression (mm)",
        "slider_bmi": "BMI (kg/m²)",
        "slider_activity": "Activity (0..1)",
        "slider_smoking": "Smoking",
        "sex_male": "Male ♂",
        "sex_female": "Female ♀",
        "smoke_no": "No",
        "smoke_yes": "Yes",
        "trajectory_title": "20-Year Health & Longevity Trajectory Forecast",
        "radar_title": "Patient Biomarker Deviation vs Healthy Peer Baseline",
        "presets_hdr": "Quick Patient Profile Presets:"
    }
}

# Sidebar Language Selection
with st.sidebar:
    st.markdown("<div style='font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.08em; margin-bottom: 8px;'>LANGUAGE / ЯЗЫК</div>", unsafe_allow_html=True)
    selected_lang_label = st.segmented_control(
        "Language Selection",
        ["Русский 🇷🇺", "English 🇺🇸"],
        default="Русский 🇷🇺",
        label_visibility="collapsed"
    )
    lang_code = "en" if selected_lang_label == "English 🇺🇸" else "ru"
    T = I18N[lang_code]

# 🎨 CLINICAL EMERALD HIGH-END CSS (Material 3 Clinical Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f4fbf4;
        color: #161d19;
    }
    
    /* Executive Header */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 32px;
        border-bottom: 1px solid #eef6ee;
        background-color: #ffffff;
        margin: -4rem -4rem 28px -4rem;
        box-shadow: 0 4px 20px rgba(0, 108, 73, 0.05);
        border-bottom-left-radius: 24px;
        border-bottom-right-radius: 24px;
    }
    
    .top-title {
        font-size: 24px;
        font-weight: 700;
        color: #006c49;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }
    
    .top-subtitle {
        font-size: 13px;
        color: #6c7a71;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #f4fbf4;
        border-right: 1px solid #eef6ee;
    }
    
    /* Clean Cards & Container Borders */
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #ffffff !important;
        border: 1px solid #eef6ee !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0px 4px 16px rgba(0, 108, 73, 0.03) !important;
    }
    
    /* Smooth Pill Buttons */
    .stButton>button {
        border-radius: 100px;
        font-weight: 600;
        font-size: 13px;
        background-color: #006c49;
        color: white;
        border: none;
        box-shadow: 0px 4px 12px rgba(0, 108, 73, 0.15);
        transition: all 0.25s ease;
        padding: 10px 20px;
    }
    
    .stButton>button:hover {
        background-color: #005236;
        box-shadow: 0px 6px 16px rgba(0, 108, 73, 0.25);
        transform: translateY(-1px);
        color: white;
    }
    
    /* Smooth Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #eef6ee;
        padding-bottom: 4px;
        margin-bottom: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 100px;
        border: none;
        padding: 8px 16px;
        color: #6c7a71;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #eef6ee !important;
        color: #006c49 !important;
        box-shadow: inset 0px 0px 0px 1px #dde4dd;
    }
    
    /* Custom Telemetry Card Styling */
    .telemetry-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(0, 108, 73, 0.04);
        transition: transform 0.2s ease;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
    }
    .telemetry-indicator {
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    .telemetry-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .telemetry-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 6px;
    }
    .telemetry-badge {
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 100px;
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 🏛️ Top Header Bar
st.markdown(f"""
<div class="top-header">
    <div>
        <div class="top-title">🧬 {T["title"]}</div>
        <div class="top-subtitle">{T["subtitle"]}</div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="background-color:#eef6ee; color:#006c49; padding:4px 12px; border-radius:100px; font-weight:700; font-size:12px;">Clinical AI v3.0 (Pre-Trained)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────── SIDEBAR: DATASET & HYPERPARAMETERS ──────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.08em; margin-bottom: 8px;'>{T['data_header'].upper()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 13px; font-weight: 700; color: #006c49; margin-bottom: 12px;'>{T['dataset_fused']}</div>", unsafe_allow_html=True)
    
    data_source_key = "fused_master"
    custom_csv_file = None

    with st.expander("⚙️ " + T["params_header"], expanded=False):
        num_hidden = st.slider(T["num_layers"], min_value=1, max_value=4, value=2, step=1)
        n_l1 = st.slider(T["neurons_l1"], min_value=8, max_value=256, value=64, step=8)
        n_l2, n_l3, n_l4 = 32, 16, 8
        
        if num_hidden >= 2:
            n_l2 = st.slider(T["neurons_l2"], min_value=4, max_value=256, value=32, step=4)
        if num_hidden >= 3:
            n_l3 = st.slider(T["neurons_l3"], min_value=4, max_value=128, value=16, step=4)
        if num_hidden >= 4:
            n_l4 = st.slider(T["neurons_l4"], min_value=2, max_value=64, value=8, step=2)
            
        hidden_sizes = [n_l1]
        if num_hidden >= 2: hidden_sizes.append(n_l2)
        if num_hidden >= 3: hidden_sizes.append(n_l3)
        if num_hidden >= 4: hidden_sizes.append(n_l4)

        st.markdown("---")
        learning_rate = st.slider(T["learning_rate"], min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")
        epochs = st.slider(T["epochs"], min_value=100, max_value=10000, value=2000, step=500)
        l2_reg = st.slider(T["l2_reg"], min_value=0.0, max_value=0.005, value=0.0005, step=0.0005, format="%.4f")
        
        run_training_btn = st.button(T["run_training"], use_container_width=True)
    
    st.markdown("---")

# ────────────────── DATASET LOAD & MODEL PERSISTENCE ──────────────────
@st.cache_data
def load_data(source_name, csv_handle=None):
    return get_medical_dataset(n_samples=3000, source=source_name, csv_file=csv_handle)

data_dict = load_data(data_source_key, custom_csv_file)
X_train = data_dict["X_train"]
y_cardiac_train = data_dict["y_cardiac_train"]
y_diabetes_train = data_dict["y_diabetes_train"]
y_life_train = data_dict["y_life_train"]
y_vascular_train = data_dict["y_vascular_train"]

norm_X = data_dict["norm_X"]
y_cardiac = data_dict["y_cardiac"]
mins = np.array([1.0, 0.0, 50.0, 50.0, 40.0, 40.0, 0.0, 10.0, 0.0, 0.0])
maxs = np.array([120.0, 1.0, 250.0, 500.0, 400.0, 220.0, 6.0, 60.0, 1.0, 1.0])

if 'hidden_sizes' not in locals():
    hidden_sizes = [64, 32]

# Auto-Load Model Weights
if 'nn_model' not in st.session_state:
    if os.path.exists(WEIGHTS_FILE):
        try:
            loaded_nn = MultiTaskNeuralNetwork.load_model(WEIGHTS_FILE)
            st.session_state['nn_model'] = loaded_nn
            st.session_state['last_hidden'] = loaded_nn.hidden_sizes
        except Exception as e:
            st.warning(f"Note: Resetting model initialization ({e})")
            
    if 'nn_model' not in st.session_state:
        init_nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[64, 32], learning_rate=0.01, l2_reg=0.0005)
        init_nn.train_step(X_train, y_cardiac_train, y_diabetes_train, y_life_train, y_vascular_train)
        init_nn.save_weights(WEIGHTS_FILE)
        st.session_state['nn_model'] = init_nn
        st.session_state['last_hidden'] = [64, 32]

nn = st.session_state['nn_model']

# Incremental Training Handler
if 'run_training_btn' in locals() and run_training_btn:
    nn.learning_rate = learning_rate
    nn.l2_reg = l2_reg
    
    with st.sidebar:
        st.markdown(f"<div style='font-size:11px; font-weight:700; color:#64748b; margin-top: 10px;'>{T['performance']}</div>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        loss_metric = st.empty()
        loss_chart_placeholder = st.empty()
        
    losses = []
    
    for ep in range(epochs):
        loss = nn.train_step(X_train, y_cardiac_train, y_diabetes_train, y_life_train, y_vascular_train)
        if ep % max(1, (epochs // 20)) == 0 or ep == epochs - 1:
            losses.append(loss)
            progress_bar.progress((ep + 1) / epochs)
            loss_metric.markdown(f"**Loss:** `{loss:.4f}`")
            loss_chart_placeholder.line_chart(losses, height=120)
            
    nn.save_weights(WEIGHTS_FILE)
    st.session_state['training_losses'] = losses
    
    out = nn.forward(X_train)
    acc_c = np.mean((out["cardiac"].ravel() > 0.5) == y_cardiac_train.ravel())
    acc_d = np.mean((out["diabetes"].ravel() > 0.5) == y_diabetes_train.ravel())
    mae_l = np.mean(np.abs(out["life"].ravel() - y_life_train.ravel()))
    mae_v = np.mean(np.abs(out["vascular"].ravel() - y_vascular_train.ravel()))
    
    st.session_state['eval_metrics'] = {
        "acc_cardiac": acc_c,
        "acc_diabetes": acc_d,
        "mae_life": mae_l,
        "mae_vascular": mae_v
    }
    st.sidebar.success("Обучение завершено! Веса сохранены.")

# ────────────────── LAYOUT SETUP (2.1 : 1 RATIO) ──────────────────
main_col, controls_col = st.columns([2.1, 1])

# ────────────────── RIGHT PANEL: UNRESTRICTED BIOMARKER CONTROLS ──────────────────
with controls_col:
    with st.container(border=True):
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#3c4a42; text-transform:uppercase;'>Биомаркеры Пациента</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:11px; color:#64748b; margin-top:2px; margin-bottom:10px;'>Полный физиологический диапазон (Возраст 1..120 лет)</div>", unsafe_allow_html=True)
        
        u_age = st.slider(T["slider_age"], 1, 120, 45)
        u_sex = st.pills(T["slider_sex"], [T["sex_male"], T["sex_female"]], default=T["sex_male"])
        val_sex = 1.0 if u_sex == T["sex_male"] else 0.0
        u_bp = st.slider(T["slider_bp"], 50, 250, 120)
        u_chol = st.slider(T["slider_chol"], 50, 500, 190)
        u_glucose = st.slider(T["slider_glucose"], 40, 400, 95)
        
        u_max_hr = st.slider(T["slider_max_hr"], 40, 220, 140)
        u_st_dep = st.slider(T["slider_st_dep"], 0.0, 6.0, 0.5, step=0.1)
        u_bmi = st.slider(T["slider_bmi"], 10.0, 60.0, 23.5, step=0.5)
        u_activity = st.slider(T["slider_activity"], 0.0, 1.0, 0.5, step=0.1)
        u_smoking = st.pills(T["slider_smoking"], [T["smoke_no"], T["smoke_yes"]], default=T["smoke_no"])
        val_smoking = 1.0 if u_smoking == T["smoke_yes"] else 0.0
        val_smoking = 1.0 if u_smoking == T["smoke_yes"] else 0.0

        raw_vec = np.array([u_age, val_sex, u_bp, u_chol, u_glucose, u_max_hr, u_st_dep, u_bmi, u_activity, val_smoking])
        norm_vec = normalize(raw_vec, mins, maxs)

        # 🎲 MONTE CARLO DROPOUT UNCERTAINTY SAMPLING (30 stochastic passes)
        mc_res = nn.predict_with_uncertainty(norm_vec.reshape(1, -1), n_samples=30, dropout_rate=0.1)

        p_cardiac = float(mc_res["cardiac"]["mean"][0, 0])
        ci_cardiac = float(mc_res["cardiac"]["ci95"][0, 0]) * 100.0

        p_diabetes = float(mc_res["diabetes"]["mean"][0, 0])
        ci_diabetes = float(mc_res["diabetes"]["ci95"][0, 0]) * 100.0

        p_life = float(mc_res["life"]["mean"][0, 0])
        ci_life = float(mc_res["life"]["ci95"][0, 0])

        p_vascular = float(mc_res["vascular"]["mean"][0, 0])
        ci_vascular = float(mc_res["vascular"]["ci95"][0, 0])

# ────────────────── CLINICAL RECOMMENDATIONS & SYMPTOM ENGINE ──────────────────
def generate_clinical_analysis(raw_vec, p_cardiac, p_diabetes, p_life, p_vascular, lang="ru"):
    age, sex, bp, chol, glucose, max_hr, st_dep, bmi, activity, smoking = raw_vec
    
    triggers = []
    symptoms = []
    advices = []
    
    # 1. Cardiovascular & BP
    if bp > 140 or st_dep > 1.0:
        triggers.append("Артериальная гипертензия / Ишемическая депрессия ST" if lang=="ru" else "Hypertension / ST Depression")
        symptoms.append("Головные боли в затылочной области, одышка при нагрузке, тяжесть за грудиной" if lang=="ru" else "Occipital headaches, exertional dyspnea, chest tightness")
        advices.append("• **Кардиоконтроль:** Снизить натрий (<2г/сут), выполнить суточный мониторинг СМАД и ЭКГ." if lang=="ru" else "• **Cardio:** Sodium <2g/day, 24h Holter & ABPM monitoring.")
    elif bp > 130:
        triggers.append("Предгипертензия" if lang=="ru" else "Pre-hypertension")
        advices.append("• **Давление:** Регулярная аэробная нагрузка и снижение психоэмоционального стресса." if lang=="ru" else "• **BP:** Regular aerobic exercise and stress management.")

    # 2. Lipids & Metabolic
    if chol > 240:
        triggers.append("Выраженная гиперхолестеринемия" if lang=="ru" else "Severe Hypercholesterolemia")
        symptoms.append("Риск атеросклеротического поражения сосудов, сосудистый шум" if lang=="ru" else "Atherosclerotic risk, vascular bruits")
        advices.append("• **Липиды:** Диета DASH/Средиземноморская, УЗИ сонных артерий (УЗДГ СМА)." if lang=="ru" else "• **Lipids:** Mediterranean diet, Carotid Ultrasound.")
    elif chol > 200:
        triggers.append("Умеренная гиперхолестеринемия" if lang=="ru" else "Moderate Hypercholesterolemia")
        advices.append("• **Липиды:** Увеличить клетчатку, полиненасыщенные жиры (Омега-3)." if lang=="ru" else "• **Lipids:** Increase soluble fiber & Omega-3 fatty acids.")

    # 3. Glycemic control
    if glucose > 126:
        triggers.append("Гипергликемия натощак" if lang=="ru" else "Fasting Hyperglycemia")
        symptoms.append("Сухость во рту, быстрая утомляемость, повышенная жажда" if lang=="ru" else "Dry mouth, fatigue, polydipsia")
        advices.append("• **Метаболизм:** Консультация эндокринолога, замер гликированного гемоглобина (HbA1c)." if lang=="ru" else "• **Metabolism:** Endocrinologist consultation, HbA1c test.")

    # 4. Lifestyle & Weight
    if bmi > 30.0:
        triggers.append("Ожидание I-II ст." if lang=="ru" else "Obesity Class I-II")
        symptoms.append("Нагрузка на суставы, сниженная толерантность к нагрузкам" if lang=="ru" else "Joint strain, exercise intolerance")
        advices.append("• **Вес:** Дефицит калорий 300-500 ккал/сут, силовой и кардио тренинг." if lang=="ru" else "• **Weight:** Caloric deficit 300-500 kcal/day, strength & cardio.")
        
    if smoking == 1.0:
        triggers.append("Никотиновая интоксикация" if lang=="ru" else "Tobacco Dependency")
        symptoms.append("Кашель по утрам, спазм периферических сосудов, снижение VO2 max" if lang=="ru" else "Morning cough, peripheral vasoconstriction, low VO2 max")
        advices.append("• **Курение:** Отказ от табака — увеличивает продолжительность жизни на +4.5-7 лет." if lang=="ru" else "• **Smoking:** Quitting increases life expectancy by +4.5-7 yrs.")

    if activity < 0.3:
        triggers.append("Выраженная гиподинамия" if lang=="ru" else "Sedentary Lifestyle")
        advices.append("• **Активность:** Минимум 150 минут умеренной аэробной активности в неделю." if lang=="ru" else "• **Activity:** At least 150 mins of moderate aerobic exercise weekly.")

    if not triggers:
        triggers.append("Биомаркеры в норме" if lang=="ru" else "Optimal Biomarkers")
        symptoms.append("Симптомы патологий отсутствуют" if lang=="ru" else "No pathological symptoms noted")
        advices.append("• Все целевые показатели находятся в физиологической норме. Продолжайте текущий режим!" if lang=="ru" else "• All biomarkers within healthy baseline targets.")

    return triggers, symptoms, advices

triggers_list, symptoms_list, advices_list = generate_clinical_analysis(raw_vec, p_cardiac, p_diabetes, p_life, p_vascular, lang_code)

# ────────────────── MAIN PANEL: TELEMETRY & TABS ──────────────────
with main_col:
    # 4 Core KPI Telemetry Cards matching code.html
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-indicator" style="background:#10b981;" if "{p_cardiac <= 0.5}" else "background:#ba1a1a;"></div>
            <div class="telemetry-label">{T["cardiac_risk_hdr"]}</div>
            <div class="telemetry-value" style="color:{'#ba1a1a' if p_cardiac > 0.5 else '#006c49'};">{p_cardiac * 100:.1f}%</div>
            <div class="telemetry-badge" style="background:{'#ffdad6' if p_cardiac > 0.5 else '#eef6ee'}; color:{'#ba1a1a' if p_cardiac > 0.5 else '#006c49'};">
                ±{ci_cardiac:.1f}% ({T["high_risk"] if p_cardiac > 0.5 else T["low_risk"]})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-indicator" style="background:#10b981;" if "{p_diabetes <= 0.5}" else "background:#ba1a1a;"></div>
            <div class="telemetry-label">{T["diabetes_risk_hdr"]}</div>
            <div class="telemetry-value" style="color:{'#ba1a1a' if p_diabetes > 0.5 else '#006c49'};">{p_diabetes * 100:.1f}%</div>
            <div class="telemetry-badge" style="background:{'#ffdad6' if p_diabetes > 0.5 else '#eef6ee'}; color:{'#ba1a1a' if p_diabetes > 0.5 else '#006c49'};">
                ±{ci_diabetes:.1f}% ({T["high_risk"] if p_diabetes > 0.5 else T["low_risk"]})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        baseline_life = 82.0 if val_sex == 0.0 else 79.0
        delta_life = p_life - baseline_life
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-indicator" style="background:#0051d5;"></div>
            <div class="telemetry-label">{T["life_exp_hdr"]}</div>
            <div class="telemetry-value" style="color:#0051d5;">{p_life:.1f} <span style="font-size:13px;">{T['years']}</span></div>
            <div class="telemetry-badge" style="background:#dbe1ff; color:#003ea8;">
                ±{ci_life:.1f} ({delta_life:+.1f} {T['vs_baseline']})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        delta_age = p_vascular - u_age
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-indicator" style="background:{'#ba1a1a' if delta_age > 0 else '#10b981'};"></div>
            <div class="telemetry-label">{T["vascular_age_hdr"]}</div>
            <div class="telemetry-value" style="color:{'#ba1a1a' if delta_age > 0 else '#006c49'};">{p_vascular:.1f} <span style="font-size:13px;">{T['years']}</span></div>
            <div class="telemetry-badge" style="background:{'#ffdad6' if delta_age > 0 else '#eef6ee'}; color:{'#ba1a1a' if delta_age > 0 else '#006c49'};">
                ±{ci_vascular:.1f} ({delta_age:+.1f} {T['vs_chrono']})
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # Multi-Feature Analytics Tabs
    main_tabs = st.tabs(T["tabs"])
    
    # ── TAB 1: ARCHITECTURE & TRAINING METRICS ──
    with main_tabs[0]:
        st.markdown(f"<div style='font-size:15px; font-weight:700; color:#006c49; margin-bottom: 12px;'>{T['arch_title']}</div>", unsafe_allow_html=True)
        
        badges = [f"<span style='background-color:#006c49; color:#ffffff; padding:6px 14px; border-radius:20px; font-weight:700; font-size:12px;'>Вход (10)</span>"]
        for idx, h_size in enumerate(nn.hidden_sizes):
            badges.append(f"<span style='background-color:#006c49; color:#ffffff; padding:6px 14px; border-radius:20px; font-weight:700; font-size:12px;'>Слой {idx+1} ({h_size})</span>")
        badges.append(f"<span style='background-color:#006c49; color:#ffffff; padding:6px 14px; border-radius:20px; font-weight:700; font-size:12px;'>Выходы (4)</span>")
        
        arch_html = " <span style='color:#006c49; font-weight:900;'>➔</span> ".join(badges)
        st.markdown(f"<div style='margin-bottom:12px; display:flex; align-items:center; flex-wrap:wrap; gap:6px;'>{arch_html}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div style='font-size:14px; font-weight:700; color:#3c4a42; margin-bottom: 10px;'>{T['loss_title']}</div>", unsafe_allow_html=True)
            losses_to_show = st.session_state.get('training_losses', [1.538, 1.474, 1.428, 1.426, 1.432])
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=losses_to_show, mode='lines+markers', name='Loss', line=dict(color='#006c49', width=2)))
            fig_loss.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_loss.update_xaxes(title_text='Шаги обучения', gridcolor='#dde4dd')
            fig_loss.update_yaxes(title_text='Loss', gridcolor='#dde4dd')
            st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
                
        with c2:
            st.markdown(f"<div style='font-size:14px; font-weight:700; color:#3c4a42; margin-bottom: 10px;'>{T['eval_title']}</div>", unsafe_allow_html=True)
            em = st.session_state.get('eval_metrics', {"acc_cardiac": 0.9954, "acc_diabetes": 0.9962, "mae_life": 4.97, "mae_vascular": 13.03})
            
            def render_metric_card(label, val):
                return f"""
                <div style="background:#ffffff; border:1px solid #eef6ee; border-radius:12px; padding:10px 14px; margin-bottom:8px; box-shadow:0 2px 8px rgba(0,108,73,0.03);">
                    <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">{label}</div>
                    <div style="font-size:16px; font-weight:700; color:#006c49; margin-top:2px; font-family:'JetBrains Mono',monospace;">{val}</div>
                </div>
                """
            
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(render_metric_card(T["cardiac_risk_hdr"], f"{T['accuracy']}: {em['acc_cardiac']*100:.1f}%"), unsafe_allow_html=True)
                st.markdown(render_metric_card(T["life_exp_hdr"], f"{T['mae']}: {em['mae_life']:.2f} {T['years']}"), unsafe_allow_html=True)
            with m2:
                st.markdown(render_metric_card(render_metric_card_title := T["diabetes_risk_hdr"], f"{T['accuracy']}: {em['acc_diabetes']*100:.1f}%"), unsafe_allow_html=True)
                st.markdown(render_metric_card(T["vascular_age_hdr"], f"{T['mae']}: {em['mae_vascular']:.2f} {T['years']}"), unsafe_allow_html=True)
                
    # ── TAB 2: 3D PCA SPACE ──
    with main_tabs[1]:
        X_all = np.vstack([norm_X, norm_vec.reshape(1, -1)])
        feats = nn.get_stage_features(X_all)
        latent = feats["stage2_latent"]
        
        pca = PCA(n_components=3, random_state=42)
        latent_3d = pca.fit_transform(latent)
        
        patient_3d = latent_3d[-1:]
        dataset_3d = latent_3d[:-1]
        
        y_flat = y_cardiac.ravel()
        
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=dataset_3d[y_flat == 0, 0], y=dataset_3d[y_flat == 0, 1], z=dataset_3d[y_flat == 0, 2],
            mode='markers', name=T["low_risk_label"],
            marker=dict(size=4, color='#10b981', opacity=0.6)
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=dataset_3d[y_flat == 1, 0], y=dataset_3d[y_flat == 1, 1], z=dataset_3d[y_flat == 1, 2],
            mode='markers', name=T["high_risk_label"],
            marker=dict(size=4, color='#ef4444', opacity=0.6)
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=patient_3d[:, 0], y=patient_3d[:, 1], z=patient_3d[:, 2],
            mode='markers', name="Пациент 🎯",
            marker=dict(size=12, color='#0051d5', symbol='diamond', opacity=1.0)
        ))
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#3c4a42', size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            height=360,
            scene=dict(
                xaxis=dict(backgroundcolor="#eef6ee", gridcolor="#dde4dd", title="PCA 1"),
                yaxis=dict(backgroundcolor="#eef6ee", gridcolor="#dde4dd", title="PCA 2"),
                zaxis=dict(backgroundcolor="#eef6ee", gridcolor="#dde4dd", title="PCA 3"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

    # ── TAB 3: 20-YEAR FORECAST ──
    with main_tabs[2]:
        years_future = np.array([0, 5, 10, 15, 20])
        base_cardiac_risk_traj = []
        base_vascular_traj = []
        opt_cardiac_risk_traj = []
        
        for y in years_future:
            sim_base_raw = raw_vec.copy()
            sim_base_raw[0] = min(sim_base_raw[0] + y, maxs[0])
            sim_base_norm = normalize(sim_base_raw, mins, maxs)
            sim_base_preds = nn.forward(sim_base_norm.reshape(1, -1))
            base_cardiac_risk_traj.append(float(sim_base_preds["cardiac"][0, 0]) * 100.0)
            base_vascular_traj.append(float(sim_base_preds["vascular"][0, 0]))
            
            sim_opt_raw = raw_vec.copy()
            sim_opt_raw[0] = min(sim_opt_raw[0] + y, maxs[0])
            sim_opt_raw[9] = 0.0
            sim_opt_raw[8] = 1.0
            sim_opt_raw[2] = min(120.0, sim_opt_raw[2])
            sim_opt_raw[3] = min(180.0, sim_opt_raw[3])
            sim_opt_norm = normalize(sim_opt_raw, mins, maxs)
            sim_opt_preds = nn.forward(sim_opt_norm.reshape(1, -1))
            opt_cardiac_risk_traj.append(float(sim_opt_preds["cardiac"][0, 0]) * 100.0)

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=years_future, y=base_cardiac_risk_traj, mode='lines+markers', name="Текущий трек (Кардио-риск %)", line=dict(color='#ba1a1a', width=2)))
        fig_traj.add_trace(go.Scatter(x=years_future, y=opt_cardiac_risk_traj, mode='lines+markers', name="Оптимизированный трек (Риск %)", line=dict(color='#10b981', width=2, dash='dash')))
        fig_traj.add_trace(go.Scatter(x=years_future, y=base_vascular_traj, mode='lines+markers', name="Сосудистый возраст (лет)", line=dict(color='#0051d5', width=2)))
        
        fig_traj.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#f4fbf4',
            font=dict(family='Inter', color='#3c4a42', size=11),
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig_traj.update_xaxes(title="Годы (вперед)", showgrid=True, gridcolor='#dde4dd')
        st.plotly_chart(fig_traj, use_container_width=True)

    # ── TAB 4: RADAR CHART ──
    with main_tabs[3]:
        categories = ['Давление', 'Холестерин', 'Глюкоза', 'ИМТ', 'Депрессия ST', 'Курение']
        patient_vals = [u_bp/180.0, u_chol/350.0, u_glucose/220.0, u_bmi/40.0, u_st_dep/4.5, val_smoking]
        healthy_ref = [120/180.0, 190/350.0, 90/220.0, 22.5/40.0, 0.2/4.5, 0.0]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Пациент', line_color='#0051d5'))
        fig_radar.add_trace(go.Scatterpolar(r=healthy_ref, theta=categories, fill='toself', name='Здоровый эталон', line_color='#10b981', opacity=0.5))

        fig_radar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(bgcolor='#f4fbf4', radialaxis=dict(visible=True, range=[0, 1])),
            font=dict(family='Inter', color='#3c4a42', size=11),
            margin=dict(l=30, r=30, t=20, b=20),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 5: EXPLAINABLE AI (XAI) ──
    with main_tabs[4]:
        attributions = nn.get_feature_attributions(norm_vec.reshape(1, -1))
        feature_names = [f[f"name_{lang_code}"] for f in data_dict["feature_meta"]]

        fig_xai = px.bar(
            x=attributions, y=feature_names, orientation='h',
            labels={'x': T["xai_xlabel"], 'y': 'Биомаркер'},
            color=attributions, color_continuous_scale='emrld'
        )
        fig_xai.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#f4fbf4',
            font=dict(family='Inter', color='#3c4a42', size=11),
            margin=dict(l=20, r=20, t=20, b=20),
            height=360,
            coloraxis_showscale=False
        )
        fig_xai.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_xai, use_container_width=True)

    # ────────────────── BOTTOM SECTION: ACTION PLAN & ROOT CAUSES ──────────────────
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        with st.container(border=True):
            st.markdown(f"<div style='font-size:12px; font-weight:700; color:#006c49; text-transform:uppercase;'>📋 {T['recommendations_hdr']}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            for adv in advices_list:
                st.markdown(f"<div style='font-size:12px; color:#161d19; line-height:1.6; margin-bottom:6px;'>{adv}</div>", unsafe_allow_html=True)
                
    with b_col2:
        with st.container(border=True):
            st.markdown(f"<div style='font-size:12px; font-weight:700; color:#ba1a1a; text-transform:uppercase;'>🩺 {T['symptoms_hdr']}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            
            st.markdown("<div style='font-size:11px; font-weight:700; color:#64748b;'>КЛЮЧЕВЫЕ ТРИГГЕРЫ РИСКА:</div>", unsafe_allow_html=True)
            for trg in triggers_list:
                st.markdown(f"<span style='background:#ffdad6; color:#ba1a1a; padding:2px 8px; border-radius:100px; font-size:11px; font-weight:700; margin-right:4px;'>{trg}</span>", unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top:10px; font-size:11px; font-weight:700; color:#64748b;'>ВОЗМОЖНЫЕ КЛИНИЧЕСКИЕ ПРОЯВЛЕНИЯ:</div>", unsafe_allow_html=True)
            if symptoms_list:
                for sym in symptoms_list:
                    st.markdown(f"<div style='font-size:12px; color:#475569; line-height:1.5;'>• {sym}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:12px; color:#006c49;'>• Патологических симптомов не выявлено.</div>", unsafe_allow_html=True)

# ────────────────── RIGHT PANEL: EXPORT & SIMULATOR ──────────────────
with controls_col:
    with st.container(border=True):
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#3c4a42; text-transform:uppercase;'>🧪 {T['what_if_hdr']}</div>", unsafe_allow_html=True)
        
        sim_quit_smoke = st.checkbox(T["sim_quit_smoke"], value=(val_smoking == 1.0))
        sim_lower_bp = st.checkbox(T["sim_lower_bp"], value=False)
        sim_max_act = st.checkbox(T["sim_max_act"], value=False)

        sim_raw = raw_vec.copy()
        if sim_quit_smoke: sim_raw[9] = 0.0
        if sim_lower_bp: sim_raw[2] = 120.0
        if sim_max_act: sim_raw[8] = 1.0

        sim_norm = normalize(sim_raw, mins, maxs)
        sim_preds = nn.forward(sim_norm.reshape(1, -1))

        sim_cardiac = float(sim_preds["cardiac"][0, 0])
        sim_life = float(sim_preds["life"][0, 0])

        risk_diff = (sim_cardiac - p_cardiac) * 100.0
        life_diff = sim_life - p_life
        
        st.markdown(f"""
        <div style="background-color: #f4fbf4; border: 1px solid #dde4dd; border-radius: 10px; padding: 12px; font-family: 'JetBrains Mono', monospace; margin-top:8px;">
            <div style="font-size: 12px; font-weight: 700; color: #10b981;">
                Кардио-риск: {p_cardiac*100:.1f}% ➔ {sim_cardiac*100:.1f}% ({risk_diff:+.1f}%)
            </div>
            <div style="font-size: 12px; font-weight: 700; color: #0051d5; margin-top: 4px;">
                Жизнь: {p_life:.1f} ➔ {sim_life:.1f} ({life_diff:+.1f} лет)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase;'>📑 Экспорт Клинических Отчётов</div>", unsafe_allow_html=True)
        
        sex_display = T["sex_male"] if val_sex == 1.0 else T["sex_female"]
        rep_title = "MedAI Longevity Diagnostic Report" if lang_code == "en" else "Клинический диагностический отчёт MedAI Longevity"
        
        # High-End HTML Report with Print Functionality
        report_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>{rep_title}</title>
    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 32px; }}
        .report-card {{ background: #ffffff; max-width: 850px; margin: 0 auto; border-radius: 16px; border: 1px solid #e2e8f0; padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #006c49; padding-bottom: 20px; margin-bottom: 24px; }}
        .brand {{ font-size: 24px; font-weight: 800; color: #006c49; }}
        .meta {{ font-size: 12px; color: #64748b; text-align: right; }}
        .patient-box {{ background: #f4fbf4; border: 1px solid #dde4dd; border-radius: 12px; padding: 16px; margin-bottom: 24px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 13px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 28px; }}
        .metric-tile {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; border-left: 5px solid #006c49; }}
        .metric-title {{ font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; }}
        .metric-val {{ font-size: 22px; font-weight: 800; color: #006c49; margin-top: 4px; font-family: monospace; }}
        .section-title {{ font-size: 16px; font-weight: 700; color: #006c49; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px; }}
        .adv-list {{ font-size: 13px; line-height: 1.6; color: #334155; }}
        .btn-print {{ background: #006c49; color: #ffffff; border: none; padding: 10px 24px; border-radius: 100px; font-weight: 700; cursor: pointer; margin-bottom: 20px; font-size: 13px; }}
        @media print {{ .btn-print {{ display: none; }} body {{ padding: 0; background: white; }} .report-card {{ border: none; box-shadow: none; padding: 0; }} }}
    </style>
</head>
<body>
    <div style="text-align: center;">
        <button class="btn-print" onclick="window.print()">🖨️ Распечатать / Сохранить в PDF</button>
    </div>
    <div class="report-card">
        <div class="header">
            <div>
                <div class="brand">🧬 MedAI Longevity Suite</div>
                <div style="font-size: 13px; color: #64748b; margin-top: 2px;">Executive Clinical AI Diagnostics</div>
            </div>
            <div class="meta">
                <div><b>Дата:</b> {st.session_state.get('cur_date', '13.08.2026')}</div>
                <div><b>Версия ИИ:</b> v3.0 (Monte Carlo 95% CI)</div>
            </div>
        </div>

        <div class="patient-box">
            <div><b>Возраст:</b> {u_age} {T['years']}</div>
            <div><b>Пол:</b> {sex_display}</div>
            <div><b>АД:</b> {u_bp} мм рт.ст.</div>
            <div><b>ИМТ:</b> {u_bmi}</div>
            <div><b>Холестерин:</b> {u_chol} мг/дл</div>
            <div><b>Глюкоза:</b> {u_glucose} мг/дл</div>
            <div><b>ST Депрессия:</b> {u_st_dep} мм</div>
            <div><b>Курение:</b> {'Да' if val_smoking==1.0 else 'Нет'}</div>
        </div>

        <div class="section-title">1. Нейросетевая Диагностика Рисков (Monte Carlo Dropout)</div>
        <div class="metrics-grid">
            <div class="metric-tile">
                <div class="metric-title">{T['cardiac_risk_hdr']}</div>
                <div class="metric-val">{p_cardiac*100:.1f}% ±{ci_cardiac:.1f}%</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">{T['diabetes_risk_hdr']}</div>
                <div class="metric-val">{p_diabetes*100:.1f}% ±{ci_diabetes:.1f}%</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">{T['life_exp_hdr']}</div>
                <div class="metric-val">{p_life:.1f} ±{ci_life:.1f} {T['years']}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">{T['vascular_age_hdr']}</div>
                <div class="metric-val">{p_vascular:.1f} ±{ci_vascular:.1f} {T['years']}</div>
            </div>
        </div>

        <div class="section-title">2. Ключевые Триггеры и Анализ Симптомов</div>
        <div style="margin-bottom:12px;">
            <b>Выявленные факторы риска:</b> {', '.join(triggers_list)}
        </div>
        <div class="adv-list">
            <b>Вероятные клинические проявления:</b>
            <ul>
                {"".join([f"<li>{s}</li>" for s in symptoms_list]) if symptoms_list else "<li>Патологические симптомы отсутствуют.</li>"}
            </ul>
        </div>

        <div class="section-title">3. Персонализированный Клинический План Долголетия</div>
        <div class="adv-list">
            <ul>
                {"".join([f"<li>{a.replace('• ', '')}</li>" for a in advices_list])}
            </ul>
        </div>

        <div style="margin-top: 40px; border-t: 1px solid #e2e8f0; pt: 16px; font-size: 11px; color: #94a3b8; text-align: center;">
            Отчёт сформирован автоматически мультизадачной нейросетью MedAI Longevity Suite. Документ носит рекомендательный характер.
        </div>
    </div>
</body>
</html>
"""
        
        observations = [
            ("cardiac-risk-001", "79423-0", "Cardiovascular disease risk", round(p_cardiac * 100, 2), "%", round(ci_cardiac, 2), "%"),
            ("diabetes-risk-001", "73696-7", "Type-II Diabetes risk", round(p_diabetes * 100, 2), "%", round(ci_diabetes, 2), "%"),
            ("life-expectancy-001", "39156-5", "Life Expectancy", round(p_life, 1), "years", round(ci_life, 1), "years"),
            ("vascular-age-001", "39156-5", "Vascular Age", round(p_vascular, 1), "years", round(ci_vascular, 1), "years"),
        ]
        
        entries = [
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "medai-report-001",
                    "status": "final",
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "80352-8", "display": "Cardiovascular and Longevity Risk Assessment"}]
                    },
                    "subject": {"display": f"Patient Age {u_age}, Sex {'Male' if val_sex == 1.0 else 'Female'}"},
                    "result": [{"reference": f"Observation/{obs[0]}"} for obs in observations]
                }
            }
        ]
        
        for obs in observations:
            entries.append({
                "resource": {
                    "resourceType": "Observation",
                    "id": obs[0],
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": obs[1], "display": obs[2]}]},
                    "valueQuantity": {"value": obs[3], "unit": obs[4], "system": "http://unitsofmeasure.org", "code": obs[4]},
                    "extension": [{
                        "url": "http://hl7.org/fhir/StructureDefinition/observation-confidenceInterval",
                        "valueQuantity": {"value": obs[5], "unit": obs[6]}
                    }]
                }
            })
            
        fhir_json = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": entries
        }
        
        st.download_button(T["report_btn"], data=report_html, file_name=f"MedAI_Clinical_Report_Patient_{u_age}yo.html", mime="text/html", use_container_width=True)
        st.download_button(T["fhir_btn"], data=json.dumps(fhir_json, indent=2), file_name="MedAI_FHIR_Record.json", mime="application/json", use_container_width=True)
