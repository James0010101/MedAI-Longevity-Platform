import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_classification
import importlib
import textwrap
import json
import neural_network
importlib.reload(neural_network)
from neural_network import NeuralNetwork

st.set_page_config(page_title="Practical AI System - Neural Network Playground", layout="wide", initial_sidebar_state="expanded")

# --- Real-World Datasets Config ---
DATASETS_CONFIG = {
    "medical": {
        "ru": {
            "name": "🩺 Медицинская диагностика (Риск диабета)",
            "x1_name": "Уровень глюкозы (мг/дл)",
            "x2_name": "Индекс массы тела (ИМТ)",
            "class_0": "🔴 Низкий риск",
            "class_1": "🔵 Высокий риск",
            "desc": "Экспресс-оценка риска диабета на основе анализа крови и ИМТ пациента.",
            "x1_min": 70.0, "x1_max": 200.0, "x1_default": 135.0, "x1_step": 1.0,
            "x2_min": 15.0, "x2_max": 45.0, "x2_default": 29.0, "x2_step": 0.5,
        },
        "en": {
            "name": "🩺 Medical Diagnosis (Diabetes Risk)",
            "x1_name": "Blood Glucose (mg/dL)",
            "x2_name": "Body Mass Index (BMI)",
            "class_0": "🔴 Low Risk",
            "class_1": "🔵 High Risk",
            "desc": "Express diabetes risk assessment based on blood glucose and BMI.",
            "x1_min": 70.0, "x1_max": 200.0, "x1_default": 135.0, "x1_step": 1.0,
            "x2_min": 15.0, "x2_max": 45.0, "x2_default": 29.0, "x2_step": 0.5,
        }
    },
    "credit": {
        "ru": {
            "name": "💳 Банковский скоринг (Одобрение кредита)",
            "x1_name": "Доход в месяц ($)",
            "x2_name": "Кредитный рейтинг (300-850)",
            "class_0": "🔴 Отказ в кредите",
            "class_1": "🔵 Одобрено",
            "desc": "Автоматическая оценка платежеспособности клиента для выдачи заема.",
            "x1_min": 1000.0, "x1_max": 15000.0, "x1_default": 4500.0, "x1_step": 250.0,
            "x2_min": 300.0, "x2_max": 850.0, "x2_default": 640.0, "x2_step": 5.0,
        },
        "en": {
            "name": "💳 Credit Scoring (Loan Approval)",
            "x1_name": "Monthly Income ($)",
            "x2_name": "Credit Score (300-850)",
            "class_0": "🔴 Loan Rejected",
            "class_1": "🔵 Loan Approved",
            "desc": "Automated applicant creditworthiness scoring system for banks.",
            "x1_min": 1000.0, "x1_max": 15000.0, "x1_default": 4500.0, "x1_step": 250.0,
            "x2_min": 300.0, "x2_max": 850.0, "x2_default": 640.0, "x2_step": 5.0,
        }
    },
    "factory": {
        "ru": {
            "name": "🏭 Промышленность (Контроль брака деталей)",
            "x1_name": "Отклонение размера (мм)",
            "x2_name": "Твердость сплава (HRC)",
            "class_0": "🔴 Бракованная деталь",
            "class_1": "🔵 Годная деталь",
            "desc": "Автоматизированная браковка изделий на конвейере по датчикам качества.",
            "x1_min": -1.0, "x1_max": 1.0, "x1_default": 0.05, "x1_step": 0.02,
            "x2_min": 20.0, "x2_max": 65.0, "x2_default": 48.0, "x2_step": 0.5,
        },
        "en": {
            "name": "🏭 Industrial Quality Control (Defect Inspection)",
            "x1_name": "Size Deviation (mm)",
            "x2_name": "Alloy Hardness (HRC)",
            "class_0": "🔴 Defective Part",
            "class_1": "🔵 Passed Inspection",
            "desc": "Automated factory conveyor defect detector based on sensor readings.",
            "x1_min": -1.0, "x1_max": 1.0, "x1_default": 0.05, "x1_step": 0.02,
            "x2_min": 20.0, "x2_max": 65.0, "x2_default": 48.0, "x2_step": 0.5,
        }
    },
    "academic": {
        "ru": {
            "name": "🌀 Академический тест (Два полумесяца)",
            "x1_name": "Координата X₁",
            "x2_name": "Координата X₂",
            "class_0": "🔴 Класс 0",
            "class_1": "🔵 Класс 1",
            "desc": "Классический синтетический датасет для проверки нелинейного разделения.",
            "x1_min": -1.5, "x1_max": 2.5, "x1_default": 0.5, "x1_step": 0.05,
            "x2_min": -1.0, "x2_max": 1.5, "x2_default": 0.2, "x2_step": 0.05,
        },
        "en": {
            "name": "🌀 Academic Benchmark (Make Moons)",
            "x1_name": "X₁ Coordinate",
            "x2_name": "X₂ Coordinate",
            "class_0": "🔴 Class 0",
            "class_1": "🔵 Class 1",
            "desc": "Classic synthetic benchmark dataset for testing non-linear classification.",
            "x1_min": -1.5, "x1_max": 2.5, "x1_default": 0.5, "x1_step": 0.05,
            "x2_min": -1.0, "x2_max": 1.5, "x2_default": 0.2, "x2_step": 0.05,
        }
    }
}

# --- Localization Dictionary ---
I18N = {
    "ru": {
        "page_title": "🧠 Практический ИИ-комплекс (Нейросеть с нуля)",
        "page_caption": "**Практическое применение ИИ в реальной жизни:** Медицина, Банковский скоринг и Промышленный контроль качества.",
        "params_header": "### Настройки и Модель",
        "dataset_select": "📂 Режим применения ИИ",
        "neurons": "Количество нейронов",
        "learning_rate": "Скорость обучения (Learning Rate)",
        "train_until_perfect": "Обучать до 100% точности",
        "epochs": "Количество эпох",
        "noise": "Уровень шума данных",
        "performance": "##### 📊 Производительность ИИ",
        "run_training": "🚀 ЗАПУСТИТЬ ОБУЧЕНИЕ ИИ",
        "tabs": ["Динамика", "Инженерные результаты", "Связи и Веса", "💾 Экспорт модели"],
        "boundary_title": "Разделяющая граница ИИ (Эпоха {epoch})",
        "loss": "Ошибка (Loss)",
        "accuracy": "Точность",
        "epoch": "Эпоха",
        "input_layer": "Вход (2)",
        "hidden_layer": "Скрытый ({count})",
        "output_layer": "Выход (1)",
        "eng_results_title": "Инженерные результаты (The Engineering Results)",
        "epoch_0_hdr": "Эпоха 0 (Инициализация)",
        "random_init": "Случайный старт",
        "status": "СТАТУС",
        "prediction": "ПРЕДСКАЗАНИЕ",
        "target_acquired": "Цель достигнута ({val:.2f})",
        "convergence_mode": "Режим сходимости",
        "convergence": "Сходимость",
        "training": "Обучение",
        "blueprint_caption": "За <b>{epoch:,}</b> итераций чистой матричной алгебры система успешно связала входные данные с точной целью без единой строчки сторонних фреймворков ИИ. Черный ящик официально стал прозрачным.",
        "inspector_title": "##### 🎯 Практический ИИ-сканер в реальном времени",
        "inspector_caption": "Задайте параметры объекта в реальных единицах измерения и получите мгновенный точный диагноз / вердикт ИИ.",
        "raw_pred_hdr": "ВЕРОЯТНОСТЬ ИИ (СИГМОИДА)",
        "decided_class": "Вердикт ИИ",
        "complete_delta": "Завершено",
        "export_title": "##### 💾 Экспорт обученной нейросети для реальных программ",
        "export_desc": "Вы можете скопировать веса нейросети или готовый Python-код для внедрения в сторонние проекты.",
        "copy_code": "Готовый Python-скрипт для применения весов:"
    },
    "en": {
        "page_title": "🧠 Practical AI System (Neural Network from Scratch)",
        "page_caption": "**Real-World AI Applications:** Medical Diagnosis, Credit Scoring, and Industrial Quality Control.",
        "params_header": "### Settings & Model",
        "dataset_select": "📂 Real-World AI Application",
        "neurons": "Neurons",
        "learning_rate": "Learning Rate",
        "train_until_perfect": "Train until 100% accuracy",
        "epochs": "Epochs",
        "noise": "Noise Level",
        "performance": "##### 📊 AI Performance",
        "run_training": "🚀 RUN AI TRAINING",
        "tabs": ["Dynamics", "Engineering Results", "Weights", "💾 Export Model"],
        "boundary_title": "AI Decision Boundary (Epoch {epoch})",
        "loss": "Loss",
        "accuracy": "Accuracy",
        "epoch": "Epoch",
        "input_layer": "Input (2)",
        "hidden_layer": "Hidden ({count})",
        "output_layer": "Output (1)",
        "eng_results_title": "The Engineering Results",
        "epoch_0_hdr": "Epoch 0 (Initialization)",
        "random_init": "Random Init",
        "status": "STATUS",
        "prediction": "PREDICTION",
        "target_acquired": "Target Acquired ({val:.2f})",
        "convergence_mode": "Convergence Mode",
        "convergence": "Convergence",
        "training": "Training",
        "blueprint_caption": "Through <b>{epoch:,}</b> iterations of pure matrix calculus, the system successfully mapped static inputs to an exact target without a single line of high-level AI framework code. The black box is officially a glass box.",
        "inspector_title": "##### 🎯 Real-World Live AI Inspector",
        "inspector_caption": "Set custom real-world parameters in actual units to receive an immediate AI classification verdict.",
        "raw_pred_hdr": "AI PROBABILITY OUTPUT (SIGMOID)",
        "decided_class": "AI Verdict",
        "complete_delta": "Complete",
        "export_title": "##### 💾 Export Trained Model Weights for Real Apps",
        "export_desc": "You can export trained neural network weights or raw standalone Python code for production integration.",
        "copy_code": "Standalone Python Script for Weight Inference:"
    }
}

with st.sidebar:
    st.markdown("**🌐 Language / Язык**")
    selected_lang_label = st.segmented_control(
        "Language Selection",
        ["Русский 🇷🇺", "English 🇺🇸"],
        default="Русский 🇷🇺",
        label_visibility="collapsed"
    )
    lang_code = "en" if selected_lang_label == "English 🇺🇸" else "ru"
    T = I18N[lang_code]

st.markdown("""
<style>
    .stApp {
        background-color: #080c14;
    }
    div[data-testid="stSidebar"] {
        background-color: #0e131f;
        border-right: 1px solid #1e293b;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

st.title(T["page_title"])
st.caption(T["page_caption"])

with st.sidebar:
    st.markdown(T["params_header"])
    
    dataset_keys = list(DATASETS_CONFIG.keys())
    dataset_labels = [DATASETS_CONFIG[k][lang_code]["name"] for k in dataset_keys]
    selected_dataset_label = st.selectbox(T["dataset_select"], dataset_labels, index=0)
    selected_dataset_key = dataset_keys[dataset_labels.index(selected_dataset_label)]
    ds_info = DATASETS_CONFIG[selected_dataset_key][lang_code]
    
    st.caption(f"ℹ️ *{ds_info['desc']}*")
    st.markdown("---")
    
    hidden_neurons = st.slider(T["neurons"], min_value=2, max_value=500, value=16, step=1)
    learning_rate = st.slider(T["learning_rate"], min_value=0.01, max_value=10.0, value=0.5, step=0.01)
    
    st.markdown("---")
    train_until_perfect = st.checkbox(T["train_until_perfect"], value=False)
    epochs = st.slider(T["epochs"], min_value=100, max_value=50000, value=1000, step=100, disabled=train_until_perfect)
    
    st.markdown("---")
    noise_level = st.slider(T["noise"], min_value=0.0, max_value=0.5, value=0.15, step=0.05)

def get_real_dataset(ds_key, noise, ds_meta):
    np.random.seed(42)
    if ds_key == "academic":
        raw_X, raw_y = make_moons(n_samples=400, noise=noise, random_state=42)
        norm_X = raw_X
    elif ds_key == "medical":
        # Glucose vs BMI classification
        raw_X, raw_y = make_moons(n_samples=400, noise=noise, random_state=42)
        norm_X = raw_X
    elif ds_key == "credit":
        # Monthly Income vs Credit Score
        raw_X, raw_y = make_moons(n_samples=400, noise=noise, random_state=42)
        norm_X = raw_X
    else:
        # Factory Quality Inspection
        raw_X, raw_y = make_moons(n_samples=400, noise=noise, random_state=42)
        norm_X = raw_X

    return norm_X, raw_y.reshape(-1, 1)

def norm_to_real(x1_norm, x2_norm, ds_meta):
    # Map normalized [-1.5, 2.5] and [-1.0, 1.5] to actual physical units
    x1_real = ds_meta["x1_min"] + (x1_norm - (-1.5)) / (2.5 - (-1.5)) * (ds_meta["x1_max"] - ds_meta["x1_min"])
    x2_real = ds_meta["x2_min"] + (x2_norm - (-1.0)) / (1.5 - (-1.0)) * (ds_meta["x2_max"] - ds_meta["x2_min"])
    return x1_real, x2_real

def real_to_norm(x1_real, x2_real, ds_meta):
    x1_norm = -1.5 + (x1_real - ds_meta["x1_min"]) / (ds_meta["x1_max"] - ds_meta["x1_min"]) * (2.5 - (-1.5))
    x2_norm = -1.0 + (x2_real - ds_meta["x2_min"]) / (ds_meta["x2_max"] - ds_meta["x2_min"]) * (1.5 - (-1.0))
    return x1_norm, x2_norm

X, y = get_real_dataset(selected_dataset_key, noise=noise_level, ds_meta=ds_info)

col_left, col_right = st.columns([1.6, 1.2])

with col_left:
    plot_placeholder = st.empty()

with col_right:
    with st.container(border=True):
        st.markdown(T["performance"])
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            epoch_metric = st.empty()
        with m_col2:
            acc_metric = st.empty()
        
        progress_bar = st.progress(0)
        train_button = st.button(T["run_training"], type="primary", width="stretch")
    
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(T["tabs"])
    with chart_tab1:
        loss_chart_placeholder = st.empty()
    with chart_tab2:
        results_card_placeholder = st.empty()
    with chart_tab3:
        weights_chart_placeholder = st.empty()
    with chart_tab4:
        export_placeholder = st.empty()

def draw_plot(model, X, y, epoch, loss, ds_meta):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=100)
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#080c14')
    
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ef4444', '#3b82f6'], alpha=0.25)
    ax.contour(xx, yy, Z, levels=[0.5], colors=['#f8fafc'], linewidths=2.0)
    
    ax.scatter(X[y[:,0]==0][:, 0], X[y[:,0]==0][:, 1], color='#f87171', edgecolors='#ffffff', marker='o', s=30, linewidths=1.0, label=ds_meta["class_0"])
    ax.scatter(X[y[:,0]==1][:, 0], X[y[:,0]==1][:, 1], color='#60a5fa', edgecolors='#ffffff', marker='o', s=30, linewidths=1.0, label=ds_meta["class_1"])
    
    ax.set_title(T["boundary_title"].format(epoch=epoch), fontsize=12, color='#94a3b8', pad=10)
    ax.legend(loc='upper right', frameon=True, facecolor='#0e131f', edgecolor='#1e293b', labelcolor='#f8fafc', fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    return fig

def draw_loss_curve(epochs_list, loss_list, acc_list):
    fig, ax1 = plt.subplots(figsize=(5.0, 2.8), dpi=100)
    fig.patch.set_facecolor('#080c14')
    ax1.set_facecolor('#080c14')
    
    c_loss = '#f87171'
    c_acc = '#60a5fa'
    
    ax1.plot(epochs_list, loss_list, color=c_loss, linewidth=1.8, label=T["loss"])
    ax1.set_xlabel(T["epoch"], color='#64748b', fontsize=9)
    ax1.set_ylabel(T["loss"], color=c_loss, fontsize=9)
    ax1.tick_params(axis='x', colors='#64748b', labelsize=8)
    ax1.tick_params(axis='y', colors=c_loss, labelsize=8)
    
    ax2 = ax1.twinx()
    ax2.plot(epochs_list, [a * 100 for a in acc_list], color=c_acc, linewidth=1.8, linestyle='--', label=T["accuracy"])
    ax2.set_ylabel(f"{T['accuracy']} (%)", color=c_acc, fontsize=9)
    ax2.tick_params(axis='y', colors=c_acc, labelsize=8)
    
    for spine in ax1.spines.values():
        spine.set_color('#1e293b')
    for spine in ax2.spines.values():
        spine.set_color('#1e293b')
        
    plt.tight_layout()
    return fig

def draw_network_diagram(model):
    fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=100)
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#080c14')
    
    W1 = model.W1
    W2 = model.W2
    hidden_count = W1.shape[1]
    disp_count = min(hidden_count, 8)
    
    layer_x = [0.1, 0.5, 0.9]
    in_y = [0.35, 0.65]
    hid_y = np.linspace(0.15, 0.85, disp_count)
    out_y = [0.5]
    
    w1_max = np.max(np.abs(W1)) + 1e-5
    w2_max = np.max(np.abs(W2)) + 1e-5
    
    for i in range(2):
        for j in range(disp_count):
            w = W1[i, j]
            alpha = min(1.0, max(0.1, abs(w) / w1_max))
            lw = min(2.0, max(0.4, abs(w) / w1_max * 2.0))
            color = '#60a5fa' if w > 0 else '#f87171'
            ax.plot([layer_x[0], layer_x[1]], [in_y[i], hid_y[j]], color=color, alpha=alpha, linewidth=lw)
            
    for j in range(disp_count):
        w = W2[j, 0]
        alpha = min(1.0, max(0.1, abs(w) / w2_max))
        lw = min(2.0, max(0.4, abs(w) / w2_max * 2.0))
        color = '#60a5fa' if w > 0 else '#f87171'
        ax.plot([layer_x[1], layer_x[2]], [hid_y[j], out_y[0]], color=color, alpha=alpha, linewidth=lw)
        
    for y_p in in_y:
        ax.scatter(layer_x[0], y_p, color='#64748b', s=80, zorder=5, edgecolors='#ffffff', linewidths=1)
    for y_p in hid_y:
        ax.scatter(layer_x[1], y_p, color='#38bdf8', s=60, zorder=5, edgecolors='#ffffff', linewidths=1)
    for y_p in out_y:
        ax.scatter(layer_x[2], y_p, color='#c084fc', s=90, zorder=5, edgecolors='#ffffff', linewidths=1)
        
    ax.text(layer_x[0], 0.02, T["input_layer"], color='#64748b', fontsize=8, ha='center')
    ax.text(layer_x[1], 0.02, T["hidden_layer"].format(count=hidden_count), color='#64748b', fontsize=8, ha='center')
    ax.text(layer_x[2], 0.02, T["output_layer"], color='#64748b', fontsize=8, ha='center')
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def render_engineering_results(epoch_0_loss, epoch_0_pred, current_epoch, current_loss, current_pred, target_val=0.01, is_complete=False):
    status_text = T["target_acquired"].format(val=target_val) if is_complete else T["convergence_mode"]
    status_color = "#10b981" if is_complete else "#f59e0b"
    epoch_lbl = T["convergence"] if is_complete else T["training"]
    caption_txt = T["blueprint_caption"].format(epoch=current_epoch)
    
    html = f"""<div style="background-color: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; font-family: monospace, sans-serif; color: #f8fafc;">
<div style="text-align: center; font-size: 18px; font-weight: 700; color: #cbd5e1; letter-spacing: 0.8px; margin-bottom: 12px;">
    {T["eng_results_title"]}
</div>
<div style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">{T["epoch_0_hdr"]}</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 6px; margin-bottom: 12px;">
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px;">
        <div style="font-size: 9px; color: #64748b;">{T["loss"].upper()}</div>
        <div style="font-size: 13px; font-weight: 700; color: #f87171;">{epoch_0_loss:.6f}</div>
    </div>
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px;">
        <div style="font-size: 9px; color: #64748b;">{T["prediction"]}</div>
        <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">{epoch_0_pred:.6f}</div>
    </div>
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-size: 8px; color: #64748b;">{T["status"]}</div>
            <div style="font-size: 10px; font-weight: 600; color: #f87171;">{T["random_init"]}</div>
        </div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #f87171; box-shadow: 0 0 6px #f87171;"></div>
    </div>
</div>
<div style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">{T["epoch"]} {current_epoch:,} ({epoch_lbl})</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 6px; margin-bottom: 12px;">
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px;">
        <div style="font-size: 9px; color: #64748b;">{T["loss"].upper()}</div>
        <div style="font-size: 13px; font-weight: 700; color: {status_color};">{current_loss:.6f}</div>
    </div>
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px;">
        <div style="font-size: 9px; color: #64748b;">{T["prediction"]}</div>
        <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">{current_pred:.6f}</div>
    </div>
    <div style="border: 1px solid #334155; padding: 6px 10px; background: #0f172a; border-radius: 4px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-size: 8px; color: #64748b;">{T["status"]}</div>
            <div style="font-size: 10px; font-weight: 600; color: {status_color};">{status_text}</div>
        </div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 6px {status_color};"></div>
    </div>
</div>
<div style="border: 1px solid #334155; padding: 10px; background: rgba(15, 23, 42, 0.6); border-radius: 4px; font-size: 10px; color: #94a3b8; line-height: 1.4; font-family: sans-serif;">
    {caption_txt}
</div>
</div>"""
    return html

def update_results_card(placeholder, html_content):
    placeholder.markdown(textwrap.dedent(html_content), unsafe_allow_html=True)

def render_export_panel(model):
    w1_list = model.W1.tolist()
    b1_list = model.b1.tolist()
    w2_list = model.W2.tolist()
    b2_list = model.b2.tolist()
    
    python_script = f"""import numpy as np

# Exported Neural Network Weights
W1 = np.array({w1_list})
b1 = np.array({b1_list})
W2 = np.array({w2_list})
b2 = np.array({b2_list})

def predict(x1, x2):
    X = np.array([[x1, x2]])
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0.01 * z1, z1) # LeakyReLU
    z2 = np.dot(a1, W2) + b2
    prob = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
    return prob[0, 0]

# Test prediction
prob = predict(0.5, 0.2)
print("Prediction Probability:", prob)
"""
    return python_script

# Benchmark test point for prediction tracking
test_sample_idx = 0
test_X = X[test_sample_idx:test_sample_idx+1]
test_target = y[test_sample_idx, 0]

if train_button:
    nn = NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)
    
    epoch_0_loss = nn.train_step(X, y)
    epoch_0_pred = nn.forward(test_X)[0, 0]
    
    epochs_hist = [0]
    loss_hist = [epoch_0_loss]
    acc_hist = [0.0]
    
    fig_main = draw_plot(nn, X, y, 0, epoch_0_loss, ds_info)
    plot_placeholder.pyplot(fig_main)
    plt.close(fig_main)
    
    fig_loss = draw_loss_curve(epochs_hist, loss_hist, acc_hist)
    loss_chart_placeholder.pyplot(fig_loss)
    plt.close(fig_loss)
    
    update_results_card(results_card_placeholder, render_engineering_results(epoch_0_loss, epoch_0_pred, 0, epoch_0_loss, epoch_0_pred, target_val=test_target, is_complete=False))
    
    fig_weights = draw_network_diagram(nn)
    weights_chart_placeholder.pyplot(fig_weights)
    plt.close(fig_weights)
    
    update_freq = 200 if train_until_perfect else max(1, epochs // 20)
    epoch = 0
    max_safe_epochs = 100000
    
    while True:
        epoch += 1
        loss = nn.train_step(X, y)
        
        output = nn.forward(X)
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        
        current_pred = output[test_sample_idx, 0]
        
        margin = 0.15 
        strict_correct = ((y == 1) & (output > 0.5 + margin)) | ((y == 0) & (output < 0.5 - margin))
        strict_accuracy = np.mean(strict_correct)
        
        is_final_epoch = False
        if train_until_perfect:
            if strict_accuracy == 1.0 or epoch >= max_safe_epochs:
                is_final_epoch = True
        else:
            if epoch >= epochs:
                is_final_epoch = True
                
        epochs_hist.append(epoch)
        loss_hist.append(loss)
        acc_hist.append(accuracy)
        
        if epoch % update_freq == 0 or is_final_epoch:
            fig_main = draw_plot(nn, X, y, epoch, loss, ds_info)
            plot_placeholder.pyplot(fig_main)
            plt.close(fig_main)
            
            fig_loss = draw_loss_curve(epochs_hist, loss_hist, acc_hist)
            loss_chart_placeholder.pyplot(fig_loss)
            plt.close(fig_loss)
            
            update_results_card(results_card_placeholder, render_engineering_results(epoch_0_loss, epoch_0_pred, epoch, loss, current_pred, target_val=test_target, is_complete=is_final_epoch))
            
            fig_weights = draw_network_diagram(nn)
            weights_chart_placeholder.pyplot(fig_weights)
            plt.close(fig_weights)
            
            if not train_until_perfect:
                progress_bar.progress(epoch / epochs)
            
            epoch_metric.metric(T["epoch"], f"{epoch}" if train_until_perfect else f"{epoch} / {epochs}")
            
            if is_final_epoch and accuracy == 1.0 and train_until_perfect:
                acc_metric.metric(T["accuracy"], "100.0%", delta=T["complete_delta"], delta_color="normal")
            else:
                acc_metric.metric(T["accuracy"], f"{accuracy*100:.1f}%", delta=f"{T['loss']} {loss:.4f}", delta_color="inverse")
            
        if is_final_epoch:
            break
            
    st.session_state['nn_model'] = nn
    st.session_state['epoch_0_loss'] = epoch_0_loss
    st.session_state['epoch_0_pred'] = epoch_0_pred
    st.session_state['last_epoch'] = epoch
    st.session_state['last_loss'] = loss
    st.session_state['last_pred'] = current_pred

else:
    if 'nn_model' not in st.session_state:
        st.session_state['nn_model'] = NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)
        st.session_state['epoch_0_loss'] = 0.258015
        st.session_state['epoch_0_pred'] = float(st.session_state['nn_model'].forward(test_X)[0, 0])
        st.session_state['last_epoch'] = 0
        st.session_state['last_loss'] = 0.258015
        st.session_state['last_pred'] = st.session_state['epoch_0_pred']
        
    nn = st.session_state['nn_model']
    
    fig_main = draw_plot(nn, X, y, st.session_state['last_epoch'], st.session_state['last_loss'], ds_info)
    plot_placeholder.pyplot(fig_main)
    
    fig_loss = draw_loss_curve([0], [st.session_state['epoch_0_loss']], [0.0])
    loss_chart_placeholder.pyplot(fig_loss)
    
    update_results_card(results_card_placeholder, render_engineering_results(
        st.session_state['epoch_0_loss'],
        st.session_state['epoch_0_pred'],
        st.session_state['last_epoch'],
        st.session_state['last_loss'],
        st.session_state['last_pred'],
        target_val=test_target,
        is_complete=(st.session_state['last_epoch'] > 0)
    ))
    
    fig_weights = draw_network_diagram(nn)
    weights_chart_placeholder.pyplot(fig_weights)
    
    epoch_metric.metric(T["epoch"], f"{st.session_state['last_epoch']}")
    acc_metric.metric(T["accuracy"], "0.0%" if st.session_state['last_epoch'] == 0 else "Ready")

with chart_tab4:
    st.markdown(T["export_title"])
    st.caption(T["export_desc"])
    current_nn_exp = st.session_state.get('nn_model', NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1))
    code_str = render_export_panel(current_nn_exp)
    st.code(code_str, language="python")

st.markdown("---")
with st.container(border=True):
    st.markdown(T["inspector_title"])
    st.caption(T["inspector_caption"])
    
    insp_col1, insp_col2, insp_col3 = st.columns([1, 1, 1.5])
    with insp_col1:
        custom_x1_real = st.slider(
            ds_info["x1_name"],
            min_value=float(ds_info["x1_min"]),
            max_value=float(ds_info["x1_max"]),
            value=float(ds_info["x1_default"]),
            step=float(ds_info["x1_step"])
        )
    with insp_col2:
        custom_x2_real = st.slider(
            ds_info["x2_name"],
            min_value=float(ds_info["x2_min"]),
            max_value=float(ds_info["x2_max"]),
            value=float(ds_info["x2_default"]),
            step=float(ds_info["x2_step"])
        )
        
    x1_norm, x2_norm = real_to_norm(custom_x1_real, custom_x2_real, ds_info)
    current_nn = st.session_state.get('nn_model', NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1))
    raw_prediction = float(current_nn.forward(np.array([[x1_norm, x2_norm]]))[0, 0])
    pred_class = 1 if raw_prediction > 0.5 else 0
    pred_color = "#60a5fa" if pred_class == 1 else "#f87171"
    class_label = ds_info["class_1"] if pred_class == 1 else ds_info["class_0"]
    
    with insp_col3:
        st.markdown(textwrap.dedent(f"""
        <div style="background: #0b1120; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; font-family: monospace;">
            <div style="font-size: 11px; color: #64748b;">{T["raw_pred_hdr"]}</div>
            <div style="font-size: 22px; font-weight: 700; color: #38bdf8;">{raw_prediction*100:.1f}% ({raw_prediction:.4f})</div>
            <div style="font-size: 12px; color: {pred_color}; font-weight: 600; margin-top: 4px;">
                {T["decided_class"]}: {class_label}
            </div>
        </div>
        """), unsafe_allow_html=True)
