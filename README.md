<div align="center">

<img src="https://raw.githubusercontent.com/James0010101/MedAI-Longevity-Platform/main/assets/header.svg" alt="MedAI Longevity Platform" width="100%"/>

<br/>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3572A5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pure NumPy](https://img.shields.io/badge/ML-Pure%20NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/Tests-38%20passed-00e68a?style=flat-square)](#-reproduce-from-scratch)
[![License](https://img.shields.io/badge/License-Research-8b949e?style=flat-square)](#)
[![Trilingual](https://img.shields.io/badge/i18n-RU%20·%20UZ%20·%20EN-ffb800?style=flat-square)](#)

**Cardiovascular risk prediction platform for Uzbekistan.**
<br/>
*Neural network from scratch · Clinical validation · Explainable AI · HL7 FHIR R4 export*

</div>

---

## What It Does

MedAI predicts **10-year coronary heart disease risk** and screens for **diabetes** using a multi-task neural network built entirely in NumPy — no sklearn, no PyTorch, no TensorFlow. Every weight, gradient, and activation is computed from first principles.

The clinical pipeline combines the neural network's predictions with the **D'Agostino 2008** analytical formula, **WHO mortality tables for Uzbekistan** (2023), and **ESC/WHO clinical guidelines** to produce actionable, personalized health recommendations ranked by impact.

---

## Architecture

<div align="center">

<img src="https://raw.githubusercontent.com/James0010101/MedAI-Longevity-Platform/main/assets/architecture.svg" alt="System Architecture" width="100%"/>

</div>

---

## Validated Metrics

Every number below is produced by an automated validation script on a held-out test set (*N* = 636), with no hardcoded constants:

| Target | MedAI Neural Net | Baseline LogReg | Reference (D'Agostino 2008) | Brier Score |
|:---|:---:|:---:|:---:|:---:|
| **10-year CHD risk** | **0.675** [0.607, 0.735] | 0.698 [0.638, 0.758] | 0.700 [0.641, 0.760] | **0.1210** |
| **Diabetes screening** | **0.884** [0.738, 0.998] | 0.827 | — | **0.0242** |

<details>
<summary><strong>Operating Characteristics at Clinical Thresholds</strong></summary>
<br/>

| Threshold | Sensitivity | Specificity |
|:---|:---:|:---:|
| **10% (Intermediate risk)** | 75.26% | 53.62% |
| **20% (High clinical risk)** | 40.21% | 82.56% |

</details>

---

## Key Capabilities

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Neural Network (Pure NumPy)
Multi-task MLP with Swish activation, Dropout regularization, and Adam optimizer. No ML frameworks — every operation hand-coded for educational transparency.

### 🔬 Explainable AI
Integrated Gradients computes each factor's contribution in percentage points relative to a healthy reference of the same age and sex.

### 💊 Personalized Recommendations
Clinical guidelines (WHO/ESC) merged with neural network delta simulations: steps ranked by risk reduction magnitude.

</td>
<td width="50%" valign="top">

### ❤️ Cardiovascular Age
Closed-form D'Agostino 2008 formula. Critical bug in the original codebase (constant collapse) identified and fixed.

### 📊 WHO Life Tables
Official 2023 demographic grid for Uzbekistan with vascular age correction for life expectancy estimation.

### 📦 Interoperability
Printable HTML/PDF report + HL7 FHIR R4 bundle with verified LOINC codes. Unit converter: mmol/L ⇄ mg/dL.

</td>
</tr>
</table>

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/James0010101/MedAI-Longevity-Platform.git
cd "MedAI Longevity Platform"
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Launch
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## 🔬 Reproduce From Scratch

Any researcher can reproduce training and validation in 4 commands:

```bash
python scripts/prepare_data.py          # Stratified split 70/15/15
python scripts/fetch_who_life_table.py  # WHO mortality tables for Uzbekistan
python scripts/train_model.py           # Train neural network, compute ROC-AUC with CI
pytest -q                               # Run all 38 automated tests
```

---

## Repository Structure

```
MedAI Longevity Platform/
├── streamlit_app.py            # Main entry: config, i18n, st.navigation
├── neural_network.py           # Pure NumPy MultiTask MLP (Swish, Dropout, Adam, IG)
├── app_pages/
│   ├── screening.py            # Risk form, metrics, top-3 actions, XAI, export
│   ├── methodology.py          # Loss curves, ROC-AUC, calibration
│   └── about.py                # Project context, roadmap
├── core/
│   ├── clinical.py             # D'Agostino 2008, Heart Age, WHO tables
│   ├── risk_model.py           # Inference, Integrated Gradients, simulations
│   ├── recommendations.py      # Clinical rules + neural net delta ranking
│   ├── report.py               # HTML report + HL7 FHIR R4 bundle
│   ├── units.py                # mmol/L ⇄ mg/dL bidirectional conversion
│   └── ...                     # config, i18n, thresholds
├── scripts/                    # Data prep, WHO fetch, training
├── tests/                      # 38 automated tests (clinical, ML, XAI, UI)
├── data/                       # Framingham dataset, WHO tables
├── models/                     # Weights, metrics, model card
└── i18n/                       # ru.json, uz.json, en.json
```

---

## Medical & Legal Disclaimer

MedAI Longevity Platform is a **research and educational prototype**. It is intended for primary risk stratification and public health awareness. It **does not replace a physician consultation**, does not make final clinical diagnoses, and does not prescribe medication.

---

<div align="center">
<sub>Built by <a href="https://github.com/James0010101">Jahongir Sadullayev</a> · Tashkent, Uzbekistan 🇺🇿</sub>
</div>
