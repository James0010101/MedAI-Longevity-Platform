# Model Card: MedAI Multi-Task Cardiovascular & Diabetes Screener

## Model Details
- **Developer:** MedAI Longevity Platform Initiative
- **Model Version:** 2.0.0
- **Model Date:** October 2026
- **Model Type:** Multi-Task Feedforward Neural Network (Pure NumPy Implementation)
- **Architecture:** 13 input features $\to$ Dense(32, Swish) $\to$ Dense(16, Swish) $\to$ Dual Classification Heads:
  - **Head 1 (`chd`):** 10-year risk of incident Coronary Heart Disease (Sigmoid activation)
  - **Head 2 (`diabetes`):** Prevalent diabetes screening indicator (Sigmoid activation)
- **Regularization:** Inverted Dropout ($p = 0.10$) + L2 weight decay ($10^{-4}$)
- **Optimization:** Adam optimizer ($\text{lr} = 0.005$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, gradient clipping at $5.0$)
- **Explainability:** Analytical Path-Integrated Gradients (50 steps) from age- and sex-matched healthy baseline profile

## Intended Use
- **Primary Use:** Non-invasive cardiovascular risk screening, educational risk communication, and health promotion for primary care and community health contexts.
- **Intended Population:** Adults aged 30–75 being evaluated for cardiovascular disease prevention.
- **Out-of-Scope Use:**
  - Autonomous clinical diagnosis of acute coronary syndromes or diabetes mellitus.
  - Replacement for laboratory diagnostic evaluations (e.g., coronary angiography, oral glucose tolerance test, HbA1c).
  - Prescription of medical drug therapy without physician consultation.

## Training & Evaluation Data
- **Cohort:** Framingham Heart Study public educational cohort (National Heart, Lung, and Blood Institute).
- **Total Records:** 4,240 individuals.
- **Data Split:** Stratified 70% Train (2,968), 15% Validation (636), 15% Test (636).
- **Features Used (13):** Biological sex, age, current smoker flag, cigarettes per day, blood pressure medication, prevalent stroke, prevalent hypertension, total cholesterol (mg/dL), systolic blood pressure (mmHg), diastolic blood pressure (mmHg), body mass index (BMI), resting heart rate, fasting blood glucose (mg/dL).
- **Imputation:** Median values calculated strictly from the training partition.

## Performance Metrics (Evaluated on Holdout Test Set $N = 636$)

### 10-Year Coronary Heart Disease (CHD) Head
| Model / Benchmark | ROC-AUC (95% Bootstrap CI) | Brier Score |
|:---|:---:|:---:|
| **MedAI Neural Network** | **0.675** [0.607, 0.735] | **0.1210** |
| Logistic Regression Baseline | 0.698 [0.638, 0.758] | 0.1185 |
| D'Agostino 2008 General CVD Reference | 0.700 [0.641, 0.760] | — |

*Clinical Decision Characteristics for MedAI Neural Network:*
- **10% Risk Threshold (Intermediate Risk):**
  - Sensitivity: **75.26%**
  - Specificity: **53.62%**
  - Positive Predictive Value (PPV): **22.50%**
- **20% Risk Threshold (High Risk):**
  - Sensitivity: **40.21%**
  - Specificity: **82.56%**
  - Positive Predictive Value (PPV): **29.32%**

### Diabetes Screening Head
| Model | ROC-AUC (95% Bootstrap CI) | Brier Score |
|:---|:---:|:---:|
| **MedAI Neural Network** | **0.884** [0.738, 0.998] | **0.0242** |
| Logistic Regression Baseline | 0.827 | 0.0261 |

## Ethical Considerations & Fairness
- **Sociodemographic Representation:** The training data originates from the Framingham, Massachusetts cohort (predominantly Caucasian population).
- **Localization Requirement:** Application in the Republic of Uzbekistan requires recalibration using local epidemiological data from Tashkent and regional health centers.
- **Epistemic Uncertainty:** The model outputs Monte Carlo Dropout uncertainty bounds ($\pm\sigma$) to indicate confidence rather than false certainty.
