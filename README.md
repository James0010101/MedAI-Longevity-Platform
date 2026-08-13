# MedAI Longevity — Multi-Task Neural Network Playground


## Overview

MedAI Longevity is an interactive Streamlit application that demonstrates a **multi-task neural network** built from scratch in pure NumPy. The network processes 10 clinical biomarkers through shared hidden layers and simultaneously predicts 4 clinical targets:

1. **Cardiac Disease Risk** (binary classification)
2. **Type-II Diabetes Risk** (binary classification)
3. **Life Expectancy** (regression, years)
4. **Biological Vascular Age** (regression, years)

## Key Features

- 🧠 **Pure NumPy neural network** — forward/backward pass, Adam optimizer, He/Xavier init, 5 activation functions
- 🎯 **Multi-head architecture** — shared latent representation → 4 task-specific output heads
- 🔍 **Explainable AI (XAI)** — per-biomarker gradient sensitivity attributions
- 🔮 **"What-If" Simulator** — toggle lifestyle interventions and see predicted outcomes change in real time
- 📊 **3D PCA visualization** — latent feature space clustering
- 📋 **AI Clinical Recommendations** — rule-based personalized clinical suggestions
- 📥 **HTML Report Generator** — downloadable diagnostic report
- 🌐 **Bilingual UI** — full Russian/English internationalization
- 📈 **Train/Test split** — 80/20 hold-out evaluation to show real generalization metrics

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Streamlit UI (entry point)
├── neural_network.py      # Multi-task neural network (pure NumPy)
├── dataset_manager.py     # Synthetic clinical dataset generator
├── test_neural_network.py # Automated tests
├── requirements.txt       # Pinned dependencies
├── .streamlit/config.toml # Streamlit executive light theme configuration
└── README.md              # This file
```

## Dataset

The synthetic dataset is generated based on epidemiological risk models inspired by:
- **Framingham Heart Study** cardiovascular risk factors
- **NHANES** metabolic and anthropometric distributions
- **WHO** mortality statistics

**10 Input Biomarkers:** Age, Sex, Resting BP, Cholesterol, Fasting Glucose, Max Heart Rate, ST Depression, BMI, Physical Activity, Smoking Status.

## Technical Details

- **Optimizer:** Adam (β₁=0.9, β₂=0.999)
- **Loss:** Weighted multi-task loss (BCE for classification heads, MSE for regression heads)
- **Regularization:** L2 weight decay, gradient clipping
- **Initialization:** He (ReLU family), Xavier (others)

## License

Educational project. Use at your own risk.
