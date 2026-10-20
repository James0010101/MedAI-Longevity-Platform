"""
Training script for MedAI Multi-Task Neural Network.
1. Trains multi-task network [13 -> 32 -> 16 -> 2 heads] with early stopping.
2. Trains LogisticRegression baselines for both heads.
3. Computes rigorous test metrics: ROC-AUC with 95% bootstrap CI, Brier score, calibration curve,
   sensitivity, specificity, and PPV at thresholds 10% and 20%.
4. Saves models/model_weights.npz, models/metrics.json, models/model_card.json.
"""
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

# Ensure root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import DATA_DIR, FEATURE_NAMES, MODELS_DIR
from neural_network import MultiTaskNeuralNetwork
from core.clinical import calculate_dagostino_risk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_NPZ = DATA_DIR / "processed_data.npz"
FRAMINGHAM_CSV = DATA_DIR / "framingham.csv"
WEIGHTS_PATH = MODELS_DIR / "model_weights.npz"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

def compute_bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) > 1:
            aucs.append(roc_auc_score(sample_y, y_pred[idx]))
    if not aucs:
        return 0.5, 0.5
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def compute_binary_metrics_at_threshold(y_true, y_pred, threshold):
    pred_bin = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (pred_bin == 1))
    fp = np.sum((y_true == 0) & (pred_bin == 1))
    fn = np.sum((y_true == 1) & (pred_bin == 0))
    tn = np.sum((y_true == 0) & (pred_bin == 0))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_NPZ.exists():
        logger.error(f"Missing {PROCESSED_NPZ}. Run scripts/prepare_data.py first.")
        return False

    with np.load(PROCESSED_NPZ, allow_pickle=False) as d:
        X_train_norm = d["X_train_norm"]
        X_val_norm = d["X_val_norm"]
        X_test_norm = d["X_test_norm"]
        X_test_raw = d["X_test_raw"]

        y_chd_train = d["y_chd_train"]
        y_chd_val = d["y_chd_val"]
        y_chd_test = d["y_chd_test"]

        y_dm_train = d["y_dm_train"]
        y_dm_val = d["y_dm_val"]
        y_dm_test = d["y_dm_test"]

        means = d["means"]
        stds = d["stds"]
        medians = d["medians"]

    logger.info(f"Loaded processed data: Train={len(X_train_norm)}, Val={len(X_val_norm)}, Test={len(X_test_norm)}")

    # 1. Initialize Neural Network
    model = MultiTaskNeuralNetwork(
        input_dim=len(FEATURE_NAMES),
        hidden_sizes=[32, 16],
        heads=["chd", "diabetes"],
        dropout=0.1,
        l2_reg=1e-4,
        lr=0.005,
        seed=42
    )

    # Training loop with mini-batches and early stopping
    batch_size = 256
    max_epochs = 3000
    patience = 120
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_weights_shared = None
    best_biases_shared = None
    best_weights_heads = None
    best_biases_heads = None

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": []
    }

    n_train = len(X_train_norm)
    y_train_dict = {"chd": y_chd_train, "diabetes": y_dm_train}
    y_val_dict = {"chd": y_chd_val, "diabetes": y_dm_val}

    logger.info("Beginning training...")
    rng_shuffle = np.random.default_rng(42)

    for epoch in range(1, max_epochs + 1):
        perm = rng_shuffle.permutation(n_train)
        X_shuffled = X_train_norm[perm]
        y_chd_shuffled = y_chd_train[perm]
        y_dm_shuffled = y_dm_train[perm]

        for i in range(0, n_train, batch_size):
            X_b = X_shuffled[i:i + batch_size]
            y_b = {
                "chd": y_chd_shuffled[i:i + batch_size],
                "diabetes": y_dm_shuffled[i:i + batch_size]
            }
            model.train_step(X_b, y_b)

        # Evaluate at epoch end
        train_probs = model.predict(X_train_norm)
        val_probs = model.predict(X_val_norm)

        train_loss = model.compute_loss(train_probs, y_train_dict)
        val_loss = model.compute_loss(val_probs, y_val_dict)

        if epoch % 5 == 0 or epoch == 1:
            history["epochs"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights_shared = [W.copy() for W in model.W_shared]
            best_biases_shared = [b.copy() for b in model.b_shared]
            best_weights_heads = {h: W.copy() for h, W in model.W_heads.items()}
            best_biases_heads = {h: b.copy() for h, b in model.b_heads.items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}.")
                break

    # Restore best weights
    model.W_shared = best_weights_shared
    model.b_shared = best_biases_shared
    model.W_heads = best_weights_heads
    model.b_heads = best_biases_heads

    # 2. Train baseline Logistic Regression
    logger.info("Training Logistic Regression baselines...")
    lr_chd = LogisticRegression(max_iter=1000, random_state=42)
    lr_chd.fit(X_train_norm, y_chd_train)
    lr_chd_test_pred = lr_chd.predict_proba(X_test_norm)[:, 1]

    lr_dm = LogisticRegression(max_iter=1000, random_state=42)
    lr_dm.fit(X_train_norm, y_dm_train)
    lr_dm_test_pred = lr_dm.predict_proba(X_test_norm)[:, 1]

    # 3. D'Agostino benchmark risk on test set (reference only)
    logger.info("Computing D'Agostino risk on test set as reference...")
    dagostino_test_pred = []
    # raw features order: male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, totChol, sysBP, diaBP, BMI, heartRate, glucose
    feat_map = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for row in X_test_raw:
        r_dag = calculate_dagostino_risk(
            age=row[feat_map["age"]],
            sex=int(row[feat_map["male"]]),
            sbp=row[feat_map["sysBP"]],
            bp_meds=int(row[feat_map["BPMeds"]]),
            tot_chol=row[feat_map["totChol"]],
            hdl=45.0,
            smoker=int(row[feat_map["currentSmoker"]]),
            diabetes=0
        )
        dagostino_test_pred.append(r_dag)
    dagostino_test_pred = np.array(dagostino_test_pred)

    # 4. Evaluate Test Set Predictions for Neural Network
    test_probs = model.predict(X_test_norm)
    nn_chd_test_pred = test_probs["chd"].ravel()
    nn_dm_test_pred = test_probs["diabetes"].ravel()

    # Compute Test Metrics: CHD Head
    nn_chd_auc = float(roc_auc_score(y_chd_test, nn_chd_test_pred))
    nn_chd_ci = compute_bootstrap_auc_ci(y_chd_test, nn_chd_test_pred)
    nn_chd_brier = float(brier_score_loss(y_chd_test, nn_chd_test_pred))

    lr_chd_auc = float(roc_auc_score(y_chd_test, lr_chd_test_pred))
    lr_chd_ci = compute_bootstrap_auc_ci(y_chd_test, lr_chd_test_pred)
    lr_chd_brier = float(brier_score_loss(y_chd_test, lr_chd_test_pred))

    dag_chd_auc = float(roc_auc_score(y_chd_test, dagostino_test_pred))
    dag_chd_ci = compute_bootstrap_auc_ci(y_chd_test, dagostino_test_pred)
    dag_chd_brier = float(brier_score_loss(y_chd_test, dagostino_test_pred))

    # Calibration curves (10 bins)
    cal_prob_true, cal_prob_pred = calibration_curve(y_chd_test, nn_chd_test_pred, n_bins=10, strategy="uniform")

    # Threshold metrics for CHD
    chd_threshold_10 = compute_binary_metrics_at_threshold(y_chd_test, nn_chd_test_pred, 0.10)
    chd_threshold_20 = compute_binary_metrics_at_threshold(y_chd_test, nn_chd_test_pred, 0.20)

    # Compute Test Metrics: Diabetes Head
    nn_dm_auc = float(roc_auc_score(y_dm_test, nn_dm_test_pred))
    nn_dm_ci = compute_bootstrap_auc_ci(y_dm_test, nn_dm_test_pred)
    nn_dm_brier = float(brier_score_loss(y_dm_test, nn_dm_test_pred))

    lr_dm_auc = float(roc_auc_score(y_dm_test, lr_dm_test_pred))
    lr_dm_ci = compute_bootstrap_auc_ci(y_dm_test, lr_dm_test_pred)
    lr_dm_brier = float(brier_score_loss(y_dm_test, lr_dm_test_pred))

    cal_dm_true, cal_dm_pred = calibration_curve(y_dm_test, nn_dm_test_pred, n_bins=10, strategy="uniform")

    test_chd_prev = float(np.mean(y_chd_test))
    test_dm_prev = float(np.mean(y_dm_test))

    logger.info("=== TEST SET RESULTS ===")
    logger.info(f"CHD AUC: NN = {nn_chd_auc:.3f} [{nn_chd_ci[0]:.3f}, {nn_chd_ci[1]:.3f}], LR = {lr_chd_auc:.3f}, D'Agostino = {dag_chd_auc:.3f}")
    logger.info(f"CHD Brier: NN = {nn_chd_brier:.4f}, LR = {lr_chd_brier:.4f}")
    logger.info(f"Diabetes AUC: NN = {nn_dm_auc:.3f} [{nn_dm_ci[0]:.3f}, {nn_dm_ci[1]:.3f}], LR = {lr_dm_auc:.3f}")

    # Build metrics.json dictionary
    metrics_data = {
        "chd": {
            "test_prevalence": round(test_chd_prev, 4),
            "nn": {
                "auc": round(nn_chd_auc, 4),
                "auc_ci_95": [round(nn_chd_ci[0], 4), round(nn_chd_ci[1], 4)],
                "brier": round(nn_chd_brier, 4),
                "threshold_10": chd_threshold_10,
                "threshold_20": chd_threshold_20,
            },
            "logistic_regression": {
                "auc": round(lr_chd_auc, 4),
                "auc_ci_95": [round(lr_chd_ci[0], 4), round(lr_chd_ci[1], 4)],
                "brier": round(lr_chd_brier, 4),
            },
            "dagostino_reference": {
                "auc": round(dag_chd_auc, 4),
                "auc_ci_95": [round(dag_chd_ci[0], 4), round(dag_chd_ci[1], 4)],
                "brier": round(dag_chd_brier, 4),
                "note": "General CVD outcome (different target from CHD-only), calculated with default HDL=45"
            },
            "calibration": {
                "prob_true": [round(float(x), 4) for x in cal_prob_true],
                "prob_pred": [round(float(x), 4) for x in cal_prob_pred]
            }
        },
        "diabetes": {
            "test_prevalence": round(test_dm_prev, 4),
            "nn": {
                "auc": round(nn_dm_auc, 4),
                "auc_ci_95": [round(nn_dm_ci[0], 4), round(nn_dm_ci[1], 4)],
                "brier": round(nn_dm_brier, 4),
            },
            "logistic_regression": {
                "auc": round(lr_dm_auc, 4),
                "auc_ci_95": [round(lr_dm_ci[0], 4), round(lr_dm_ci[1], 4)],
                "brier": round(lr_dm_brier, 4),
            },
            "calibration": {
                "prob_true": [round(float(x), 4) for x in cal_dm_true],
                "prob_pred": [round(float(x), 4) for x in cal_dm_pred]
            }
        },
        "training": {
            "best_epoch": int(best_epoch),
            "total_epochs": int(history["epochs"][-1]),
            "best_val_loss": round(float(best_val_loss), 4),
            "history": history
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {METRICS_PATH}")

    # Metadata to save inside model_weights.npz
    metadata = {
        "feature_names": FEATURE_NAMES,
        "means": [float(x) for x in means],
        "stds": [float(x) for x in stds],
        "medians": [float(x) for x in medians],
        "trained_at": datetime.now().isoformat(),
        "best_epoch": int(best_epoch),
        "chd_auc": round(nn_chd_auc, 4),
        "diabetes_auc": round(nn_dm_auc, 4)
    }

    model.save(WEIGHTS_PATH, metadata=metadata)
    logger.info(f"Saved {WEIGHTS_PATH}")

    # Compute data hash
    data_hash = ""
    if FRAMINGHAM_CSV.exists():
        with open(FRAMINGHAM_CSV, "rb") as f:
            data_hash = hashlib.sha256(f.read()).hexdigest()

    # Model Card
    model_card = {
        "model_name": "MedAI Multi-Task Cardiovascular & Diabetes Screener",
        "version": "2.0.0",
        "architecture": f"Feedforward MLP [13 -> 32 -> 16 -> 2 heads (chd, diabetes)]",
        "activation": "Swish",
        "optimizer": "Adam (lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=5.0)",
        "regularization": "Inverted Dropout (0.1) + L2 (1e-4)",
        "training_framework": "Pure NumPy implementation",
        "trained_date": datetime.now().isoformat(),
        "dataset": {
            "name": "Framingham Heart Study (Public Educational Cohort)",
            "sha256": data_hash,
            "train_samples": len(X_train_norm),
            "val_samples": len(X_val_norm),
            "test_samples": len(X_test_norm),
            "features_count": len(FEATURE_NAMES),
            "features_list": FEATURE_NAMES
        },
        "performance_summary": {
            "chd_roc_auc": round(nn_chd_auc, 4),
            "chd_auc_ci_95": [round(nn_chd_ci[0], 4), round(nn_chd_ci[1], 4)],
            "diabetes_roc_auc": round(nn_dm_auc, 4),
            "diabetes_auc_ci_95": [round(nn_dm_ci[0], 4), round(nn_dm_ci[1], 4)]
        },
        "intended_use": "Screening and health education tool for 10-year coronary heart disease risk and current diabetes status. Research prototype not intended for autonomous medical diagnosis."
    }

    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {MODEL_CARD_PATH}")

    return True

if __name__ == "__main__":
    success = train_and_evaluate()
    sys.exit(0 if success else 1)
