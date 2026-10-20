"""
Prepares the Framingham dataset:
1. Loads data/framingham.csv
2. Stratified train/val/test split (70/15/15) by TenYearCHD with random_state=42
3. Imputes missing values using medians computed strictly on train
4. Standardizes (z-score) features using statistics computed strictly on train
5. Saves split indices and processed datasets
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add root directory to sys.path to import core
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import DATA_DIR, FEATURE_NAMES

INPUT_CSV = DATA_DIR / "framingham.csv"
SPLIT_INDICES_NPZ = DATA_DIR / "split_indices.npz"
PROCESSED_DATA_NPZ = DATA_DIR / "processed_data.npz"

def prepare_data():
    if not INPUT_CSV.exists():
        logger.error(f"Input file {INPUT_CSV} does not exist.")
        return False

    logger.info(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    n_samples = len(df)
    logger.info(f"Loaded {n_samples} rows.")

    # Validate target columns
    if "TenYearCHD" not in df.columns or "diabetes" not in df.columns:
        logger.error("Missing target columns TenYearCHD or diabetes.")
        return False

    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            logger.error(f"Missing required feature: {feat}")
            return False

    indices = np.arange(n_samples)
    y_chd_all = df["TenYearCHD"].values.astype(np.float32)
    y_dm_all = df["diabetes"].values.astype(np.float32)

    # 1. Stratified split: 70% train, 15% val, 15% test
    # First split off 30% for val+test
    train_idx, val_test_idx, _, y_val_test_chd = train_test_split(
        indices,
        y_chd_all,
        test_size=0.30,
        stratify=y_chd_all,
        random_state=42
    )

    # Then split val+test 50/50 -> 15% val, 15% test
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=0.50,
        stratify=y_val_test_chd,
        random_state=42
    )

    logger.info(f"Split sizes: Train={len(train_idx)} ({len(train_idx)/n_samples:.1%}), "
                f"Val={len(val_idx)} ({len(val_idx)/n_samples:.1%}), "
                f"Test={len(test_idx)} ({len(test_idx)/n_samples:.1%})")

    # Save split indices
    np.savez_compressed(
        SPLIT_INDICES_NPZ,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )
    logger.info(f"Saved split indices to {SPLIT_INDICES_NPZ}")

    # Extract feature matrix
    df_features = df[FEATURE_NAMES].copy()

    # 2. Imputation medians computed STRICTLY on train
    train_features = df_features.iloc[train_idx].copy()
    
    medians = {}
    for col in FEATURE_NAMES:
        if col == "cigsPerDay":
            # For non-smokers, cigsPerDay is 0
            # For smokers, compute median among smokers who smoke > 0
            smokers_cigs = train_features.loc[train_features["currentSmoker"] == 1, "cigsPerDay"]
            smoker_median = smokers_cigs.median()
            medians["cigsPerDay_smoker"] = float(smoker_median)
            medians["cigsPerDay"] = 0.0
        else:
            med_val = float(train_features[col].median())
            medians[col] = med_val

    # Apply imputation function
    def impute_df(df_part):
        d = df_part.copy()
        # rule: if currentSmoker == 0, cigsPerDay = 0
        non_smoker_mask = (d["currentSmoker"] == 0)
        d.loc[non_smoker_mask, "cigsPerDay"] = 0.0
        
        for col in FEATURE_NAMES:
            if col == "cigsPerDay":
                # missing for smokers
                d["cigsPerDay"] = d["cigsPerDay"].fillna(medians["cigsPerDay_smoker"])
            else:
                d[col] = d[col].fillna(medians[col])
        return d.values.astype(np.float32)

    X_train_raw = impute_df(df_features.iloc[train_idx])
    X_val_raw = impute_df(df_features.iloc[val_idx])
    X_test_raw = impute_df(df_features.iloc[test_idx])

    # 3. Standardization (z-score) parameters computed STRICTLY on train
    means = np.mean(X_train_raw, axis=0)
    stds = np.std(X_train_raw, axis=0)
    # Guard against zero std
    stds[stds < 1e-8] = 1.0

    X_train_norm = (X_train_raw - means) / stds
    X_val_norm = (X_val_raw - means) / stds
    X_test_norm = (X_test_raw - means) / stds

    y_chd_train = y_chd_all[train_idx]
    y_chd_val = y_chd_all[val_idx]
    y_chd_test = y_chd_all[test_idx]

    y_dm_train = y_dm_all[train_idx]
    y_dm_val = y_dm_all[val_idx]
    y_dm_test = y_dm_all[test_idx]

    logger.info("Computed normalization stats on train:")
    for i, col in enumerate(FEATURE_NAMES):
        logger.info(f"  {col:16s}: median={medians.get(col, 0):.2f}, mean={means[i]:.2f}, std={stds[i]:.2f}")

    # Save processed data
    np.savez_compressed(
        PROCESSED_DATA_NPZ,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_train_norm=X_train_norm,
        X_val_norm=X_val_norm,
        X_test_norm=X_test_norm,
        y_chd_train=y_chd_train,
        y_chd_val=y_chd_val,
        y_chd_test=y_chd_test,
        y_dm_train=y_dm_train,
        y_dm_val=y_dm_val,
        y_dm_test=y_dm_test,
        feature_names=np.array(FEATURE_NAMES),
        means=means,
        stds=stds,
        medians=np.array([medians[c] for c in FEATURE_NAMES]),
        cigs_smoker_median=medians["cigsPerDay_smoker"]
    )
    logger.info(f"Saved processed data to {PROCESSED_DATA_NPZ}")
    return True

if __name__ == "__main__":
    success = prepare_data()
    sys.exit(0 if success else 1)
