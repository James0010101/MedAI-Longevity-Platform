import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Min-max normalization helper with small epsilon to prevent division by zero
def normalize(X, mins, maxs):
    return (X - mins) / (maxs - mins + 1e-8)

def calculate_framingham_vascular_age(age, sex, bp, chol, smoking, diabetes):
    """
    Computes Vascular Age using the official Framingham Heart Study 
    multivariable risk equations (D'Agostino et al., Circulation 2008).
    Analytical closed-form solution.
    """
    age = np.clip(np.asarray(age, dtype=float), 20.0, 95.0)
    sex = np.asarray(sex, dtype=float)  # 1 = male, 0 = female
    bp = np.clip(np.asarray(bp, dtype=float), 80.0, 240.0)
    chol = np.clip(np.asarray(chol, dtype=float), 100.0, 450.0)
    smoking = np.asarray(smoking, dtype=float)
    diabetes = np.asarray(diabetes, dtype=float)

    is_male = (sex >= 0.5)
    hdl_m, hdl_f = 50.0, 60.0

    ln_age = np.log(age)
    ln_chol = np.log(chol)
    ln_sbp = np.log(bp)

    # Men (D'Agostino 2008)
    sum_beta_m = ln_age * 3.06117 + ln_chol * 1.1237 - np.log(hdl_m) * 0.93263 + ln_sbp * 1.93303 + smoking * 0.65451 + diabetes * 0.57367
    opt_const_m = np.log(180.0) * 1.1237 - np.log(50.0) * 0.93263 + np.log(125.0) * 1.93303
    vasc_age_m = np.exp((sum_beta_m - opt_const_m) / 3.06117)

    # Women (D'Agostino 2008)
    sum_beta_f = ln_age * 2.32888 + ln_chol * 1.20904 - np.log(hdl_f) * 0.70833 + ln_sbp * 2.82263 + smoking * 0.52873 + diabetes * 0.69154
    opt_const_f = np.log(180.0) * 1.20904 - np.log(60.0) * 0.70833 + np.log(125.0) * 2.82263
    vasc_age_f = np.exp((sum_beta_f - opt_const_f) / 2.32888)

    vasc_age = np.where(is_male, vasc_age_m, vasc_age_f)
    return np.clip(vasc_age, 20.0, 95.0)


def calculate_actuarial_life_expectancy(age, sex, vascular_age, smoking, diabetes):
    """
    Computes estimated life expectancy based on WHO/CDC baseline longevity tables
    adjusted for vascular acceleration, tobacco hazard, and diabetes status.
    """
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    base_life = np.where(sex >= 0.5, 79.0, 83.0)

    vasc_delta = vascular_age - age
    life_exp = base_life - 0.40 * np.maximum(0, vasc_delta) + 0.15 * np.maximum(0, -vasc_delta)
    life_exp = life_exp - 3.5 * smoking - 3.0 * diabetes
    return np.clip(life_exp, np.maximum(55.0, age + 2.0), 96.0)


def get_medical_dataset(n_samples=500, seed=42, test_ratio=0.2, source="synthetic", csv_file=None):
    rng = np.random.default_rng(seed)

    # Reference physiological bounds for feature normalization
    mins = np.array([1.0, 0.0, 50.0, 50.0, 40.0, 40.0, 0.0, 10.0, 0.0, 0.0])
    maxs = np.array([120.0, 1.0, 250.0, 500.0, 400.0, 220.0, 6.0, 60.0, 1.0, 1.0])

    if source == "framingham":
        csv_path = os.path.join(os.path.dirname(__file__), "framingham.csv")
        if not os.path.exists(csv_path):
            logger.warning("framingham.csv not found locally, falling back to synthetic.")
            return get_medical_dataset(n_samples=n_samples, seed=seed, test_ratio=test_ratio, source="synthetic")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            for col in ['totChol', 'glucose', 'BMI', 'heartRate', 'cigsPerDay', 'BPMeds']:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            age = df['age'].values.astype(float)
            sex = df['male'].values.astype(float)
            bp = df['sysBP'].values.astype(float)
            chol = df['totChol'].values.astype(float)
            glucose = df['glucose'].values.astype(float)
            heart_rate = df['heartRate'].values.astype(float)
            max_hr = np.clip(208.0 - 0.7 * age + 0.15 * (heart_rate - 72.0), 60.0, 220.0)
            st_dep = np.where(df['prevalentHyp'].values == 1, 1.2, 0.2) + np.where(df['prevalentStroke'].values == 1, 1.0, 0.0)
            bmi = df['BMI'].values.astype(float)
            activity = np.clip(0.7 - 0.008 * (bmi - 24.0) - 0.1 * df['prevalentHyp'].values, 0.1, 1.0)
            smoking = df['currentSmoker'].values.astype(float)

            raw_X = np.column_stack([age, sex, bp, chol, glucose, max_hr, st_dep, bmi, activity, smoking])
            y_cardiac = df['TenYearCHD'].values.astype(float).reshape(-1, 1)
            y_diabetes = df['diabetes'].values.astype(float).reshape(-1, 1)

            vasc_age = calculate_framingham_vascular_age(age, sex, bp, chol, smoking, y_diabetes.ravel())
            vascular_age = vasc_age.reshape(-1, 1)

            life_exp = calculate_actuarial_life_expectancy(age, sex, vasc_age, smoking, y_diabetes.ravel())
            life_expectancy = life_exp.reshape(-1, 1)
            n_samples = len(raw_X)
        except Exception as e:
            logger.warning(f"Failed to process framingham dataset ({e}), falling back to synthetic.")
            return get_medical_dataset(n_samples=n_samples, seed=seed, test_ratio=test_ratio, source="synthetic")

    elif source == "fused_master":
        try:
            d_framingham = get_medical_dataset(source="framingham")
            raw_X = d_framingham["raw_X"]
            y_cardiac = d_framingham["y_cardiac"]
            y_diabetes = d_framingham["y_diabetes"]
            life_expectancy = d_framingham["y_life_expectancy"]
            vascular_age = d_framingham["y_vascular_age"]
            n_samples = len(raw_X)
        except Exception as e:
            logger.warning(f"Failed to load fused_master dataset ({e}), falling back to synthetic source.")
            return get_medical_dataset(n_samples=1000, seed=seed, test_ratio=test_ratio, source="synthetic")

    elif source == "pima_diabetes":
        try:
            from sklearn.datasets import fetch_openml
            pima = fetch_openml(data_id=37, as_frame=True, parser="auto")
            df = pima.data
            y_sr = pima.target

            glucose = df.iloc[:, 1].values.astype(float)
            bp = np.where(df.iloc[:, 2].values > 0, df.iloc[:, 2].values.astype(float), 120.0)
            bmi = np.where(df.iloc[:, 5].values > 0, df.iloc[:, 5].values.astype(float), 25.0)
            age = df.iloc[:, 7].values.astype(float)

            sex = np.zeros_like(age)
            chol = 180.0 + 0.5 * (glucose - 100) + rng.normal(0, 10, len(age))
            max_hr = 160.0 - 0.5 * (age - 30) + rng.normal(0, 8, len(age))
            st_dep = np.clip(rng.exponential(scale=0.5, size=len(age)), 0.0, 4.0)
            activity = np.clip(0.6 - 0.005 * (bmi - 25), 0.1, 1.0)
            smoking = (rng.uniform(0, 1, len(age)) < 0.20).astype(float)

            raw_X = np.column_stack([age, sex, bp, chol, glucose, max_hr, st_dep, bmi, activity, smoking])
            y_diabetes = (y_sr.values.astype(int) == 1).astype(float).reshape(-1, 1)

            cardiac_score = 0.03 * (age - 45) + 0.02 * (bp - 120) + 0.015 * (chol - 200) + 0.6 * st_dep
            y_cardiac = (1 / (1 + np.exp(-cardiac_score)) > 0.5).astype(float).reshape(-1, 1)
            life_expectancy = np.clip(85.0 - 0.07 * (bp - 120) - 0.05 * (glucose - 100) - 3.0 * y_diabetes.ravel(), 58.0, 92.0).reshape(-1, 1)
            vascular_age = np.clip(age + 0.15 * (bp - 120) + 0.06 * (chol - 200), 25.0, 95.0).reshape(-1, 1)
            n_samples = len(raw_X)
        except Exception as e:
            logger.warning(f"Failed to load pima_diabetes dataset ({e}), falling back to synthetic source.")
            return get_medical_dataset(n_samples=n_samples, seed=seed, test_ratio=test_ratio, source="synthetic")

    elif source == "custom_csv" and csv_file is not None:
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            numeric_df = df.select_dtypes(include=[np.number]).fillna(df.mean(numeric_only=True))
            if numeric_df.shape[1] >= 10:
                raw_X = numeric_df.iloc[:, :10].values.astype(float)
            else:
                padded = np.zeros((len(df), 10))
                padded[:, :numeric_df.shape[1]] = numeric_df.values
                raw_X = padded

            age = raw_X[:, 0]
            bp = raw_X[:, 2]
            chol = raw_X[:, 3]
            bmi = raw_X[:, 7]
            cardiac_score = 0.03 * (age - 50) + 0.02 * (bp - 130) + 0.01 * (chol - 200)
            y_cardiac = (1 / (1 + np.exp(-cardiac_score)) > 0.5).astype(float).reshape(-1, 1)
            y_diabetes = (1 / (1 + np.exp(-0.04 * (chol - 200) - 0.05 * (bmi - 25))) > 0.5).astype(float).reshape(-1, 1)
            life_expectancy = np.clip(82.0 - 0.08 * (bp - 120) - 0.03 * (chol - 200), 55.0, 95.0).reshape(-1, 1)
            vascular_age = np.clip(age + 0.15 * (bp - 120) + 0.05 * (chol - 200), 20.0, 95.0).reshape(-1, 1)
            n_samples = len(df)
        except Exception as e:
            logger.warning(f"Failed to load custom_csv dataset ({e}), falling back to synthetic source.")
            return get_medical_dataset(n_samples=n_samples, seed=seed, test_ratio=test_ratio, source="synthetic")

    elif source == "openml_heart":
        try:
            from sklearn.datasets import fetch_openml
            openml_data = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
            X_df = openml_data.data
            y_sr = openml_data.target

            age = X_df.iloc[:, 0].values.astype(float)
            sex = (X_df.iloc[:, 1].values == 1).astype(float) if X_df.iloc[:, 1].dtype == object else X_df.iloc[:, 1].values.astype(float)
            bp = X_df.iloc[:, 3].values.astype(float)
            chol = X_df.iloc[:, 4].values.astype(float)
            glucose = np.where(X_df.iloc[:, 5].values == 1, 140.0, 95.0)
            max_hr = X_df.iloc[:, 7].values.astype(float)
            st_dep = X_df.iloc[:, 9].values.astype(float) if X_df.shape[1] > 9 else np.zeros_like(age)
            bmi = 22.0 + 0.1 * (chol - 200) + rng.normal(0, 2, len(age))
            activity = np.clip(1.0 - (age - 30) / 60.0 + rng.normal(0, 0.1, len(age)), 0.0, 1.0)
            smoking = (rng.uniform(0, 1, len(age)) < 0.25).astype(float)

            raw_X = np.column_stack([age, sex, bp, chol, glucose, max_hr, st_dep, bmi, activity, smoking])
            y_cardiac = (y_sr.values.astype(int) == 2).astype(float).reshape(-1, 1) if y_sr.dtype != float else (y_sr.values > 1).astype(float).reshape(-1, 1)

            diabetes_score = 0.03 * (glucose - 100) + 0.05 * (bmi - 25)
            y_diabetes = (1 / (1 + np.exp(-diabetes_score)) > 0.5).astype(float).reshape(-1, 1)
            life_expectancy = np.clip(82.0 - 0.08 * (bp - 120) - 0.03 * (chol - 200) - 4.0 * y_cardiac.ravel(), 55.0, 92.0).reshape(-1, 1)
            vascular_age = np.clip(age + 0.18 * (bp - 120) + 0.08 * (chol - 200), 25.0, 95.0).reshape(-1, 1)
            n_samples = len(raw_X)
        except Exception as e:
            logger.warning(f"Failed to load openml_heart dataset ({e}), falling back to synthetic source.")
            return get_medical_dataset(n_samples=n_samples, seed=seed, test_ratio=test_ratio, source="synthetic")

    else:

        age = rng.uniform(1, 105, n_samples)
        sex = rng.binomial(1, 0.52, size=n_samples).astype(float)
        bp = rng.uniform(70, 200, n_samples)
        chol = rng.uniform(100, 400, n_samples)
        glucose = rng.uniform(60, 300, n_samples)
        max_hr = rng.uniform(60, 210, n_samples)
        st_dep = np.clip(rng.exponential(scale=0.8, size=n_samples), 0.0, 5.5)
        bmi = rng.uniform(14.0, 50.0, n_samples)
        activity = rng.uniform(0.0, 1.0, n_samples)
        smoking = rng.binomial(1, 0.28, size=n_samples).astype(float)

        cardiac_score = (
            0.045 * (age - 50) +
            0.35 * sex +
            0.03 * (bp - 130) +
            0.018 * (chol - 210) -
            0.025 * (max_hr - 140) +
            0.80 * st_dep +
            0.035 * (bmi - 25) -
            0.75 * activity +
            0.95 * smoking +
            0.0004 * (bp - 120) * (chol - 200)
        )
        prob_cardiac = 1 / (1 + np.exp(-cardiac_score))
        y_cardiac = (prob_cardiac > 0.5).astype(float).reshape(-1, 1)

        diabetes_score = (
            0.03 * (age - 45) +
            0.04 * (glucose - 100) +
            0.08 * (bmi - 25) +
            0.015 * (bp - 120) -
            0.90 * activity +
            0.45 * smoking +
            0.001 * (glucose - 100) * (bmi - 25)
        )
        prob_diabetes = 1 / (1 + np.exp(-diabetes_score))
        y_diabetes = (prob_diabetes > 0.5).astype(float).reshape(-1, 1)

        vascular_age = calculate_framingham_vascular_age(age, sex, bp, chol, smoking, y_diabetes.ravel()).reshape(-1, 1)
        life_expectancy = calculate_actuarial_life_expectancy(age, sex, vascular_age.ravel(), smoking, y_diabetes.ravel()).reshape(-1, 1)

        raw_X = np.column_stack([age, sex, bp, chol, glucose, max_hr, st_dep, bmi, activity, smoking])

    norm_X = normalize(raw_X, mins, maxs)

    n_test = max(1, int(n_samples * test_ratio))
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    feature_meta = [
        {"key": "age", "name_ru": "Возраст (лет)", "name_en": "Age (years)", "min": 30, "max": 80, "default": 52, "unit": "лет"},
        {"key": "sex", "name_ru": "Пол", "name_en": "Sex", "min": 0, "max": 1, "default": 1, "unit": "0=Ж, 1=М"},
        {"key": "bp", "name_ru": "Давление (мм рт.ст.)", "name_en": "Resting BP (mmHg)", "min": 95, "max": 180, "default": 128, "unit": "мм рт.ст."},
        {"key": "chol", "name_ru": "Холестерин (мг/дл)", "name_en": "Cholesterol (mg/dL)", "min": 150, "max": 350, "default": 215, "unit": "мг/дл"},
        {"key": "glucose", "name_ru": "Глюкоза (мг/дл)", "name_en": "Fasting Glucose (mg/dL)", "min": 70, "max": 220, "default": 98, "unit": "мг/дл"},
        {"key": "max_hr", "name_ru": "Макс. пульс (уд/мин)", "name_en": "Max Heart Rate (bpm)", "min": 90, "max": 200, "default": 145, "unit": "уд/мин"},
        {"key": "st_dep", "name_ru": "Депрессия ST (ЭКГ)", "name_en": "ST Depression (ECG)", "min": 0.0, "max": 4.5, "default": 0.8, "unit": "мм"},
        {"key": "bmi", "name_ru": "Индекс массы тела ИМТ", "name_en": "BMI (kg/m²)", "min": 18.5, "max": 40.0, "default": 24.5, "unit": "кг/м²"},
        {"key": "activity", "name_ru": "Физ. активность (0..1)", "name_en": "Physical Activity (0..1)", "min": 0.0, "max": 1.0, "default": 0.6, "unit": "индекс"},
        {"key": "smoking", "name_ru": "Курение (0=Нет, 1=Да)", "name_en": "Smoking (0=No, 1=Yes)", "min": 0, "max": 1, "default": 0, "unit": "флаг"},
    ]

    return {
        "source": source,
        "n_samples": n_samples,
        "raw_X": raw_X,
        "norm_X": norm_X,

        "y_cardiac": y_cardiac,
        "y_diabetes": y_diabetes,
        "y_life_expectancy": life_expectancy,
        "y_vascular_age": vascular_age,

        "X_train": norm_X[train_idx],
        "y_cardiac_train": y_cardiac[train_idx],
        "y_diabetes_train": y_diabetes[train_idx],
        "y_life_train": life_expectancy[train_idx],
        "y_vascular_train": vascular_age[train_idx],

        "X_test": norm_X[test_idx],
        "y_cardiac_test": y_cardiac[test_idx],
        "y_diabetes_test": y_diabetes[test_idx],
        "y_life_test": life_expectancy[test_idx],
        "y_vascular_test": vascular_age[test_idx],

        "mins": mins,
        "maxs": maxs,
        "feature_meta": feature_meta,
        "target_names": {
            "cardiac": {"ru": ["Низкий кардио-риск ✅", "Высокий кардио-риск ⚠️"], "en": ["Low Cardiac Risk ✅", "High Cardiac Risk ⚠️"]},
            "diabetes": {"ru": ["Низкий риск диабета ✅", "Высокий риск диабета II ⚠️"], "en": ["Low Diabetes Risk ✅", "High Diabetes Risk II ⚠️"]}
        }
    }
