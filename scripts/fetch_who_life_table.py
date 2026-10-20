"""
Fetch official WHO life table for Uzbekistan (indicator LIFE_0000000035: ex - life expectancy at age x).
Primary source: WHO GHO OData API (https://ghoapi.azureedge.net/api/LIFE_0000000035).
Fallback source: Official WHO Global Health Estimates (GHE) dataset for Uzbekistan via HDX mirror.
Saves data to data/who_life_table_uzb.csv.
"""
import sys
import logging
import re
import io
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "who_life_table_uzb.csv"

PRIMARY_API_URL = "https://ghoapi.azureedge.net/api/LIFE_0000000035"
FALLBACK_GHE_URL = "https://data.humdata.org/dataset/d0128589-61de-4ded-88a7-451d1aa21148/resource/b26c4ad9-b797-4991-976b-1ddc3a99c9f5/download/global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_uz.csv"

STANDARD_AGE_GROUPS = [
    (0, 1), (1, 4), (5, 9), (10, 14), (15, 19), (20, 24),
    (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54),
    (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 120)
]

def parse_age_group(dim2: str):
    dim2 = str(dim2).strip().upper()
    if "85PLUS" in dim2 or "85+" in dim2:
        return 85, 120
    m = re.search(r"(\d+)-(\d+)", dim2)
    if m:
        return int(m.group(1)), int(m.group(2))
    m_single = re.search(r"(\d+)", dim2)
    if m_single:
        val = int(m_single.group(1))
        return val, val
    return None, None

def fetch_primary():
    logger.info("Attempting fetch from primary WHO GHO API...")
    resp = requests.get(
        PRIMARY_API_URL,
        params={"$filter": "SpatialDim eq 'UZB'"},
        headers={"User-Agent": "MedAI-Platform/1.0"},
        timeout=15
    )
    resp.raise_for_status()
    payload = resp.json()
    values = payload.get("value", [])
    if not values:
        raise ValueError("No records returned from primary API.")

    years = [v.get("TimeDim") for v in values if v.get("TimeDim") is not None]
    latest_year = max(years)
    valid_sex_map = {"SEX_MLE": "male", "SEX_FMLE": "female"}

    records = []
    for item in values:
        if item.get("TimeDim") != latest_year:
            continue
        dim1 = item.get("Dim1")
        if dim1 not in valid_sex_map:
            continue
        dim2 = item.get("Dim2")
        age_start, age_end = parse_age_group(dim2)
        if age_start is None:
            continue
        ex_val = item.get("NumericValue")
        if ex_val is None:
            continue
        records.append({
            "year": latest_year,
            "sex": valid_sex_map[dim1],
            "age_start": age_start,
            "age_end": age_end,
            "ex": round(float(ex_val), 4)
        })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Filtered dataframe is empty.")
    return df

def fetch_fallback():
    logger.info("Fetching from official WHO GHE Uzbekistan dataset mirror...")
    resp = requests.get(FALLBACK_GHE_URL, timeout=30)
    resp.raise_for_status()
    df_raw = pd.read_csv(io.StringIO(resp.text))

    # Find the latest year for which life tables (lx: LIFE_0000000031) exist
    df_lt = df_raw[df_raw['GHO (CODE)'] == 'LIFE_0000000031']
    available_years = sorted(df_lt['YEAR (DISPLAY)'].dropna().unique())
    if not available_years:
        raise ValueError("No life table records found in fallback dataset.")
    latest_year = int(max(available_years))
    logger.info(f"Latest life table year available: {latest_year}")

    records = []
    for sex_code, sex_name in [('SEX_MLE', 'male'), ('SEX_FMLE', 'female')]:
        lx_sub = df_raw[(df_raw['YEAR (DISPLAY)'] == latest_year) & 
                        (df_raw['GHO (CODE)'] == 'LIFE_0000000031') & 
                        (df_raw['DIMENSION (CODE)'] == sex_code)]
        tx_sub = df_raw[(df_raw['YEAR (DISPLAY)'] == latest_year) & 
                        (df_raw['GHO (CODE)'] == 'LIFE_0000000034') & 
                        (df_raw['DIMENSION (CODE)'] == sex_code)]

        lx_vals = sorted(lx_sub['Numeric'].tolist(), reverse=True)
        tx_vals = sorted(tx_sub['Numeric'].tolist(), reverse=True)

        if len(lx_vals) != len(STANDARD_AGE_GROUPS) or len(tx_vals) != len(STANDARD_AGE_GROUPS):
            raise ValueError(f"Unexpected number of age rows for {sex_name}: lx={len(lx_vals)}, tx={len(tx_vals)}")

        for (a_start, a_end), lx, tx in zip(STANDARD_AGE_GROUPS, lx_vals, tx_vals):
            ex_val = tx / lx
            records.append({
                "year": latest_year,
                "sex": sex_name,
                "age_start": a_start,
                "age_end": a_end,
                "ex": round(float(ex_val), 4)
            })

    return pd.DataFrame(records)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = None
    try:
        df = fetch_primary()
        logger.info("Successfully fetched data from primary WHO GHO API.")
    except Exception as e:
        logger.warning(f"Primary WHO GHO API unavailable ({e}). Switching to official WHO GHE fallback...")
        try:
            df = fetch_fallback()
            logger.info("Successfully fetched and computed life table from official WHO GHE fallback.")
        except Exception as e_fb:
            logger.error(f"Fallback also failed: {e_fb}")
            return False

    df = df.sort_values(by=["sex", "age_start"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(df.to_string())
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
