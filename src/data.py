
# 1. Define required patient vital columns for model training and evaluation.
# 2. Clean and standardize column names to ensure consistency across datasets.
# 3. Convert data types, handle missing values, and prepare valid numeric inputs.
# 4. Load and preprocess training and evaluation CSVs into clean DataFrames.
# 5. Provide a quick summary of dataset size and columns for logging.
# 6. Load training data only** → `load_train_only()` loads just the training CSV when evaluation data isn’t needed.
# 7. Fit and save data scaler** → `fit_and_save_scaler()` trains a `StandardScaler` on training vitals and saves it as `scaler.joblib`.
# 8. Ensure consistent scaling** → Guarantees that the same normalization is applied in both training and inference.
# 9. Enable model persistence** → By saving the scaler, future predictions use the same scaling logic as training.


from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# columns in both train and eval
# A checklist of required column names.
REQUIRED_VITALS = ["hr", "sbp", "dbp", "spo2"]

# This function cleans and validates the dataframe's columns.
def _standardize_columns(df: pd.DataFrame, is_eval: bool) -> pd.DataFrame:
    
    # Dictionary for correcting specific column names (e.g., typos).
    rename_map = {
        "hr": "hr",
        "sbp": "sbp",
        "dbp": "dbp",
        "spo2": "spo2",
        "is_anomaly": "is_anomaly",
    }

    # Make all column names lowercase and apply the rename_map rules.
    df = df.rename(columns={c: rename_map.get(c.lower(), c.lower()) for c in df.columns})
    
    # Find which of the required columns are missing from the dataframe.
    missing = [c for c in REQUIRED_VITALS if c not in df.columns]

    # If the missing list is not empty, stop the program with an error.
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    # If in evaluation mode, the is_anomaly column must exist.
    if is_eval and "is_anomaly" not in df.columns:
        raise ValueError("Evaluation CSV must have an 'is_anomaly' column.")

    # If all checks pass, return the cleaned dataframe.
    return df

def _coerce_dtypes(df: pd.DataFrame, is_eval: bool) -> pd.DataFrame:


    """ This function ensures that:
    1. All patient vital sign columns (hr, sbp, dbp, spo2) are numeric.
    2. The "is_anomaly" column is properly converted to boolean (True/False) for evaluation data.
    3. Any invalid or missing values are removed from the dataset.
    """
    # Convert vitals (hr, sbp, dbp, spo2) to numbers, bad values → NaN
    for col in REQUIRED_VITALS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If evaluation data, fix 'is_anomaly' → make it True/False
    if is_eval and df["is_anomaly"].dtype != "bool":
        df["is_anomaly"] = (
            df["is_anomaly"]
            .astype(str).str.lower()
            .isin(["1", "true", "t", "yes"])
        )

    # Drop rows with missing/invalid vitals
    before = len(df)
    df = df.dropna(subset=REQUIRED_VITALS)
    after = len(df)

    # Show how many rows got removed (if any)
    if after < before:
        print(f"Dropped {before - after} bad row(s).")

    return df

def load_vitals(train_csv: str | Path, eval_csv: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and evaluation vitals CSVs with light validation and cleaning.
    Returns (train_df, eval_df).
    """
    train_csv = Path(train_csv)
    eval_csv  = Path(eval_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found at: {train_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV not found at: {eval_csv}")

    train_df = pd.read_csv(train_csv)
    eval_df  = pd.read_csv(eval_csv)

    train_df = _standardize_columns(train_df, is_eval=False)
    eval_df  = _standardize_columns(eval_df,  is_eval=True)

    train_df = _coerce_dtypes(train_df, is_eval=False)
    eval_df  = _coerce_dtypes(eval_df,  is_eval=True)

    return train_df, eval_df

def summarize(df: pd.DataFrame, name: str) -> str:
    """Return a tiny one-line summary for logging/prints."""
    return f"{name}: n={len(df)} rows | cols={list(df.columns)}"

def load_train_only(train_csv: str | Path) -> pd.DataFrame:
    """
    Load only the training CSV using the same standardization and dtype logic.
    """
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found at: {train_csv}")
    df = pd.read_csv(train_csv)
    df = _standardize_columns(df, is_eval=False)
    df = _coerce_dtypes(df, is_eval=False)
    return df

def fit_and_save_scaler(train_df: pd.DataFrame, model_dir: str | Path) -> Path:
    """
    Fit a StandardScaler on the training vitals and save it to model_dir/scaler.joblib.
    Returns the saved file path.
    """
    X = train_df[REQUIRED_VITALS].values  # shape: (n_rows, 4)
    scaler = StandardScaler().fit(X)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    return scaler_path
