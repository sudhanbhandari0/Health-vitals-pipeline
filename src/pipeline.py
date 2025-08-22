from prefect import flow, task, get_run_logger
from pathlib import Path
from data import load_vitals, summarize, load_train_only, fit_and_save_scaler


@task(name="ingest_data")
def ingest_data(train_csv: str, eval_csv: str):
    """
    Load and validate training + evaluation CSVs.
    """

    logger = get_run_logger()

    # Use our helper from data.py to load & clean both datasets
    train_df, eval_df = load_vitals(train_csv, eval_csv)

    # Log dataset summaries for quick confirmation
    logger.info(summarize(train_df, "TRAIN"))
    logger.info(summarize(eval_df, "EVAL"))

    # Return row counts (we'll use this later in model training step)
    return {
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
    }


@task(name="fit_scaler")
def fit_scaler_task(train_csv: str, model_dir: str):
    """
    Fit StandardScaler on training vitals and save it.
    """
    logger = get_run_logger()
    train_df = load_train_only(train_csv)
    scaler_path = fit_and_save_scaler(train_df, model_dir)
    logger.info(f"Saved scaler to: {scaler_path}")
    return str(scaler_path)

@flow(name="vitals_flow")
def vitals_flow():
    project_root = Path(__file__).resolve().parents[1]
    train_csv = project_root / "data" / "raw_vitals_normal.csv"
    eval_csv  = project_root / "data" / "new_day_with_anomalies_groundtruth.csv"
    models_dir = project_root / "models"

    info = ingest_data(str(train_csv), str(eval_csv))
    scaler_path = fit_scaler_task(str(train_csv), str(models_dir))

    return {"info": info, "scaler_path": scaler_path}

if __name__ == "__main__":
    vitals_flow()