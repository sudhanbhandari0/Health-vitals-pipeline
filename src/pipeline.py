from prefect import flow, task, get_run_logger
from pathlib import Path

from data import load_vitals, summarize, load_train_only, fit_and_save_scaler
from model import train_autoencoder, TrainingConfig
from eval import score_and_save


@task(name="ingest_data")
def ingest_data(train_csv: str, eval_csv: str):
    """Load and validate training + evaluation CSVs."""
    logger = get_run_logger()
    train_df, eval_df = load_vitals(train_csv, eval_csv)
    logger.info(summarize(train_df, "TRAIN"))
    logger.info(summarize(eval_df, "EVAL"))
    return {"train_rows": len(train_df), "eval_rows": len(eval_df)}


@task(name="fit_scaler")
def fit_scaler_task(train_csv: str, model_dir: str):
    """Fit StandardScaler on training vitals and save it."""
    logger = get_run_logger()
    train_df = load_train_only(train_csv)
    scaler_path = fit_and_save_scaler(train_df, model_dir)
    logger.info(f"Saved scaler to: {scaler_path}")
    return str(scaler_path)


@task(name="train_model")
def train_model_task(train_csv: str, scaler_path: str, models_dir: str):
    """
    Train the autoencoder on scaled training vitals, compute threshold,
    and save artifacts to models_dir.
    """
    logger = get_run_logger()
    result = train_autoencoder(
        train_csv=train_csv,
        scaler_path=scaler_path,
        model_dir=models_dir,
        cfg=TrainingConfig(epochs=5)  # adjust if needed
    )
    logger.info(f"Saved model: {result['model_path']}")
    logger.info(f"Threshold: {result['threshold']:.6f}")
    return result


@task(name="batch_score")
def batch_score_task(train_csv: str, eval_csv: str, models_dir: str, out_csv: str):
    """Score the eval CSV and write data/scored/scored.csv."""
    logger = get_run_logger()
    path = score_and_save(train_csv, eval_csv, models_dir, out_csv)
    logger.info(f"Wrote scored CSV: {path}")
    return str(path)


@flow(name="vitals_flow")
def vitals_flow():
    project_root = Path(__file__).resolve().parents[1]
    train_csv  = project_root / "data" / "training_data_normal_only.csv"
    eval_csv   = project_root / "data" / "evaluation_data_human_vitals.csv"
    models_dir = project_root / "models"
    scored_csv = project_root / "data" / "scored" / "scored.csv"

    info = ingest_data(str(train_csv), str(eval_csv))
    scaler_path = fit_scaler_task(str(train_csv), str(models_dir))
    train_result = train_model_task(str(train_csv), scaler_path, str(models_dir))
    scored_path = batch_score_task(str(train_csv), str(eval_csv), str(models_dir), str(scored_csv))

    return {
        "info": info,
        "scaler_path": scaler_path,
        "train_result": train_result,
        "scored_csv": scored_path,
    }


if __name__ == "__main__":
    vitals_flow()
