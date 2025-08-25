# src/eval.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any, Dict
import json

import joblib
import torch
import numpy as np

from model import Autoencoder, REQUIRED_VITALS
from data import load_vitals


def load_artifacts(models_dir: str | Path) -> Tuple[Any, torch.nn.Module, float]:
    """
    Load the three saved tools needed for scoring:
      - scaler.joblib  (StandardScaler fitted on training vitals)
      - autoencoder.pt (trained model weights)
      - threshold.json (decision cutoff for recon_error)
    Returns (scaler, model, threshold).
    """
    models_dir = Path(models_dir)

    # 1) Scaler (stores mean/std from training)
    scaler = joblib.load(models_dir / "scaler.joblib")

    # 2) Model (same architecture as training) + weights
    
    cfg_path = models_dir / "train_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        model = Autoencoder(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], bottleneck=cfg["bottleneck"])
    else:
        model = Autoencoder(input_dim=len(REQUIRED_VITALS), hidden_dim=8, bottleneck=2)


    state_dict = torch.load(models_dir / "autoencoder.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Threshold (float)
    with open(models_dir / "threshold.json", "r") as f:
        threshold = float(json.load(f)["threshold"])

    return scaler, model, threshold


def scale_eval_matrix(eval_df, scaler) -> np.ndarray:
    """Apply the saved StandardScaler to eval vitals; returns (N,4) float32 matrix."""
    X = eval_df[REQUIRED_VITALS].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled


def reconstruction_errors(model: torch.nn.Module, X_scaled: np.ndarray) -> np.ndarray:
    """Compute per-row MSE reconstruction error on scaled inputs."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        recon_t = model(x_t)
    recon = recon_t.detach().cpu().numpy().astype(np.float32)
    return np.mean((recon - X_scaled) ** 2, axis=1)


def score_and_save(
    train_csv: str | Path,
    eval_csv: str | Path,
    models_dir: str | Path,
    out_csv: str | Path,
) -> Path:
    """
    End-to-end batch scoring:
      1) Clean/validate eval CSV using data.load_vitals (ensures columns & dtypes).
      2) Load artifacts (scaler, model, threshold).
      3) Scale eval vitals and compute per-row reconstruction errors.
      4) Flag anomalies: predicted_is_anomaly = (recon_error > threshold).
      5) Save scored CSV with columns:
         hr, sbp, dbp, spo2, recon_error, predicted_is_anomaly, [is_anomaly if present]
    Returns the path to the written CSV.
    """
    models_dir = Path(models_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Cleaned/validated dfs (train_df unused; ensures schema)
    _, eval_df = load_vitals(train_csv, eval_csv)

    # Artifacts
    scaler, model, threshold = load_artifacts(models_dir)

    # Errors + flags
    X_scaled = scale_eval_matrix(eval_df, scaler)
    errs = reconstruction_errors(model, X_scaled)
    preds = (errs > threshold)

    # Assemble scored DF
    scored = eval_df.copy()
    scored["recon_error"] = errs
    scored["predicted_is_anomaly"] = preds.astype(bool)

    cols = REQUIRED_VITALS + ["recon_error", "predicted_is_anomaly"]
    if "is_anomaly" in scored.columns:
        cols.append("is_anomaly")
    scored = scored[cols]

    # Write
    scored.to_csv(out_csv, index=False)
    return out_csv


def compute_metrics(scored_df) -> Dict[str, float]:
    """
    Compute Precision/Recall/F1 when ground truth 'is_anomaly' is present.
    Returns {} if ground truth is missing.
    """
    if "is_anomaly" not in scored_df.columns:
        return {}

    y_true = scored_df["is_anomaly"].astype(bool).to_numpy()
    y_pred = scored_df["predicted_is_anomaly"].astype(bool).to_numpy()

    tp = int(((y_true == True) & (y_pred == True)).sum())
    fp = int(((y_true == False) & (y_pred == True)).sum())
    fn = int(((y_true == True) & (y_pred == False)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# if __name__ == "__main__":
#     project = Path(__file__).resolve().parents[1]
#     out_csv = project / "data" / "scored" / "scored.csv"
#     print(score_and_save(
#         project / "data" / "raw_vitals_normal.csv",
#         project / "data" / "new_day_with_anomalies_groundtruth.csv",
#         project / "models",
#         out_csv,
#     ))

# if __name__ == "__main__":
#     from pathlib import Path
#     project = Path(__file__).resolve().parents[1]
#     train_csv =project/"data"/"raw_vitals_normal.csv"
#     eval_csv = project/"data"/"new_day_with_anomalies_groundtruth.csv"
#     models_dir = project/"models"

#     _, eval_df = load_vitals(train_csv, eval_csv)

#     scaler, model, threshold = load_artifacts(models_dir)

#     X_scaled = scale_eval_matrix(eval_df, scaler)
#     errs = reconstruction_errors(model, X_scaled)

#     print("eval rows:", len(eval_df))
#     print("scaled shape:", X_scaled.shape)
#     print("errors: min/med/max =", float(errs.min()), float(np.median(errs)), float(errs.max()))
#     print("threshold:", threshold)
#     print("predicted anomaly rate:", float((errs > threshold).mean()))














