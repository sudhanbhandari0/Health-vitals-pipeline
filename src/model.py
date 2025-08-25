import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset




REQUIRED_VITALS = ["hr", "sbp", "dbp", "spo2"]  # 4 inputs

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = len(REQUIRED_VITALS), hidden_dim: int = 8, bottleneck: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    hidden_dim: int = 8
    bottleneck: int = 2

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_scaled_train_matrix(train_csv: str | Path, scaler_path: str | Path) -> np.ndarray:
    """
    Load training CSV via data.load_train_only and return scaled vitals (N,4) float32.
    We import inside the function to avoid import-time cycles between modules.
    """

    from data import load_train_only

    train_df = load_train_only(train_csv)
    X = train_df[REQUIRED_VITALS].to_numpy(dtype=np.float32)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    return X_scaled

def make_loader(X_scaled: np.ndarray, batch_size: int) -> DataLoader:
    x_t = torch.tensor(X_scaled, dtype=torch.float32)
    ds = TensorDataset(x_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def train_one_epoch(model:nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, loader:DataLoader) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for (batch_x,) in loader:
        optimizer.zero_grad()
        recon = model(batch_x)
        loss = criterion(recon,batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        total_n += batch_x.size(0)
    return total_loss/max(total_n, 1)

def reconstruction_errors(model: nn.Module, X_scaled: np.ndarray) -> np.ndarray:
    """Compute per-row MSE reconstruction errors for X_scaled using the model."""
    model.eval()
    with torch.no_grad():
        X_t     = torch.tensor(X_scaled, dtype=torch.float32)
        recon_t = model(X_t)
    recon_np = recon_t.detach().cpu().numpy().astype(np.float32)
    X_np     = X_scaled.astype(np.float32)
    return np.mean((recon_np - X_np) ** 2, axis=1)

def percentile_threshold(errs: np.ndarray, p: float = 99.5) -> float:
    return float(np.percentile(errs,p))

def save_model(model: torch.nn.Module, model_dir: str | Path, filename: str = "autoencoder.pt"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok = True)
    model_path = model_dir / filename

    torch.save(model.state_dict(), model_path)
    return model_path

def save_threshold(threshold:float, model_dir:str | Path, filename: str = "threshold.json"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok = True)
    th_path = model_dir / filename
    with open(th_path, "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    return th_path

def save_config(cfg: TrainingConfig, out_dir: str | Path, filename="train_config.json"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / filename).write_text(json.dumps({
        "hidden_dim": cfg.hidden_dim,
        "bottleneck": cfg.bottleneck,
        "input_dim": len(REQUIRED_VITALS),
    }))


def train_autoencoder(
    train_csv: str|Path,
    scaler_path: str|Path,
    model_dir: str|Path,
    cfg: TrainingConfig
):
    _set_seed(cfg.seed)

    X_scaled = load_scaled_train_matrix(train_csv, scaler_path)
    loader = make_loader(X_scaled, cfg.batch_size)

    model = Autoencoder(input_dim=len(REQUIRED_VITALS), hidden_dim=cfg.hidden_dim, bottleneck=cfg.bottleneck)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    avg_loss = 0.0
    for _ in range(cfg.epochs):
        avg_loss = train_one_epoch(model, optimizer, criterion, loader)

    errs = reconstruction_errors(model, X_scaled)
    th = percentile_threshold(errs, 99.5)

    models_dir = Path(model_dir)
    model_path = save_model(model, models_dir)
    th_path = save_threshold(th, models_dir)

    return {
        "model_path": str(model_path),
        "threshold_path": str(th_path),
        "threshold": float(th),
        "train_loss_last_epoch": float(avg_loss),
        "train_err_min": float(errs.min()),
        "train_err_med": float(np.median(errs)),
        "train_err_max": float(errs.max()),
    }
