from __future__ import annotations
import copy
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error


class TemporalANN(nn.Module):
    """LSTM-based ANN for multivariate time-series regression."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=recurrent_dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(self.norm(sequence[:, -1, :])).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def predict_ann(model: nn.Module, x: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0)


def fit_ann(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            config: dict, seed: int = 42, max_epochs: int = 35, patience: int = 7,
            device: str | None = None) -> dict:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lookback = int(config.get("lookback", x_train.shape[1]))
    xt = x_train[:, -lookback:, :]
    xv = x_val[:, -lookback:, :]
    model = TemporalANN(
        input_size=xt.shape[-1], hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)), dropout=float(config.get("dropout", 0.2))
    ).to(device)
    optimizer_name = str(config.get("optimizer", "AdamW")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 1e-4)))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 1e-4)))
    loss_fn = nn.HuberLoss(delta=1.0)
    batch_size = int(config.get("batch_size", 128))
    loader = DataLoader(TensorDataset(torch.tensor(xt), torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=False)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict_ann(model, xv, device=device)
        val_mae_scaled = float(mean_absolute_error(y_val, val_pred))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_huber_loss": train_loss, "val_mae_scaled": val_mae_scaled})
        if np.isfinite(val_mae_scaled) and val_mae_scaled < best_val:
            best_val = val_mae_scaled
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience or not np.isfinite(val_mae_scaled):
            break
    duration = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("ANN training diverged or did not produce a finite validation score.")
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "state_dict": best_state, "best_epoch": best_epoch,
            "best_val_mae_scaled": best_val, "history": history, "training_seconds": duration,
            "device": device, "lookback": lookback}
