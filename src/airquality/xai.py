from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from .models import predict_ann


def integrated_gradients_full(model, x_one: np.ndarray, steps: int = 64, device: str = "cpu") -> np.ndarray:
    """Return signed Integrated Gradients for every (lag, feature) coordinate."""
    model.eval()
    x = torch.tensor(x_one[None, ...], dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    grads = []
    for alpha in alphas:
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        value = model(point).sum()
        grad = torch.autograd.grad(value, point)[0]
        grads.append(grad.detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    return ((x - baseline) * avg_grad).squeeze(0).detach().cpu().numpy()


def integrated_gradients(model, x_one: np.ndarray, feature_names: list[str], steps: int = 64,
                         device: str = "cpu") -> pd.DataFrame:
    """Aggregate absolute/signed contributions across lags from a standardised-mean baseline."""
    attribution = integrated_gradients_full(model, x_one, steps=steps, device=device)
    signed = attribution.sum(axis=0)
    absolute = np.abs(attribution).sum(axis=0)
    return pd.DataFrame({"feature": feature_names, "signed_attribution": signed,
                         "absolute_attribution": absolute}).sort_values("absolute_attribution", ascending=False)


def grouped_permutation_importance(model, x: np.ndarray, y_raw: np.ndarray, target_scaler,
                                   feature_names: list[str], repeats: int = 5, seed: int = 42,
                                   device: str = "cpu") -> pd.DataFrame:
    """Permute a whole input channel across cases while preserving its 24-lag profile."""
    rng = np.random.default_rng(seed)
    baseline_scaled = predict_ann(model, x, device=device)
    baseline = target_scaler.inverse_transform(baseline_scaled.reshape(-1, 1)).ravel()
    from sklearn.metrics import mean_absolute_error
    base_mae = mean_absolute_error(y_raw, baseline)
    results = []
    for feature_idx, name in enumerate(feature_names):
        deltas = []
        for _ in range(repeats):
            perm = rng.permutation(len(x))
            xp = x.copy()
            xp[:, :, feature_idx] = xp[perm, :, feature_idx]
            pred_scaled = predict_ann(model, xp, device=device)
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            deltas.append(mean_absolute_error(y_raw, pred) - base_mae)
        results.append({"feature": name, "mae_increase": float(np.mean(deltas)),
                        "std": float(np.std(deltas, ddof=1)) if repeats > 1 else 0.0})
    return pd.DataFrame(results).sort_values("mae_increase", ascending=False)
