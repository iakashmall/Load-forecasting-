"""
metrics.py
----------
Comprehensive evaluation metrics for Power_Load_kW forecasting.

All metrics are expressed in the original kW scale (or %) so they
are directly interpretable against the PowerLoad_Dataset values
(range ≈ 308 – 680 kW, mean ≈ 499 kW).

Metrics
-------
MAE        : Mean Absolute Error (kW)
RMSE       : Root Mean Squared Error (kW)
MAPE       : Mean Absolute Percentage Error (%)
sMAPE      : Symmetric MAPE (%)
MASE       : Mean Absolute Scaled Error (vs seasonal naïve baseline)
R²         : Coefficient of Determination
CVRMSE     : Coefficient of Variation of RMSE (%)
Peak Error : Max absolute error during peak demand hours
FSS        : Forecast Skill Score vs seasonal naïve persistence
"""

import numpy as np
import pandas as pd


# ── Individual metrics ────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %. Rows where |y_true| < 1e-6 are excluded."""
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    err = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return float(100 * np.mean(err))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in %."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    safe  = np.where(denom > 1e-8, np.abs(y_true - y_pred) / denom, 0.0)
    return float(100 * np.mean(safe))


def mase(
    y_true:          np.ndarray,
    y_pred:          np.ndarray,
    y_train:         np.ndarray,
    seasonal_period: int = 24,
) -> float:
    """
    MASE scaled by the in-sample seasonal naïve MAE (period = 24 h for hourly,
    or 7 days for daily data).
    """
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return float(mae(y_true, y_pred) / (scale + 1e-8))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CVRMSE in %."""
    return float(100 * rmse(y_true, y_pred) / (np.mean(np.abs(y_true)) + 1e-8))


def peak_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index:  pd.DatetimeIndex | None = None,
    peak_hours: tuple[int, int] = (17, 21),
) -> float:
    """
    Maximum absolute error during peak demand hours (17:00–21:00).
    If index is None, returns overall max |error|.
    """
    if index is not None and len(index) == len(y_true):
        mask = (index.hour >= peak_hours[0]) & (index.hour <= peak_hours[1])
        if mask.any():
            return float(np.max(np.abs(y_true[mask] - y_pred[mask])))
    return float(np.max(np.abs(y_true - y_pred)))


def forecast_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_period: int = 24,
) -> float:
    """
    FSS = 1 − RMSE_model / RMSE_naïve.
    Positive → model outperforms seasonal naïve; negative → worse.
    """
    n = len(y_true) - seasonal_period
    if n <= 0:
        return 0.0
    naive_rmse = rmse(y_true[seasonal_period:], y_true[:n])
    model_rmse = rmse(y_true[seasonal_period:], y_pred[seasonal_period:])
    return float(1 - model_rmse / (naive_rmse + 1e-8))


# ── Composite function ────────────────────────────────────────────────────────

def compute_metrics(
    y_true:          np.ndarray | pd.Series,
    y_pred:          np.ndarray | pd.Series,
    y_train:         np.ndarray | None = None,
    index:           pd.DatetimeIndex | None = None,
    seasonal_period: int = 24,
) -> dict:
    """
    Compute all standard forecasting metrics for Power_Load_kW.

    Parameters
    ----------
    y_true          : ground-truth kW values
    y_pred          : model predictions (kW)
    y_train         : training target — needed for MASE & FSS
    index           : DatetimeIndex aligned with y_true for peak error
    seasonal_period : 24 (hourly) or 7 (daily)

    Returns
    -------
    dict with keys: mae, rmse, mape, smape, r2, cvrmse,
                    peak_error, forecast_skill [, mase]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    result = {
        "mae":            mae(y_true, y_pred),
        "rmse":           rmse(y_true, y_pred),
        "mape":           mape(y_true, y_pred),
        "smape":          smape(y_true, y_pred),
        "r2":             r_squared(y_true, y_pred),
        "cvrmse":         cvrmse(y_true, y_pred),
        "peak_error":     peak_error(y_true, y_pred, index),
        "forecast_skill": forecast_skill_score(y_true, y_pred, seasonal_period),
    }

    if y_train is not None:
        result["mase"] = mase(
            y_true, y_pred, np.asarray(y_train, dtype=float), seasonal_period
        )

    return result


def metrics_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame.

    Parameters
    ----------
    results : {"ModelName": metrics_dict, …}
    """
    rows = [{"model": name, **m} for name, m in results.items()]
    df   = pd.DataFrame(rows).set_index("model")
    df   = df.round(4)
    return df


def print_metrics(m: dict, model_name: str = "") -> None:
    """Pretty-print a metrics dict."""
    header = f" {model_name} Evaluation " if model_name else " Metrics "
    print(f"\n{'═'*50}")
    print(f"{header:^50}")
    print("─" * 50)
    labels = {
        "mae":            "MAE (kW)",
        "rmse":           "RMSE (kW)",
        "mape":           "MAPE (%)",
        "smape":          "sMAPE (%)",
        "mase":           "MASE",
        "r2":             "R²",
        "cvrmse":         "CVRMSE (%)",
        "peak_error":     "Peak Error (kW)",
        "forecast_skill": "Forecast Skill Score",
    }
    for key, label in labels.items():
        if key in m:
            print(f"  {label:<24}: {m[key]:.4f}")
    print("═" * 50)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng    = np.random.default_rng(0)
    y_true = 499 + 50 * np.sin(np.linspace(0, 4 * np.pi, 720))   # ~499 kW mean
    y_pred = y_true + rng.normal(0, 15, 720)                       # ±15 kW noise
    y_tr   = y_true[:504]

    m = compute_metrics(y_true, y_pred, y_tr)
    print_metrics(m, "Synthetic Test")