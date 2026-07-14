"""
arima.py
--------
ARIMA model for Power Load forecasting.

Data used : daily-aggregated Power_Load_kW from PowerLoad_Dataset.csv
            (hourly → daily mean via load_data_daily())

The raw dataset has irregular hourly timestamps; resampling to daily
gives a clean, regular 

series suited for ARIMA.

Supports
--------
- ADF stationarity test(required cuz these model only work for stationary datapoints
- Grid-search order selection (AIC / BIC)
- Rolling one-step-ahead forecast (walk-forward)
- Multi-step out-of-sample forecast with confidence intervals
- Model persistence (save / load)
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.tsa.arima.model
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_data_daily, train_test_split


TARGET     = "Power_Load_kW"
MODEL_PATH = Path(__file__).parent / "arima_model.pkl"


# ── Stationarity 

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test on the daily load series."""
    res = adfuller(series.dropna(), autolag="AIC")#ADF test on the series, autolag to select lag based on AIC
    info = {#Returned dictionary  
        "statistic":     round(res[0], 4),
        "p_value":       round(res[1], 6),
        "is_stationary": res[1] < alpha,
        "critical_1%":   res[4]["1%"],
        "critical_5%":   res[4]["5%"],
    } 
    status = "✓ Stationary" if info["is_stationary"] else "✗ Non-stationary (d ≥ 1)"
    print(f"[ARIMA] ADF p={info['p_value']:.6f}  {status}")
    return info


# ── Order selection 

def select_order(
    series: pd.Series,
    p_range: range = range(0, 4),
    d_range: range = range(0, 2),
    q_range: range = range(0, 4),
    criterion: str = "aic",
) -> tuple[int, int, int]:
    """Grid search over (p, d, q) returning the order with lowest AIC/BIC."""
    best_score, best_order = np.inf, (1, 1, 1)
    print(f"[ARIMA] Grid search (criterion={criterion.upper()}) …")

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    fit = statsmodels.tsa.arima.model.ARIMA(series, order=(p, d, q)).fit()# trains ARIMA model with given order
                    score = getattr(fit, criterion)
                    if score < best_score:
                        best_score, best_order = score, (p, d, q)
                except Exception:
                    continue

    print(f"[ARIMA] Best order {best_order}  {criterion.upper()}={best_score:.2f}")
    return best_order


# ── Model class 

class ARIMAForecaster:
    """
    ARIMA wrapper for daily Power_Load_kW.

    Parameters
    ----------
    order : (p, d, q). If None, auto-selected via grid search.
    """

    def __init__(self, order: tuple[int, int, int] | None = None):
        self.order   = order
        self.fitted_ = None
        self.series_ = None

    # ── fit 

    def fit(
        self,
        series: pd.Series,
        auto_order: bool = True,
    ) -> "ARIMAForecaster":
        """
        Fit on daily Power_Load_kW series.

        Parameters
        ----------
        series     : pd.Series with DatetimeIndex (daily)
        auto_order : run grid search if True and self.order is None
        """
        self.series_ = series.copy()

        if self.order is None:
            self.order = select_order(series) if auto_order else (2, 1, 2)

        print(f"\n[ARIMA] Fitting ARIMA{self.order}  n={len(series)} days …")
        self.fitted_ = statsmodels.tsa.arima.model.ARIMA(series, order=self.order).fit()# this estimates ARIMA model mathematically.
        print(self.fitted_.summary().tables[0].as_text())
        return self

    # ── multi-step forecast ───────────────────────────────────────────────────

    def predict(
        self,
        steps: int = 30,# number of days to forecast
        alpha: float = 0.05,#95% confidence intervals
    ) -> pd.DataFrame:
        """
        Out-of-sample forecast for `steps` days ahead.

        Returns
        -------
        pd.DataFrame  columns: forecast, lower_ci, upper_ci
        """
        fc = self.fitted_.get_forecast(steps=steps)#contains forecast values and confidence intervals
        mu = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)
        return pd.DataFrame({
            "forecast": mu,
            "lower_ci": ci.iloc[:, 0],
            "upper_ci": ci.iloc[:, 1],
        })

    # ── rolling walk-forward forecast 

    def rolling_forecast(
        self,
        test_series: pd.Series,# daily test Power_Load_kW
        retrain_every: int = 7,# re-fit every N=7 steps to absorb new observations
    ) -> np.ndarray:
        """
        Walk-forward one-step-ahead forecast over the test period.

        Parameters
        ----------
        test_series   : daily test Power_Load_kW
        retrain_every : re-fit every N = 7 steps to absorb new observations

        Returns
        -------
        np.ndarray  shape (len(test_series),)
        """
        history  = list(self.series_.values)# initial training series as history for walk-forward forecasting
        preds    = []# to store predictions, will be returned as numpy array at the end
        fitted   = self.fitted_# initial fitted model from training data, will be re-fitted every `retrain_every` steps

        print(f"[ARIMA] Rolling forecast — {len(test_series)} test days …")
        for i, obs in enumerate(test_series):# iterates over each observation in the test series, 
            #making a one-step-ahead forecast and then updating the history with the actual observed value.

            if i % retrain_every == 0:
                try:
                    fitted = statsmodels.tsa.arima.model.ARIMA(history, order=self.order).fit()
                except Exception:
                    pass

            fc = fitted.forecast(steps=1)
            preds.append(float(fc[0]))# appends the forecasted value for the next time step to the preds list.
            history.append(float(obs))# after forecasting the next value, 
            #it appends the actual observed value to the history list, 
            #which will be used for the next forecast. 
            #This way, the model is always updated with the latest information from the test series.

        return np.array(preds)

    # ── diagnostics 

    def plot_diagnostics(self) -> None:
        """Save a 4-panel residual diagnostics figure."""
        if self.fitted_ is None:
            return
        # imported here to avoid global dependency on matplotlib for non-diagnostics use cases
        import matplotlib.pyplot as plt # type: ignore
        fig = self.fitted_.plot_diagnostics(figsize=(14, 8))
        fig.suptitle("ARIMA Residual Diagnostics", y=1.01)
        out = Path(__file__).parent / "arima_diagnostics.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()# closes the figure to free up memory after saving and prevent memory leaking if this method is called multiple times in a loop or in a larger application.
        print(f"[ARIMA] Diagnostics → {out}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = MODEL_PATH) -> None:
        joblib.dump({"order": self.order, "fitted": self.fitted_,
                     "series": self.series_}, path)
        print(f"[ARIMA] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "ARIMAForecaster":
        state = joblib.load(path)
        inst  = cls(order=state["order"])
        inst.fitted_  = state["fitted"]
        inst.series_  = state["series"]
        print(f"[ARIMA] Loaded ← {path}")
        return inst

    def summary(self) -> str:
        return str(self.fitted_.summary()) if self.fitted_ else "Not fitted."


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    daily = load_data_daily()
    train, test = train_test_split(daily, test_months=3)

    train_s = train[TARGET]
    test_s  = test[TARGET]

    check_stationarity(train_s)

    print("\nTraining ARIMA model …")
    arima = ARIMAForecaster()
    arima.fit(train_s, auto_order=True)
    arima.plot_diagnostics()

    preds = arima.rolling_forecast(test_s)

    from evaluation.metrics import compute_metrics, print_metrics
    m = compute_metrics(test_s.values[:len(preds)], preds, train_s.values)
    print_metrics(m, "ARIMA")

