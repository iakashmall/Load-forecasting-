"""
stl_decomposition.py
--------------------
Improved STL-based forecasting model for daily Power_Load_kW.

MODEL PIPELINE
--------------
1. STL decomposition
      → Trend
      → Seasonal
      → Residual

2. Forecast each component separately
      Trend    → Holt-Winters Exponential Smoothing
      Seasonal → Holt-Winters Seasonal Forecast
      Residual → ARIMA(2,0,2)

3. Final Forecast
      Forecast = Trend + Seasonal + Residual

"""

# Imports

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Holt-Winters Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# STL decomposition
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_data_daily, train_test_split

# Constants

TARGET = "Power_Load_kW"

MODEL_PATH = Path(__file__).parent / "stl_model.pkl"

# STL Forecaster Class

class STLForecaster:
    """
    STL decomposition based forecasting model.

    PARAMETERS
    ----------
    period : int
        Seasonal cycle length.
        7 → weekly seasonality

    robust : bool
        Robust STL reduces effect of outliers/spikes.
    """

    def __init__(
        self,
        period: int = 7,
        robust: bool = True,
    ):

        self.period = period
        self.robust = robust

        # Trained sub-models
        self._trend_model_ = None
        self._seasonal_model_ = None
        self._resid_model_ = None

        # STL result
        self._stl_result_ = None

        # Original training series
        self.series_ = None

    # FIT
    
    def fit(self, daily_df: pd.DataFrame):

        """
        Train STL forecasting system.
        STEPS
        -----
        1. STL decomposition
        2. Train trend model
        3. Train seasonal model
        4. Train residual ARIMA
        """

        # Target series
        series = daily_df[TARGET].dropna().copy()

        self.series_ = series

        print(
            f"\n[STL] Decomposing series "
            f"(period={self.period}, n={len(series)}) ..."
        )

        # STL decomposition
        # seasonal=13 gives smoother seasonal extraction
        stl = STL(
            series,
            period=self.period,
            robust=self.robust,
            seasonal=13,
        )

        result = stl.fit()

        self._stl_result_ = result

        # Extract components
        trend = pd.Series(result.trend, index=series.index)

        seasonal = pd.Series(result.seasonal, index=series.index)

        residual = pd.Series(result.resid, index=series.index)

        # TREND MODEL

        print("[STL] Training trend model ...")
    
        """
        Holt-Winters:
        - models trend
        - adapts gradually
        - smoother than raw ARIMA trend
        """

        self._trend_model_ = ExponentialSmoothing(
            trend.dropna(),

            trend="add",

            damped_trend=True,

            initialization_method="estimated",
        ).fit(optimized=True)

        # SEASONAL MODEL

        print("[STL] Training seasonal model ...")

        """
        Instead of naive cycle repetition,
        we forecast seasonality using
        Exponential Smoothing.
        """

        self._seasonal_model_ = ExponentialSmoothing(
            seasonal.dropna(),

            seasonal="add",

            seasonal_periods=self.period,

            initialization_method="estimated",
        ).fit(optimized=True)

        # ──────────────────────────────────────────────────────────────
        # RESIDUAL MODEL
        # ──────────────────────────────────────────────────────────────

        print("[STL] Training residual ARIMA model ...")

        """
        Residual contains random leftover noise.

        ARIMA helps model remaining temporal structure.
        """

        self._resid_model_ = ARIMA(
            residual.dropna(),

            order=(2, 0, 2)
        ).fit()

        print("[STL] All models trained successfully.")

        return self


    # ──────────────────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, steps: int = 30):

        """
        Multi-step future forecasting.

        FINAL FORECAST:
        ----------------
        Trend Forecast
        + Seasonal Forecast
        + Residual Forecast
        """

        if self._trend_model_ is None:
            raise RuntimeError("Call fit() before predict().")

        # Trend forecast
        trend_fc = self._trend_model_.forecast(steps)

        # Seasonal forecast
        seasonal_fc = self._seasonal_model_.forecast(steps)

        # Residual forecast
        resid_fc = self._resid_model_.forecast(steps=steps)

        # Final combined forecast
        final_fc = trend_fc + seasonal_fc + resid_fc

        # Future timestamps
        last_ts = self.series_.index[-1]

        future_idx = pd.date_range(
            start=last_ts + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        return pd.Series(
            final_fc.values,
            index=future_idx,
            name="stl_forecast"
        )


    # ──────────────────────────────────────────────────────────────────────
    # ROLLING FORECAST
    # ──────────────────────────────────────────────────────────────────────

    def rolling_forecast(
        self,
        test_df: pd.DataFrame,
        retrain_every: int = 14,
    ) -> np.ndarray:

        """
        Walk-forward validation.

        HOW IT WORKS
        ------------
        1. Predict next day
        2. Observe actual value
        3. Add actual value to history
        4. Periodically retrain

        retrain_every=14 reduces overfitting noise.
        """

        history_df = pd.DataFrame({
            TARGET: self.series_
        })

        preds = []

        print(
            f"[STL] Rolling forecast — "
            f"{len(test_df)} test days ..."
        )

        for i, (ts, obs) in enumerate(test_df[TARGET].items()):

            # Periodic retraining
            if i % retrain_every == 0:

                self.fit(history_df)

            # Predict next day
            fc = self.predict(steps=1)

            preds.append(float(fc.iloc[0]))

            # Add actual observation
            new_row = pd.DataFrame(
                {TARGET: [obs]},
                index=[ts]
            )

            history_df = pd.concat(
                [history_df, new_row]
            )

        return np.array(preds)


    # ──────────────────────────────────────────────────────────────────────
    # COMPONENTS
    # ──────────────────────────────────────────────────────────────────────

    def get_components(self):

        """
        Return STL decomposition components.
        """

        if self._stl_result_ is None:
            raise RuntimeError("Call fit() first.")

        r = self._stl_result_

        return pd.DataFrame({

            "observed": r.observed,

            "trend": r.trend,

            "seasonal": r.seasonal,

            "residual": r.resid,

        }, index=self.series_.index)


    # ──────────────────────────────────────────────────────────────────────
    # PLOT COMPONENTS
    # ──────────────────────────────────────────────────────────────────────

    def plot_components(self):

        """
        Save STL decomposition plots.
        """

        import matplotlib.pyplot as plt

        comps = self.get_components()

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 12),
            sharex=True
        )

        labels = [
            "Observed",
            "Trend",
            "Seasonal",
            "Residual"
        ]

        for ax, col, label in zip(
            axes,
            comps.columns,
            labels
        ):

            ax.plot(comps[col], linewidth=1)

            ax.set_ylabel(label)

        axes[0].set_title(
            "STL Decomposition"
        )

        fig.tight_layout()

        out = Path(__file__).parent / "stl_components.png"

        fig.savefig(
            out,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close()

        print(f"[STL] Components plot → {out}")


    # ──────────────────────────────────────────────────────────────────────
    # SAVE MODEL
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path=MODEL_PATH):

        """
        Save trained STL system.
        """

        joblib.dump({

            "period": self.period,

            "robust": self.robust,

            "trend_model": self._trend_model_,

            "seasonal_model": self._seasonal_model_,

            "resid_model": self._resid_model_,

            "series": self.series_,

        }, path)

        print(f"[STL] Saved → {path}")


    # ──────────────────────────────────────────────────────────────────────
    # LOAD MODEL
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path=MODEL_PATH):

        """
        Load trained STL model.
        """

        s = joblib.load(path)

        inst = cls(

            period=s["period"],

            robust=s["robust"]
        )

        inst._trend_model_ = s["trend_model"]

        inst._seasonal_model_ = s["seasonal_model"]

        inst._resid_model_ = s["resid_model"]

        inst.series_ = s["series"]

        print(f"[STL] Loaded ← {path}")

        return inst


    # ──────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────

    def summary(self):

        return (
            f"\nSTLForecaster\n"
            f"------------------\n"
            f"Period : {self.period}\n"
            f"Robust : {self.robust}\n"
            f"Series Length : "
            f"{len(self.series_) if self.series_ is not None else 'N/A'}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load daily data
    daily = load_data_daily()

    # Train-test split
    train, test = train_test_split(
        daily,
        test_months=3
    )

    # Create model
    stl = STLForecaster(

        period=7,

        robust=True
    )

    # Train
    stl.fit(train)

    # Save decomposition plots
    stl.plot_components()

    # Rolling predictions
    preds = stl.rolling_forecast(

        test,

        retrain_every=14
    )

    # Evaluation
    from evaluation.metrics import (
        compute_metrics,
        print_metrics
    )

    metrics = compute_metrics(

        test[TARGET].values[:len(preds)],

        preds,

        train[TARGET].values,
    )

    print_metrics(metrics, "STL Improved")

    # Save model
    stl.save()