"""
sarima.py
---------
SARIMA / SARIMAX model for daily Power_Load_kW forecasting.

Uses daily-aggregated data from PowerLoad_Dataset.csv.

Key choices
-----------
- Seasonal period s = 7  (weekly seasonality on daily data)
- Exogenous regressors: Temperature_C, HolidayFlag (daily mean/max)
  available directly in the dataset → passed to SARIMAX when use_exog=True
- Rolling walk-forward re-fitting every 7 days
"""

import sys# interacting with python runtime and modifying  import paths
import warnings# used to hide warning messages from statmodels, which can be excessive during model fitting
from pathlib import Path# moern way to handle file paths and directories

import joblib# used for saving trained models
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX#imports SARIMAX forecasting model from statsmodels library

warnings.filterwarnings("ignore")#hides unnecessary warning message

#importing files form parent folder 
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_loader import load_data_daily, train_test_split #importing my custom functions

TARGET     = "Power_Load_kW"
MODEL_PATH = Path(__file__).parent / "sarima_model.pkl"# where trained model will be saved

# Exogenous columns available in the daily dataset
# or say, external variables used for predicting
EXOG_COLS  = ["Temperature_C", "HolidayFlag", "Weekly_PreDispatch_Projection"]


# ── Order selection ───────────────────────────────────────────────────────────
#this function automatically finds the best SARIMA parameters
#p= auto regressive order, d= differencing order, q= moving average order
#P= seasonal auto regressive order, D= seasonal differencing order, 
#Q= seasonal moving average order, s= seasonal period
def select_sarima_order(
    series: pd.Series,
    s: int = 7,
    criterion: str = "aic",
    max_p: int = 2,#AR order : how many past values to use for predicting the next value
    max_q: int = 2,#MA order : how many past forecast errors to use for predicting the next value
    max_P: int = 1,#Seasonal AR order : how many past seasonal values to use for predicting the next value
    max_Q: int = 1,#Seasonal MA order : how many past seasonal forecast errors to use for predicting the next value
) -> tuple:
    """Constrained grid search for SARIMA(p,d,q)(P,D,Q,s)."""
    best_score, best_order = np.inf, (1, 1, 1, 1, 1, 1, s)#initializes best_score to infinity and best_order to a default SARIMA configuration. The function will iterate through the specified ranges of p, q, P, and Q to find the combination that yields the lowest AIC or BIC score.
    print(f"[SARIMA] Grid search (s={s}, criterion={criterion.upper()}) …")

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    try:
                        fit = SARIMAX(
                            series,
                            order=(p, 1, q),
                            seasonal_order=(P, 1, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                        score = getattr(fit, criterion) # model quality vs complexity (lower is better)     
                        if score < best_score:
                            best_score = score
                            best_order = (p, 1, q, P, 1, Q, s)
                    except Exception:
                        continue

    p, d, q, P, D, Q, s_ = best_order
    print(f"[SARIMA] Best → SARIMA({p},{d},{q})({P},{D},{Q},{s_})  "
          f"{criterion.upper()}={best_score:.2f}")
    return best_order


# ── Model class ───────────────────────────────────────────────────────────────

class SARIMAForecaster:
    """
    SARIMA(X) wrapper for daily Power_Load_kW.

    Parameters
    ----------
    order          : (p, d, q)
    seasonal_order : (P, D, Q, s)  — default weekly seasonality (s=7)
    use_exog       : include Temperature_C, HolidayFlag,
                     Weekly_PreDispatch_Projection as regressors
    """

    def __init__(
        self,
        order:          tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 7),
        use_exog:       bool  = True,# without this this model will be only SARIMA 
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.use_exog       = use_exog
        self.fitted_        = None
        self.series_        = None
        self.exog_train_    = None

    # ── exog builder ──────────────────────────────────────────────────────────

    def _get_exog(self, df: pd.DataFrame) -> np.ndarray | None:# builds the exogenous regressor matrix based on the specified columns in EXOG_COLS. If use_exog is False or if none of the specified columns are present in the DataFrame, it returns None. Otherwise, it returns a NumPy array containing the values of the exogenous regressors.
        cols = [c for c in EXOG_COLS if c in df.columns]
        if not self.use_exog or not cols:
            return None
        return df[cols].values.astype(float)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        daily_df: pd.DataFrame,
        auto_order: bool = False,# runs grid search automatically 
    ) -> "SARIMAForecaster":
        """
        Fit SARIMA(X) on daily_df (output of load_data_daily).

        Parameters
        ----------
        daily_df   : daily DataFrame with Power_Load_kW + exog columns
        auto_order : grid-search optimal (p,d,q)(P,D,Q,s) before fitting
        """
        series = daily_df[TARGET].copy()
        exog   = self._get_exog(daily_df)

        self.series_      = series
        self.exog_train_  = exog

        if auto_order:
            p, d, q, P, D, Q, s = select_sarima_order(series, s=self.seasonal_order[3])
            self.order          = (p, d, q)
            self.seasonal_order = (P, D, Q, s)

        print(
            f"\n[SARIMA] Fitting SARIMA{self.order}×{self.seasonal_order}  "
            f"n={len(series)} days  exog={'yes' if exog is not None else 'no'} …"
        )
        self.fitted_ = SARIMAX(
            series, exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)
        print(self.fitted_.summary().tables[0].as_text())
        return self

    # ── forecast ──────────────────────────────────────────────────────────────

    def predict( # generates future predictions and confidence intervals
        self,
        steps: int = 30,
        exog_future: np.ndarray | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Multi-step out-of-sample forecast.

        Parameters
        ----------
        exog_future : shape (steps, n_exog).  Required if use_exog=True.

        Returns
        -------
        pd.DataFrame  columns: forecast, lower_ci, upper_ci
        """
        fc = self.fitted_.get_forecast(steps=steps, exog=exog_future)
        mu = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)
        return pd.DataFrame({
            "forecast": mu,
            "lower_ci": ci.iloc[:, 0],
            "upper_ci": ci.iloc[:, 1],
        })

    # ── rolling walk-forward ──────────────────────────────────────────────────

    def rolling_forecast(
        self,
        test_df: pd.DataFrame,
        retrain_every: int = 7,
    ) -> np.ndarray:
        """
        Walk-forward one-step-ahead forecast over test_df.

        Parameters
        ----------
        test_df       : daily test DataFrame (same columns as train)
        retrain_every : re-fit model every N steps
        """
        history_y    = list(self.series_.values)# stores past  observed values
        history_exog = (list(self.exog_train_) if self.exog_train_ is not None # stores past exogenous variables
                        else None)
        preds  = []
        fitted = self.fitted_
        test_s = test_df[TARGET]
        test_e = self._get_exog(test_df)

        print(f"[SARIMA] Rolling forecast — {len(test_s)} test days …")
        for i, obs in enumerate(test_s):
            if i % retrain_every == 0 and i > 0:
                try:
                    he = np.array(history_exog) if history_exog else None
                    fitted = SARIMAX(
                        history_y, exog=he,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, start_params=fitted.params)
                except Exception:
                    pass

            ef = test_e[i : i + 1] if test_e is not None else None
            # fc = fitted.get_forecast(steps=1, exog=ef)
            fc = fitted.get_forecast(steps=1, exog=ef)

            preds.append(float(np.asarray(fc.predicted_mean)[0]))
            #preds.append(float(fc.predicted_mean.iloc[0]))

            history_y.append(float(obs))
            if history_exog is not None and test_e is not None:
                history_exog.append(test_e[i])

        return np.array(preds)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def plot_diagnostics(self) -> None:
        if self.fitted_ is None:
            return
        import matplotlib.pyplot as plt # type: ignore
        fig = self.fitted_.plot_diagnostics(figsize=(14, 8))
        fig.suptitle("SARIMA Residual Diagnostics", y=1.01)
        out = Path(__file__).parent / "sarima_diagnostics.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[SARIMA] Diagnostics → {out}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = MODEL_PATH) -> None:
        joblib.dump({
            "order": self.order, "seasonal_order": self.seasonal_order,
            "use_exog": self.use_exog, "fitted": self.fitted_,
            "series": self.series_, "exog_train": self.exog_train_,
        }, path)
        print(f"[SARIMA] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "SARIMAForecaster":
        s = joblib.load(path)
        inst = cls(order=s["order"], seasonal_order=s["seasonal_order"],
                   use_exog=s["use_exog"])
        inst.fitted_       = s["fitted"]
        inst.series_       = s["series"]
        inst.exog_train_   = s["exog_train"]
        print(f"[SARIMA] Loaded ← {path}")
        return inst

    def summary(self) -> str:
        return str(self.fitted_.summary()) if self.fitted_ else "Not fitted."


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    daily = load_data_daily()
    train_daily, test_daily = train_test_split(daily, test_months=3)

    sarima = SARIMAForecaster(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), use_exog=True
    )
    sarima.fit(train_daily, auto_order=False)
    sarima.plot_diagnostics()

    preds = sarima.rolling_forecast(test_daily)

    from evaluation.metrics import compute_metrics, print_metrics
    m = compute_metrics(
        test_daily[TARGET].values[:len(preds)], preds,
        train_daily[TARGET].values,
    )
    print_metrics(m, "SARIMA")
    sarima.save()