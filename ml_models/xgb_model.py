"""
xgboost.py
----------
XGBoost gradient-boosted tree model for hourly Power_Load_kW forecasting.

Input data
----------
Full feature matrix built by preprocessing.build_feature_matrix() on the
hourly-resampled PowerLoad_Dataset.csv.  All 8 original dataset columns
plus ~80 engineered features (lags, rolling, Fourier, domain).

Forecast strategy
-----------------
Direct multi-step: one XGBRegressor per horizon step.
Single-step mode also available.

Features
--------
- Optional Optuna hyperparameter tuning
- Time-series cross-validation (expanding window)
- SHAP feature importance
- Prediction with mean + optional quantile intervals
- Model persistence (joblib)
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_data, train_test_split
from data.preprocessing import build_feature_matrix

TARGET     = "Power_Load_kW"
MODEL_PATH = Path(__file__).parent / "xgboost_model.pkl"

DEFAULT_PARAMS = {
    "n_estimators":          900,#XGBoost builds 900 trees sequentially
    "max_depth":             6,#controls model complexity higher -> may lead to overfitting)
    "learning_rate":         0.03,
    "subsample":             0.8,#Uses only 80% of the rows per tree(reduces overfitting)
    "colsample_bytree":      0.8,#Uses only 80% of the features per tree(Improves generalization)
    "min_child_weight":      5,
    "reg_alpha":             0.1,#This helps preventing overfitting by penalizing overly complex trees
    "reg_lambda":            1.0,#This helps preventing overfitting by penalizing overly complex trees
    "objective":             "reg:squarederror",
    "tree_method":           "hist",
    "random_state":          42,
    "n_jobs":                -1,
}


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_splits: int = 3,
) -> dict:
    """Bayesian search via Optuna. Falls back to defaults if not installed."""
    try:
        import optuna # type: ignore
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[XGB] Optuna not installed — using default params.")
        return DEFAULT_PARAMS.copy()

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial):
        p = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "objective": "reg:squarederror", "tree_method": "hist",
            "random_state": 42, "n_jobs": -1,
        }
        rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = xgb.XGBRegressor(**p)
            m.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[val_idx], y_train[val_idx])],
                  verbose=False)
            rmses.append(np.sqrt(np.mean(
                (m.predict(X_train[val_idx]) - y_train[val_idx]) ** 2)))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"objective": "reg:squarederror", "tree_method": "hist",
                 "random_state": 42, "n_jobs": -1})
    print(f"[XGB] Best params: {best}")
    return best


# ── Model class ───────────────────────────────────────────────────────────────

class XGBoostForecaster:
    """
    XGBoost load forecaster.

    Parameters
    ----------
    params  : XGBRegressor hyperparameters. None → defaults.
    horizon : forecast horizon (hours). One model per step.
    """

    def __init__(
        self,
        params:  dict | None = None,
        horizon: int         = 24,
    ):
        self.params         = params or DEFAULT_PARAMS.copy()
        self.horizon        = horizon
        self.models_:        list[xgb.XGBRegressor] = []
        self.feature_names_: list[str] = []

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series   | np.ndarray,
        X_val:   pd.DataFrame | np.ndarray | None = None,
        y_val:   pd.Series   | np.ndarray | None = None,
        tune:    bool = False,
    ) -> "XGBoostForecaster":
        """Train one XGBRegressor per forecast step."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        if tune:
            self.params = tune_hyperparameters(X_train, y_train)

        print(f"\n[XGB] Training XGBoost  horizon={self.horizon}h  "
              f"n_train={len(X_train)}  n_features={X_train.shape[1]} …")
        self.models_ = []

        for step in range(self.horizon):
           # y_s = y_train[step:] if step == 0 else np.roll(y_train, -step)[:-step]
            #X_s = X_train[:len(y_s)]
            if step == 0:
                X_s = X_train
                y_s = y_train
            else:
                X_s = X_train[:-step]
                
                y_s = y_train[step:]

            params = {k: v for k, v in self.params.items()
                      if k != "early_stopping_rounds"}
            m = xgb.XGBRegressor(**params)
            ev = [(X_val, y_val[step:][:len(X_val)])] if X_val is not None else None
            m.fit(X_s, y_s, eval_set=ev, verbose=False)
            self.models_.append(m)

            if (step + 1) % 6 == 0 or step == self.horizon - 1:
                print(f"  Step {step + 1:3d}/{self.horizon} fitted.")

        print(f"[XGB] Done — {len(self.models_)} model(s).")
        return self

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict all horizon steps.  Returns shape (n_samples, horizon)."""
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        return np.column_stack([m.predict(Xv) for m in self.models_])

    def predict_one_step(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Single-step forecast from the first model."""
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        return self.models_[0].predict(Xv)

    # ── feature importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Average MDI importance across all horizon models."""
        scores = np.mean([m.feature_importances_ for m in self.models_], axis=0)
        names  = self.feature_names_ or [f"f{i}" for i in range(len(scores))]
        df = (pd.DataFrame({"feature": names, "importance": scores})
              .sort_values("importance", ascending=False)
              .head(top_n))
        print("\n[XGB] Top features:")
        print(df.to_string(index=False))
        return df

    def shap_values(
        self,
        X: pd.DataFrame | np.ndarray,
        step: int = 0,
    ) -> np.ndarray | None:
        """SHAP values for step `step` (requires the `shap` package)."""
        try:
            import shap # type: ignore
        except ImportError:
            print("[XGB] shap not installed — skipping.")
            return None
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        exp = shap.TreeExplainer(self.models_[step])
        return exp.shap_values(Xv)

    # ── cross-validation ──────────────────────────────────────────────────────

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series   | np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        from evaluation.metrics import compute_metrics
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        yv = y.values if isinstance(y, pd.Series) else y
        tscv   = TimeSeriesSplit(n_splits=n_splits)
        scores = {"rmse": [], "mae": [], "mape": []}

        for fold, (tr, va) in enumerate(tscv.split(Xv)):
            self.fit(Xv[tr], yv[tr])
            m = compute_metrics(yv[va], self.predict_one_step(Xv[va]))
            for k in scores:
                scores[k].append(m[k])
            print(f"  Fold {fold+1}  RMSE={m['rmse']:.1f}  MAPE={m['mape']:.2f}%")

        agg = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in scores.items()}
        print(f"\n[XGB] CV RMSE={agg['rmse']['mean']:.1f}±{agg['rmse']['std']:.1f}")
        return agg

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = MODEL_PATH) -> None:
        joblib.dump({"models": self.models_, "params": self.params,
                     "horizon": self.horizon,
                     "feature_names": self.feature_names_}, path)
        print(f"[XGB] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "XGBoostForecaster":
        s = joblib.load(path)
        inst = cls(params=s["params"], horizon=s["horizon"])
        inst.models_        = s["models"]
        inst.feature_names_ = s["feature_names"]
        print(f"[XGB] Loaded ← {path}")
        return inst

    def summary(self) -> str:
        return (f"XGBoostForecaster  horizon={self.horizon}h  "
                f"models={len(self.models_)}  "
                f"features={len(self.feature_names_)}")


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    train_df, test_df = train_test_split(df, test_months=3)
    X_train, y_train  = build_feature_matrix(train_df)
    X_test,  y_test   = build_feature_matrix(test_df)

    #xgb_m = XGBoostForecaster(horizon=24)
    xgb_m = XGBoostForecaster(horizon=1)#For now (one step forecasting)
    split_idx = int(len(X_train) * 0.9)

    X_tr = X_train[:split_idx]
    y_tr = y_train[:split_idx]

    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    #xgb_m.fit(X_train, y_train)
    xgb_m.fit(
    X_tr,
    y_tr,
    X_val,
    y_val
    )
    
    xgb_m.feature_importance()

    preds = xgb_m.predict_one_step(X_test)

    from evaluation.metrics import compute_metrics, print_metrics
    m = compute_metrics(y_test.values, preds, y_train.values)
    print_metrics(m, "XGBoost")
    xgb_m.save()
    
    train_preds = xgb_m.predict_one_step(X_train)

    train_metrics = compute_metrics(
        y_train.values,
        train_preds,
    )

    print_metrics(train_metrics, "TRAIN")