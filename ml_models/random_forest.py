"""
random_forest.py
----------------
Random Forest model for hourly Power_Load_kW forecasting.

Input data
----------
Full feature matrix from preprocessing.build_feature_matrix() on the
hourly-resampled PowerLoad_Dataset.csv.

Features
--------
- Randomized hyperparameter search (RandomizedSearchCV)
- Direct multi-step: one RF per horizon step
- Prediction intervals via tree-level percentiles (no extra package needed)
- MDI + permutation feature importance
- Time-series cross-validation
- Model persistence
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sk_perm_imp
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_data, train_test_split
from data.preprocessing import build_feature_matrix

TARGET     = "Power_Load_kW"
MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"

PARAM_GRID = {
    "n_estimators":      [200, 400, 600, 800],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
    "bootstrap":         [True, False],
}

DEFAULT_PARAMS = {
    "n_estimators":      400,
    "max_depth":         None,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "bootstrap":         True,
    "n_jobs":            -1,
    "random_state":      42,
}


class RandomForestForecaster:
    """
    Random Forest load forecaster with uncertainty quantification.

    Parameters
    ----------
    params  : RF hyperparameters. None → defaults.
    horizon : forecast horizon (hours). One model per step.
    """

    def __init__(
        self,
        params:  dict | None = None,
        horizon: int         = 24,
    ):
        self.params         = params or DEFAULT_PARAMS.copy()
        self.horizon        = horizon
        self.models_:        list[RandomForestRegressor] = []
        self.feature_names_: list[str] = []

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series   | np.ndarray,
        tune:       bool = False,
        tune_n_iter: int = 20,
    ) -> "RandomForestForecaster":
        """Train one RF model per forecast step."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        if tune:
            self.params = self._tune(X_train, y_train, tune_n_iter)

        print(f"\n[RF] Training Random Forest  horizon={self.horizon}h  "
              f"n_train={len(X_train)}  n_features={X_train.shape[1]} …")
        self.models_ = []

        for step in range(self.horizon):
            y_s = y_train[step:] if step == 0 else np.roll(y_train, -step)[:-step]
            X_s = X_train[:len(y_s)]
            m = RandomForestRegressor(**self.params)
            m.fit(X_s, y_s)
            self.models_.append(m)

            if (step + 1) % 6 == 0 or step == self.horizon - 1:
                print(f"  Step {step + 1:3d}/{self.horizon} fitted.")

        print(f"[RF] Done — {len(self.models_)} model(s).")
        return self

    def _tune(self, X: np.ndarray, y: np.ndarray, n_iter: int) -> dict:
        base  = RandomForestRegressor(random_state=42, n_jobs=-1)
        tscv  = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            base, PARAM_GRID, n_iter=n_iter, cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)
        best = search.best_params_
        best.update({"random_state": 42, "n_jobs": -1})
        print(f"[RF] Best params: {best}  RMSE={-search.best_score_:.2f}")
        return best

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Multi-step point forecast.  Returns shape (n_samples, horizon)."""
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        return np.column_stack([m.predict(Xv) for m in self.models_])

    def predict_one_step(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Single-step (step-1) forecast."""
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        return self.models_[0].predict(Xv)

    def predict_interval(
        self,
        X:          pd.DataFrame | np.ndarray,
        lower_pct:  float = 5.0,
        upper_pct:  float = 95.0,
        step:       int   = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prediction intervals using per-tree prediction percentiles.

        Returns (mean, lower, upper) each shape (n_samples,).
        """
        Xv  = X.values if isinstance(X, pd.DataFrame) else X
        # (n_trees, n_samples)
        tree_preds = np.array([t.predict(Xv) for t in self.models_[step].estimators_])
        mean  = tree_preds.mean(axis=0)
        lower = np.percentile(tree_preds, lower_pct, axis=0)
        upper = np.percentile(tree_preds, upper_pct, axis=0)
        return mean, lower, upper

    # ── feature importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Average MDI importance across all horizon models."""
        scores = np.mean([m.feature_importances_ for m in self.models_], axis=0)
        names  = self.feature_names_ or [f"f{i}" for i in range(len(scores))]
        df = (pd.DataFrame({"feature": names, "mdi_importance": scores})
              .sort_values("mdi_importance", ascending=False)
              .head(top_n))
        print("\n[RF] Top MDI feature importances:")
        print(df.to_string(index=False))
        return df

    def permutation_importance(
        self,
        X:         pd.DataFrame | np.ndarray,
        y:         pd.Series   | np.ndarray,
        step:      int = 0,
        n_repeats: int = 10,
        top_n:     int = 20,
    ) -> pd.DataFrame:
        """Permutation importance on the step-`step` model."""
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        yv = y.values if isinstance(y, pd.Series) else y
        res = sk_perm_imp(self.models_[step], Xv, yv,
                          n_repeats=n_repeats, random_state=42, n_jobs=-1)
        names = self.feature_names_ or [f"f{i}" for i in range(Xv.shape[1])]
        df = (pd.DataFrame({
            "feature":          names,
            "mean_importance":  res.importances_mean,
            "std_importance":   res.importances_std,
        }).sort_values("mean_importance", ascending=False).head(top_n))
        print(f"\n[RF] Permutation importance (step={step}):")
        print(df.to_string(index=False))
        return df

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
        print(f"\n[RF] CV RMSE={agg['rmse']['mean']:.1f}±{agg['rmse']['std']:.1f}")
        return agg

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = MODEL_PATH) -> None:
        joblib.dump({"models": self.models_, "params": self.params,
                     "horizon": self.horizon,
                     "feature_names": self.feature_names_}, path)
        print(f"[RF] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "RandomForestForecaster":
        s = joblib.load(path)
        inst = cls(params=s["params"], horizon=s["horizon"])
        inst.models_        = s["models"]
        inst.feature_names_ = s["feature_names"]
        print(f"[RF] Loaded ← {path}")
        return inst

    def summary(self) -> str:
        return (f"RandomForestForecaster  horizon={self.horizon}h  "
                f"n_trees={self.params.get('n_estimators','?')}  "
                f"features={len(self.feature_names_)}")


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    train_df, test_df = train_test_split(df, test_months=3)
    X_train, y_train  = build_feature_matrix(train_df)
    X_test,  y_test   = build_feature_matrix(test_df)

    rf = RandomForestForecaster(horizon=24)
    rf.fit(X_train, y_train)
    rf.feature_importance()

    preds         = rf.predict_one_step(X_test)
    mean, lo, hi  = rf.predict_interval(X_test)

    from evaluation.metrics import compute_metrics, print_metrics
    m = compute_metrics(y_test.values, preds, y_train.values)
    print_metrics(m, "Random Forest")
    rf.save()