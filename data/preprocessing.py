"""
preprocessing.py
----------------
Feature engineering pipeline built exactly around PowerLoad_Dataset.csv.

All columns are created from the real dataset; no synthetic data is used.

Feature groups produced
-----------------------
Group A - Raw dataset columns (passed through)
    Temperature_C, Humidity_%, WindSpeed_mps, Precipitation_mm,
    DayOfWeek, HolidayFlag,
    Daily_PostDispatch_Load, Weekly_PreDispatch_Projection

Group B - Calendar features (derived from DatetimeIndex)
    hour, day_of_week, day_of_month, month, quarter, year,
    day_of_year, week_of_year, is_weekend,
    is_morning_peak, is_evening_peak, is_night, is_business_hour

Group C -
    Fourier (cyclical) encodings
    sin/cos for daily (24 h), weekly (168 h), annual (8 760 h) cycles

Group D - Autoregressive lag features  (lags of Power_Load_kW)
    lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h

Group E - Rolling statistics of Power_Load_kW
    mean, std, min, max  for windows 6 h, 12 h, 24 h, 48 h, 168 h

Group F - Difference features
    diff_1h, diff_24h, diff_168h

Group G - Domain / interaction features
    hdd, cdd, temp_sq, temp_humidity_interaction,
    wind_chill, heat_index, load_temp_interaction
"""

import sys #path manipulation for imports
from pathlib import Path #working with file paths

import joblib #saving/loading scikit-learn objects (like scalers)
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #scaling features for machine learning models

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET      = "Power_Load_kW"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"#path where scaler object will be saved.

# ═══════════════════════════════════════════════════════════════════
#  Feature building blocks
# ═══════════════════════════════════════════════════════════════════

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"]          = idx.hour
    df["day_of_week"]   = idx.dayofweek          # 0 = Mon
    df["day_of_month"]  = idx.day
    df["month"]         = idx.month
    df["quarter"]       = idx.quarter
    df["year"]          = idx.year
    df["day_of_year"]   = idx.dayofyear
    df["week_of_year"]  = idx.isocalendar().week.astype(int)
    df["is_weekend"]    = (idx.dayofweek >= 5).astype(int)
    df["is_morning_peak"] = ((idx.hour >= 8)  & (idx.hour <= 10)).astype(int)
    df["is_evening_peak"] = ((idx.hour >= 18) & (idx.hour <= 21)).astype(int)
    df["is_night"]        = ((idx.hour >= 22) | (idx.hour <=  5)).astype(int)
    df["is_business_hour"]= (
        (idx.hour >= 9) & (idx.hour <= 17) & (idx.dayofweek < 5)
    ).astype(int)
    return df


def add_fourier_features(
    df: pd.DataFrame,
    periods: dict[str, float] | None = None,
    n_harmonics: int = 3,
) -> pd.DataFrame:
    """
    Encode cyclical patterns using sine / cosine pairs.
    `periods` maps name → cycle length in hours.
    """
    df = df.copy()
    if periods is None:
        periods = {"daily": 24.0, "weekly": 168.0, "annual": 8760.0}

    t = np.arange(len(df), dtype=float)#  
    for name, period in periods.items():
        for k in range(1, n_harmonics + 1):
            df[f"sin_{name}_k{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{name}_k{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Autoregressive lag features of Power_Load_kW."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f"lag_{lag}h"] = df[TARGET].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling statistics of Power_Load_kW (using shift(1) to avoid leakage)."""
    df = df.copy()
    if windows is None:
        windows = [6, 12, 24, 48, 168]
    for w in windows:
        roll = df[TARGET].shift(1).rolling(window=w, min_periods=1)
        df[f"roll_mean_{w}h"] = roll.mean()
        df[f"roll_std_{w}h"]  = roll.std()
        df[f"roll_min_{w}h"]  = roll.min()
        df[f"roll_max_{w}h"]  = roll.max()
    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """First-difference and seasonal-difference of Power_Load_kW."""
    df = df.copy()
    df["diff_1h"]   = df[TARGET].diff(1)
    df["diff_24h"]  = df[TARGET].diff(24)
    df["diff_168h"] = df[TARGET].diff(168)
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-knowledge features derived from the weather columns
    that are present in PowerLoad_Dataset.csv.
    """
    df = df.copy()
    T  = df.get("Temperature_C",  pd.Series(25, index=df.index))
    H  = df.get("Humidity_%",     pd.Series(60, index=df.index))
    W  = df.get("WindSpeed_mps",  pd.Series(5,  index=df.index))

    # Heating / cooling degree values (base 18 °C)
    df["hdd"] = np.maximum(18 - T, 0)
    df["cdd"] = np.maximum(T - 18, 0)

    # Nonlinear temperature
    df["temp_sq"] = T ** 2 # captures potential nonlinear effects of temperature on load

    # Cross-interaction terms
    df["temp_humidity_interaction"] = T * H / 100.0
    df["wind_chill"]   = T - 0.7 * W          # simplified wind-chill index
    df["heat_index"]   = T + 0.33 * (H / 100 * 6.105) - 4.0   # simplified HI

    # Load × temperature (peak-demand interaction)
    if TARGET in df.columns:
        df["load_temp_interaction"] = df[TARGET] * T

    # Dispatch-based features (already in dataset)
    if "Daily_PostDispatch_Load" in df.columns and "Weekly_PreDispatch_Projection" in df.columns:
        df["dispatch_ratio"] = (
            df["Daily_PostDispatch_Load"] / (df["Weekly_PreDispatch_Projection"] + 1e-6)
        )
        df["dispatch_diff"] = (
            df["Daily_PostDispatch_Load"] - df["Weekly_PreDispatch_Projection"]
        )

    return df


# ═══════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the complete feature engineering pipeline on a DataFrame that
    has already been loaded by data_loader.load_data().

    Returns
    -------
    X : pd.DataFrame   (all feature columns, NaN rows removed)
    y : pd.Series      (Power_Load_kW, aligned with X)
    """
    df = add_calendar_features(df)
    df = add_fourier_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)
    df = add_domain_features(df)

    if drop_na:
        df = df.dropna()

    y = df[TARGET].copy()

    # Drop the target and any date-index proxy columns from X
    drop_cols = [TARGET, "day_of_year"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Drop duplicate DayOfWeek if it conflicts with day_of_week
    if "DayOfWeek" in X.columns and "day_of_week" in X.columns:
        X = X.drop(columns=["DayOfWeek"])   # day_of_week (0-based) is cleaner

    return X, y


# ═══════════════════════════════════════════════════════════════════
#  Feature engineering for daily data (statistical models)
# ═══════════════════════════════════════════════════════════════════

def build_daily_features(
    daily_df: pd.DataFrame,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Lighter feature set for daily-aggregated data used by
    ARIMA / SARIMA / STL.  Returns (X, y) where y = daily mean load.
    """
    df = daily_df.copy()

    idx = df.index
    df["day_of_week"]   = idx.dayofweek
    df["month"]         = idx.month
    df["week_of_year"]  = idx.isocalendar().week.astype(int)
    df["is_weekend"]    = (idx.dayofweek >= 5).astype(int)
    df["quarter"]       = idx.quarter

    # Fourier for weekly (7 d) and annual (365 d) cycles
    t = np.arange(len(df), dtype=float)
    for name, period in [("weekly", 7.0), ("annual", 365.0)]:
        for k in range(1, 3):
            df[f"sin_{name}_k{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{name}_k{k}"] = np.cos(2 * np.pi * k * t / period)

    # Lag / rolling on daily target
    for lag in [1, 2, 7, 14]:
        df[f"lag_{lag}d"] = df[TARGET].shift(lag)
    for w in [3, 7, 14]:
        roll = df[TARGET].shift(1).rolling(w, min_periods=1)
        df[f"roll_mean_{w}d"] = roll.mean()
        df[f"roll_std_{w}d"]  = roll.std()

    # Weather features
    if "Temperature_C" in df.columns:
        df["hdd"]     = np.maximum(18 - df["Temperature_C"], 0)
        df["cdd"]     = np.maximum(df["Temperature_C"] - 18, 0)
        df["temp_sq"] = df["Temperature_C"] ** 2
    if "Precipitation_mm" in df.columns:
        df["log_precip"] = np.log1p(df["Precipitation_mm"])

    if drop_na:
        df = df.dropna()

    y = df[TARGET].copy()
    drop_cols = [TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()# drop the target column from the feature set, leaving only the engineered features and any other columns that are not the target.
    return X, y # returns the feature matrix X and target vector y for the daily-aggregated data.


# ═══════════════════════════════════════════════════════════════════
#  Scaling helpers
# ═══════════════════════════════════════════════════════════════════

def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    method: str = "standard",
    save:   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler on X_train, transform both splits.

    Parameters
    ----------
    method : 'standard' (Z-score) | 'minmax' ([0, 1])
    """
    Scaler = StandardScaler if method == "standard" else MinMaxScaler
    scaler = Scaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    if save:
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocessing] Scaler saved → {SCALER_PATH}")
    return X_tr, X_te


def load_scaler():
    return joblib.load(SCALER_PATH)


# ═══════════════════════════════════════════════════════════════════
#  LSTM sequence builder
# ═══════════════════════════════════════════════════════════════════

def create_sequences(
    data:    np.ndarray,
    seq_len: int = 168,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over `data` (2-D array, target = column 0).

    Returns
    -------
    X_seq : (n_windows, seq_len, n_features)
    y_seq : (n_windows, horizon)
    """
    X_list, y_list = [], []
    for i in range(seq_len, len(data) - horizon + 1):
        X_list.append(data[i - seq_len : i])
        y_list.append(data[i : i + horizon, 0])
    return np.array(X_list), np.array(y_list)


# ═══════════════════════════════════════════════════════════════════
#  Sanity check
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.data_loader import load_data, train_test_split

    df = load_data()
    train_df, test_df = train_test_split(df)

    X_train, y_train = build_feature_matrix(train_df)
    X_test,  y_test  = build_feature_matrix(test_df)

    print(f"\nX_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print(f"\nFeatures ({X_train.shape[1]}):")
    for i, c in enumerate(X_train.columns, 1):
        print(f"  {i:3d}. {c}")