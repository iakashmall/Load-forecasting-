"""
data_loader.py
--------------
Loads PowerLoad_Dataset.csv and prepares it for every downstream model.

Dataset facts (from EDA):
  - 10 000 rows, 10 columns
  - Date range : 2018-01-01  →  2023-06-30
  - Timestamps are IRREGULAR  (gaps of 1 h, 2 h, 3 h … present)
  - Target     : Power_Load_kW
  - Exogenous  : Temperature_C, Humidity_%, WindSpeed_mps,
                 Precipitation_mm, DayOfWeek, HolidayFlag,
                 Daily_PostDispatch_Load, Weekly_PreDispatch_Projection

Pipeline (Preprocessing stages)
--------
1. Parse Timestamp → DatetimeIndex
2. Sort chronologically
3. Resample to a uniform hourly grid (mean for numeric, ffill for flags)
4. Expose `load_data()` → full hourly DataFrame
5. Expose `load_data_daily()` → daily-aggregated DataFrame (for ARIMA/SARIMA/STL)
6. Expose `train_test_split()` → chronological split
"""

from pathlib import Path # Modern way to handle file paths (cross-platform, more robust than os.path)
import numpy as np
import pandas as pd

# ── Column map  ──────────────────────────────────────────────────────────────
TARGET        = "Power_Load_kW"

# Columns that come straight from the raw CSV
RAW_FEATURE_COLS = [
    "Temperature_C",
    "Humidity_%",
    "WindSpeed_mps",
    "Precipitation_mm",
    "DayOfWeek",
    "HolidayFlag",
    "Daily_PostDispatch_Load",
    "Weekly_PreDispatch_Projection",
]

# Default CSV location (place PowerLoad_Dataset.csv here, or pass the path)
DEFAULT_CSV = Path(__file__).parent / "PowerLoad_Dataset.csv"


# ── Main loader ───────────────────────────────────────────────────────────────

def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:# function returns pandas dataframe, takes optional csv_path argument which can be a string, Path object, or None
    """
    Load and clean PowerLoad_Dataset.csv → uniform hourly DataFrame.

    Parameters
    ----------
    csv_path : path to CSV.  If None, looks for PowerLoad_Dataset.csv
               in the same folder as this file.

    Returns
    -------
    pd.DataFrame
        Index  : DatetimeIndex  (hourly, no gaps, UTC-naive)
        Columns: Power_Load_kW + all 8 feature columns + derived
                 is_weekend, hour, month, day_of_year
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(# raise error if the specified path does not exist, with a helpful message guiding the user to place the CSV in the correct location or pass its path
            f"[data_loader] CSV not found at '{path}'.\n"
            f"  → Copy PowerLoad_Dataset.csv to '{path.parent}' "
            f"or pass its path to load_data()."
        )

    print(f"[data_loader] Reading {path.name} …")
    #converts "Timestamp" column into datetime objects automatically
    raw = pd.read_csv(path, parse_dates=["Timestamp"])#  parsing the "Timestamp" column as dates
    print(f"[data_loader] Raw shape : {raw.shape}")

    # ── 1. Rename & sort ──────────────────────────────────────────────────────
    raw = raw.rename(columns={"Timestamp": "timestamp"})
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    # ── 2. Set DatetimeIndex ──────────────────────────────────────────────────
    raw = raw.set_index("timestamp")# sets the "timestamp" column as the index of the DataFrame, which is important for time series analysis and resampling
    raw.index = pd.to_datetime(raw.index)

    # ── 3. Resample to uniform 1-hour grid ────────────────────────────────────
    #   Numeric cols  → mean within each hour
    #   Flag cols     → max  (if any reading in the hour is a flag, keep it)
    flag_cols    = ["DayOfWeek", "HolidayFlag"]
    numeric_cols = [c for c in raw.columns if c not in flag_cols]#takes all columns EXCEPT flags

    hourly_num   = raw[numeric_cols].resample("h").mean()#creates uniform hourly intervals
    hourly_flags = raw[flag_cols].resample("h").max()

    df = pd.concat([hourly_num, hourly_flags], axis=1)

    # ── 4. Fill remaining NaNs introduced by resampling ───────────────────────
    #   DayOfWeek can be filled from the index
    df["DayOfWeek"] = df.index.dayofweek + 1   # 1=Mon … 7=Sun  (matches dataset)
    df["HolidayFlag"] = df["HolidayFlag"].fillna(0).astype(int)

    #   For weather & load: forward-fill then backward-fill short gaps
    df = df.ffill().bfill()

    # ── 5. Derived calendar columns ───────────────────────────────────────────
    df["hour"]        = df.index.hour
    df["month"]       = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)

    print(
        f"[data_loader] Hourly grid : {len(df):,} rows  "
        f"({df.index.min().date()} → {df.index.max().date()})"
    )
    _print_nulls(df)# checks for any remaining null values in the DataFrame after resampling and filling, and prints a message accordingly
    return df


def load_data_daily(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Return a daily-aggregated DataFrame suitable for statistical models
    (ARIMA, SARIMA, STL).

    Aggregation rules
    -----------------
    Power_Load_kW               → mean
    Temperature_C               → mean
    Humidity_%                  → mean
    WindSpeed_mps               → mean
    Precipitation_mm            → sum
    DayOfWeek                   → mode (most frequent day-of-week label)
    HolidayFlag                 → max  (1 if any hour was holiday)
    Daily_PostDispatch_Load     → mean
    Weekly_PreDispatch_Projection → mean

    Returns
    -------
    pd.DataFrame  (DatetimeIndex, freq='D')
    """
    df = load_data(csv_path)

    agg = { # dictionary defining how to aggregate each column when resampling to daily frequency
        "Power_Load_kW":                "mean",
        "Temperature_C":                "mean",
        "Humidity_%":                   "mean",
        "WindSpeed_mps":                "mean",
        "Precipitation_mm":             "sum",
        "DayOfWeek":                    lambda x: x.mode().iloc[0],# mode()= most frequent value, .iloc[0] = takes the first one in case of ties
        "HolidayFlag":                  "max",
        "Daily_PostDispatch_Load":      "mean",
        "Weekly_PreDispatch_Projection":"mean",
    } 
    daily = df.resample("D").agg(agg)# 'D'= daily frequency, .agg(agg) applies the specified aggregation rules to each column.  

    # Recalculate clean calendar cols on daily index
    daily["month"]       = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear
    daily["is_weekend"]  = (daily.index.dayofweek >= 5).astype(int)

    print(f"[data_loader] Daily grid  : {len(daily):,} rows")
    return daily


# ── Train / test split ────────────────────────────────────────────────────────

def train_test_split(
    df: pd.DataFrame,
    test_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — last `test_months` months become the test set.
    
    Time-series must preserve order.
    Random splitting causes:future leakage.
    
    Returns
    -------
    (train_df, test_df)
    """
    split_date = df.index.max() - pd.DateOffset(months=test_months)# calculates the date that is `test_months` months before the last date in the DataFrame index, which will be used as the cutoff for splitting the data into training and testing sets.
    train = df[df.index <= split_date].copy()
    test  = df[df.index >  split_date].copy()
    print(
        f"[data_loader] Train : {len(train):>6,} rows  "
        f"({train.index.min().date()} → {train.index.max().date()})\n"
        f"[data_loader] Test  : {len(test):>6,} rows  "
        f"({test.index.min().date()} → {test.index.max().date()})"
    )
    return train, test


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_nulls(df: pd.DataFrame) -> None:
    n = df.isnull().sum()
    bad = n[n > 0]
    if bad.empty:
        print("[data_loader] No nulls after resampling ✓")
    else:
        print(f"[data_loader] Remaining nulls:\n{bad}")


def get_feature_columns(df: pd.DataFrame, exclude_target: bool = True) -> list[str]:
    """
    Return the list of model-ready feature columns (excludes target,
    raw calendar duplicates already captured by Fourier features).
    """
    base = [
        "Temperature_C", "Humidity_%", "WindSpeed_mps", "Precipitation_mm",
        "DayOfWeek", "HolidayFlag",
        "Daily_PostDispatch_Load", "Weekly_PreDispatch_Projection",
        "hour", "month", "is_weekend",
    ]
    return [c for c in base if c in df.columns]


# ── Standalone sanity check ───────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print("\n── Hourly Head ──────────────────────────────────")
    print(df.head(3).to_string())
    print("\n── Describe ─────────────────────────────────────")
    print(df.describe().round(2).to_string())

    daily = load_data_daily()
    print("\n── Daily Head ───────────────────────────────────")
    print(daily.head(3).to_string())

    train, test = train_test_split(df)
    print("\n── Train/Test Shapes ─────────────────────────────")# 1
    print(f"Train: {train.shape}  |  Test: {test.shape}")   # 2  
    
    