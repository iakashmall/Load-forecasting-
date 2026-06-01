"""
exploratory_analysis.py
-----------------------
Exploratory Data Analysis for PowerLoad_Dataset.csv.

Generates
---------
01_load_overview.png          - full time-series + monthly bar
02_seasonality_profiles.png   - box plots by hour / weekday / month
03_feature_correlations.png   - Pearson heatmap
04_acf_pacf.png               - ACF / PACF on hourly load
05_stl_decomposition.png      - STL on daily mean load
06_weather_vs_load.png        - scatter matrix: weather features vs load
07_load_duration_curve.png    - sorted load duration curve
08_anomaly_detection.png      - IQR-based anomaly flags
09_dispatch_vs_load.png       - daily post-dispatch vs actual load
10_holiday_effect.png         - holiday vs non-holiday load comparison
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_loader import load_data, load_data_daily

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")


# ── Helpers ───────────────────────────────────────────────────────────────────
# Reusable helper to save figures 
def _save(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / f"{name}.png" # creates plots/01_load_overview.png 
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] → {path.name}")


# ── 01 Load overview ──────────────────────────────────────────────────────────
# Creates Daily load trends and Monthly average load 
def plot_load_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=False)#2 rows, 1 column

    # Daily mean time series
    daily_mean = df["Power_Load_kW"].resample("D").mean()# Converts hourly -> daily average 
    axes[0].plot(daily_mean.index, daily_mean.values,
                 lw=0.6, color="#2196F3", alpha=0.8, label="Daily mean")
    axes[0].plot(daily_mean.rolling(30).mean(),
                 lw=2, color="#FF5722", label="30-day rolling mean")
    axes[0].set_title("Power Load (kW) — Daily Mean Over Full Dataset", fontsize=13)
    axes[0].set_ylabel("Load (kW)")
    axes[0].legend()

    #Plot monthly averages as bar chart
    monthly = df["Power_Load_kW"].resample("ME").mean() # Producs monthly averages
    axes[1].bar(monthly.index, monthly.values, width=25,
                color="#4CAF50", alpha=0.85)
    axes[1].set_title("Monthly Average Load (kW)", fontsize=13)
    axes[1].set_ylabel("Avg Load (kW)")
    axes[1].tick_params(axis="x", rotation=45)# Rotate x-axis labels for better readability

    fig.tight_layout()
    _save(fig, "01_load_overview")


# ── 02 Seasonality profiles ───────────────────────────────────────────────────
# This section visualises : hourlycycles, weekly cycles, monthly cycles 
def plot_seasonality_profiles(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    df_plot = df.copy()
    df_plot["hour"]    = df.index.hour
    df_plot["dow"]     = df.index.dayofweek
    df_plot["month"]   = df.index.month

    kw = dict(flierprops={"marker": ".", "alpha": 0.3, "ms": 3})

    #Box plot shows median, quartiles, outliers, variablity 
    sns.boxplot(data=df_plot, x="hour",  y="Power_Load_kW",
                ax=axes[0], color="#42A5F5", **kw)
    axes[0].set_title("Load by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Load (kW)")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.boxplot(data=df_plot, x="dow", y="Power_Load_kW",
                ax=axes[1], color="#66BB6A", **kw)
    axes[1].set_xticklabels(day_labels)
    axes[1].set_title("Load by Day of Week")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    sns.boxplot(data=df_plot, x="month", y="Power_Load_kW",
                ax=axes[2], color="#FFA726", **kw)
    axes[2].set_xticklabels(month_labels, rotation=45)
    axes[2].set_title("Load by Month")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")

    fig.suptitle("Seasonality Profiles — PowerLoad_Dataset", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "02_seasonality_profiles")


# ── 03 Correlation heatmap ────────────────────────────────────────────────────

def plot_correlations(df: pd.DataFrame) -> None:
    cols = [
        "Power_Load_kW", "Temperature_C", "Humidity_%", "WindSpeed_mps",
        "Precipitation_mm", "DayOfWeek", "HolidayFlag",
        "Daily_PostDispatch_Load", "Weekly_PreDispatch_Projection",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    # Creates a correlation heatmap with annotations, masking the upper triangle for clarity
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )# Masking the upper triangle to avoid redundancy, 
    #annotating with correlation values, 
    #and using a diverging colormap centered at 0 for better visual distinction of positive vs negative correlations.
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_feature_correlations")


# ── 04 ACF / PACF ─────────────────────────────────────────────────────────────

def plot_acf_pacf(df: pd.DataFrame, lags: int = 72) -> None:
    series = df["Power_Load_kW"].dropna()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    plot_acf (series, lags=lags, ax=axes[0], alpha=0.05,
              title="ACF of Hourly Power Load (kW)")
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05,
              title="PACF of Hourly Power Load (kW)", method="ywm")
    fig.tight_layout()
    _save(fig, "04_acf_pacf")


# ── 05 STL decomposition ──────────────────────────────────────────────────────

def plot_stl_decomposition(daily_df: pd.DataFrame) -> None:
    series = daily_df["Power_Load_kW"].dropna()
    stl    = STL(series, period=7, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, component, label, color in zip(
        axes,
        [result.observed, result.trend, result.seasonal, result.resid],
        ["Observed", "Trend", "Seasonal (weekly)", "Residual"],
        ["#2196F3", "#FF5722", "#4CAF50", "#9E9E9E"],
    ):
        ax.plot(component, lw=0.9, color=color)
        ax.set_ylabel(label)
    axes[0].set_title("STL Decomposition — Daily Mean Power Load (kW)", fontsize=13)
    fig.tight_layout()
    _save(fig, "05_stl_decomposition")


# ── 06 Weather vs Load scatter ────────────────────────────────────────────────

def plot_weather_vs_load(df: pd.DataFrame) -> None:
    weather_cols = ["Temperature_C", "Humidity_%", "WindSpeed_mps", "Precipitation_mm"]
    weather_cols = [c for c in weather_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(weather_cols), figsize=(6 * len(weather_cols), 5))
    if len(weather_cols) == 1:
        axes = [axes]

    sample = df.sample(min(2000, len(df)), random_state=42)
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for ax, col, color in zip(axes, weather_cols, palette):
        ax.scatter(sample[col], sample["Power_Load_kW"],
                   s=6, alpha=0.35, color=color, edgecolors="none")
        # Trend line
        m, b = np.polyfit(sample[col].dropna(),
                           sample.loc[sample[col].notna(), "Power_Load_kW"], 1)
        xs = np.linspace(sample[col].min(), sample[col].max(), 100)
        ax.plot(xs, m * xs + b, color="red", lw=1.5)
        corr = df[[col, "Power_Load_kW"]].corr().iloc[0, 1]
        ax.set_title(f"{col}\n(r = {corr:.3f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Power Load (kW)" if ax is axes[0] else "")

    fig.suptitle("Weather Features vs Power Load (kW)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "06_weather_vs_load")


# ── 07 Load duration curve ────────────────────────────────────────────────────

def plot_load_duration_curve(df: pd.DataFrame) -> None:
    sorted_load = np.sort(df["Power_Load_kW"].dropna().values)[::-1]
    pct = np.linspace(0, 100, len(sorted_load))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pct, sorted_load, color="#7C4DFF", lw=2)
    ax.fill_between(pct, sorted_load, alpha=0.12, color="#7C4DFF")
    ax.set_xlabel("% of Hours (%)")
    ax.set_ylabel("Power Load (kW)")
    ax.set_title("Load Duration Curve", fontsize=13)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, "07_load_duration_curve")


# ── 08 Anomaly detection ──────────────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, iqr_k: float = 1.5) -> pd.DataFrame:
    Q1 = df["Power_Load_kW"].quantile(0.25)
    Q3 = df["Power_Load_kW"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_k * IQR, Q3 + iqr_k * IQR

    df = df.copy()
   
    df["anomaly"] = ((df["Power_Load_kW"] < lower) |
                     (df["Power_Load_kW"] > upper)).astype(int)
    n = df["anomaly"].sum()
    print(f"  [EDA] Anomalies: {n} ({100*n/len(df):.2f}%)  "
          f"bounds=[{lower:.1f}, {upper:.1f}] kW")

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df.index, df["Power_Load_kW"],
            lw=0.4, alpha=0.6, color="#90CAF9", label="Load")
    ax.scatter(df.index[df["anomaly"] == 1],
               df.loc[df["anomaly"] == 1, "Power_Load_kW"],
               color="#F44336", s=10, zorder=5, label=f"Anomalies ({n})")
    ax.axhline(lower, color="#FF9800", ls="--", lw=1)
    ax.axhline(upper, color="#FF9800", ls="--", lw=1)
    ax.set_title(f"Anomaly Detection (IQR × {iqr_k})", fontsize=13)
    ax.set_ylabel("Load (kW)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "08_anomaly_detection")
    return df


# ── 09 Dispatch vs actual ─────────────────────────────────────────────────────

def plot_dispatch_vs_actual(daily_df: pd.DataFrame) -> None:
    cols = ["Power_Load_kW", "Daily_PostDispatch_Load", "Weekly_PreDispatch_Projection"]
    cols = [c for c in cols if c in daily_df.columns]
    if len(cols) < 2:
        return

    fig, axes = plt.subplots(2, 1, figsize=(18, 8))

    # Time series overlay
    for col, color in zip(cols, ["#263238", "#2196F3", "#FF5722"]):
        axes[0].plot(daily_df.index, daily_df[col],
                     lw=1.2, alpha=0.8, color=color, label=col)
    axes[0].set_title("Actual vs Dispatch Loads (Daily)", fontsize=13)
    axes[0].set_ylabel("Load (kW)")
    axes[0].legend()

    # Scatter: post-dispatch vs actual
    if "Daily_PostDispatch_Load" in daily_df.columns:
        x = daily_df["Daily_PostDispatch_Load"].dropna()
        y = daily_df.loc[x.index, "Power_Load_kW"]
        axes[1].scatter(x, y, s=6, alpha=0.4, color="#4CAF50")
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        axes[1].plot(xs, m * xs + b, "r--", lw=1.5)
        corr = x.corr(y)
        axes[1].set_title(f"Daily PostDispatch vs Actual Load  (r={corr:.3f})", fontsize=12)
        axes[1].set_xlabel("Daily_PostDispatch_Load (kW)")
        axes[1].set_ylabel("Power_Load_kW (daily mean)")

    fig.tight_layout()
    _save(fig, "09_dispatch_vs_load")


# ── 10 Holiday effect ─────────────────────────────────────────────────────────

def plot_holiday_effect(df: pd.DataFrame) -> None:
    if "HolidayFlag" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    sns.boxplot(data=df, x="HolidayFlag", y="Power_Load_kW",
                ax=axes[0], palette=["#42A5F5", "#EF5350"])
    axes[0].set_xticklabels(["Non-Holiday", "Holiday"])
    axes[0].set_title("Load Distribution: Holiday vs Non-Holiday")
    axes[0].set_ylabel("Power Load (kW)")

    # Hourly mean profile
    df_h = df.copy()
    df_h["hour"] = df.index.hour
    hol    = df_h[df_h["HolidayFlag"] == 1].groupby("hour")["Power_Load_kW"].mean()
    nonhol = df_h[df_h["HolidayFlag"] == 0].groupby("hour")["Power_Load_kW"].mean()
    axes[1].plot(nonhol, color="#42A5F5", lw=2, label="Non-Holiday")
    axes[1].plot(hol,    color="#EF5350", lw=2, label="Holiday")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Mean Load (kW)")
    axes[1].set_title("Hourly Load Profile: Holiday vs Non-Holiday")
    axes[1].legend()

    fig.suptitle("Holiday Effect on Power Load", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "10_holiday_effect")


# ── Descriptive stats ─────────────────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    s = df["Power_Load_kW"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    s["skewness"] = df["Power_Load_kW"].skew()
    s["kurtosis"] = df["Power_Load_kW"].kurt()
    s["cv_%"]     = 100 * df["Power_Load_kW"].std() / df["Power_Load_kW"].mean()
    print("\n── Power_Load_kW Statistics ─────────────────────────")
    print(s.to_string())
    return s


# ── Master runner ─────────────────────────────────────────────────────────────

def run_full_eda(csv_path: str | None = None) -> None:
    print("\n" + "═" * 50)
    print("  PowerLoad EDA — starting all plots")
    print("═" * 50 + "\n")

    df       = load_data(csv_path)
    daily_df = load_data_daily(csv_path)

    descriptive_statistics(df)
    plot_load_overview(df)
    plot_seasonality_profiles(df)
    plot_correlations(df)
    plot_acf_pacf(df)
    plot_stl_decomposition(daily_df)
    plot_weather_vs_load(df)
    plot_load_duration_curve(df)
    plot_anomalies(df)
    plot_dispatch_vs_actual(daily_df)
    plot_holiday_effect(df)

    print(f"\n[EDA] All plots saved → {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    run_full_eda()