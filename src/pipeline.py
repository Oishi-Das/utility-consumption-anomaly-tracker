"""
src/pipeline.py

Turns raw meter readings into a clean daily consumption series, then flags
anomalous days using two independent statistical methods and estimates the
cost impact of flagged spikes.

Two real input shapes are supported out of the box:
    - "interval"   : meter reports consumption *per interval* (e.g. kWh every
                     15 minutes). Daily value = sum of same-day intervals.
    - "cumulative" : meter reports a running total (e.g. a water meter that
                     only ticks up). Daily value = the *average daily rate*
                     between two successive readings, i.e.
                     (reading_t - reading_t-1) / (days_elapsed between them).
                     This matters for sensors that aren't read every single
                     day -- treating a 5-day gap as a "1-day consumption
                     spike" would be wrong, so we normalise by elapsed time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_interval_series(path: str, timestamp_col: str = "timestamp",
                          value_col: str = "consumption_kwh") -> pd.Series:
    """Load a per-interval meter file and collapse it to a daily total."""
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    daily = (
        df.set_index(timestamp_col)[value_col]
        .resample("D")
        .sum(min_count=1)   # a day with zero readings stays NaN, not 0
        .dropna()
    )
    daily.index.name = "date"
    daily.name = "value"
    return daily


def load_cumulative_series(path: str, timestamp_col: str = "timestamp",
                            value_col: str = "cumulative_kl") -> tuple[pd.Series, int]:
    """
    Load a cumulative (running-total) meter file and convert it to an
    average daily consumption rate between successive readings.

    Real cumulative meters occasionally go BACKWARDS (meter replaced,
    counter reset, misread) -- a negative delta there isn't "negative
    consumption", it's a data-quality event. Those points are excluded
    from the consumption series and counted so the pipeline can report
    them honestly instead of quietly feeding a nonsense value into the
    anomaly detector.

    Returns (clean_daily_rate_series, n_meter_reset_events_excluded).
    """
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.dropna(subset=[value_col])

    # the meter reading barely changes within a day -> one reading per day
    daily_reading = df.set_index(timestamp_col)[value_col].resample("D").last().dropna()

    cumulative = daily_reading.sort_index()
    days_elapsed = cumulative.index.to_series().diff().dt.days
    delta = cumulative.diff()

    daily_rate = (delta / days_elapsed).iloc[1:]   # first row has no prior reading
    daily_rate.index.name = "date"
    daily_rate.name = "value"
    daily_rate = daily_rate.dropna()

    is_meter_reset = daily_rate < 0
    n_reset_events = int(is_meter_reset.sum())
    clean = daily_rate[~is_meter_reset]
    return clean, n_reset_events


def rolling_zscore_flags(series: pd.Series, window: int = 14, threshold: float = 3.0,
                          use_log: bool = False) -> pd.DataFrame:
    """
    Flag points whose rolling z-score (computed from the PRECEDING window,
    never the current point) exceeds `threshold` in absolute value.

    `use_log=True` computes the z-score on log1p(series) instead of the raw
    values. Consumption-rate data (especially averaged over uneven gaps
    between meter readings) is often right-skewed with a long tail of
    legitimately-high-but-normal days; scoring those on the raw scale
    flags a large chunk of the whole series as "anomalous", which isn't
    useful to anyone. `rolling_mean`/`rolling_std` in the returned frame
    are still computed on the ORIGINAL scale, so the human-readable
    baseline and any downstream cost-impact math stay in real units.

    The rolling std is floored at 5% of the series' overall (global) std
    (computed in whichever space the z-score itself uses). Without this, a
    window that happens to be unusually flat produces a near-zero local
    std, and dividing by it turns an ordinary day into an absurd z-score
    of thousands -- a textbook division-by-small-number bug that silently
    wrecks the whole downstream ranking if left in.
    """
    stat_series = np.log1p(series) if use_log else series

    roll_mean_stat = stat_series.shift(1).rolling(window, min_periods=max(5, window // 2)).mean()
    roll_std_stat = stat_series.shift(1).rolling(window, min_periods=max(5, window // 2)).std(ddof=0)

    global_std_stat = stat_series.std(ddof=0)
    std_floor = max(global_std_stat * 0.05, 1e-9)
    safe_std = roll_std_stat.clip(lower=std_floor)

    z = (stat_series - roll_mean_stat) / safe_std
    flagged = z.abs() > threshold

    # display/cost-impact baseline stays in original units regardless of use_log
    roll_mean_display = series.shift(1).rolling(window, min_periods=max(5, window // 2)).mean()
    roll_std_display = series.shift(1).rolling(window, min_periods=max(5, window // 2)).std(ddof=0)

    return pd.DataFrame({
        "value": series,
        "rolling_mean": roll_mean_display,
        "rolling_std": roll_std_display,
        "zscore": z,
        "zscore_flag": flagged.fillna(False),
    })


def iqr_flags(series: pd.Series, k: float = 1.5, use_log: bool = False) -> pd.Series:
    """Flag points outside Q1 - k*IQR .. Q3 + k*IQR (global, not rolling).
    See `rolling_zscore_flags` for why `use_log` matters on skewed series."""
    stat_series = np.log1p(series) if use_log else series
    q1, q3 = stat_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (stat_series < lower) | (stat_series > upper)


def detect_anomalies(series: pd.Series, window: int = 14, z_threshold: float = 3.0,
                      iqr_k: float = 1.5, use_log: bool = False) -> pd.DataFrame:
    """
    Combine the rolling z-score method with a global IQR cross-check.

    severity:
        "high"     - both methods agree it's anomalous
        "moderate" - only the (more sensitive) z-score method flags it
    direction:
        "spike" / "drop" relative to the rolling mean (or the series
        median, for points too early to have a rolling mean yet)

    Set `use_log=True` for right-skewed, non-negative series (e.g. a
    consumption rate derived from sparse meter readings) -- see
    `rolling_zscore_flags` docstring for why.
    """
    result = rolling_zscore_flags(series, window=window, threshold=z_threshold, use_log=use_log)
    result["iqr_flag"] = iqr_flags(series, k=iqr_k, use_log=use_log)
    result["is_anomaly"] = result["zscore_flag"] | result["iqr_flag"]

    baseline = result["rolling_mean"].fillna(series.median())
    result["direction"] = np.where(result["value"] >= baseline, "spike", "drop")
    result["severity"] = np.where(
        result["zscore_flag"] & result["iqr_flag"], "high",
        np.where(result["is_anomaly"], "moderate", "none")
    )
    return result


def estimate_cost_impact(anomaly_df: pd.DataFrame, rate_per_unit: float) -> pd.DataFrame:
    """
    Estimate the $ cost of each flagged SPIKE as the excess consumption
    above the rolling baseline, times a unit rate. Drops get no cost
    figure (a drop isn't "wasted spend"; if anything it may indicate a
    sensor fault or an outage worth investigating on its own).
    """
    df = anomaly_df.copy()
    baseline = df["rolling_mean"].fillna(df["value"].median())
    excess = (df["value"] - baseline).clip(lower=0)
    df["excess_vs_baseline"] = excess
    df["estimated_cost_impact"] = np.where(
        (df["is_anomaly"]) & (df["direction"] == "spike"),
        excess * rate_per_unit,
        0.0,
    )
    return df
