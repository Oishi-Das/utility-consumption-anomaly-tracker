"""
src/modeling.py

IMPORTANT - read this before citing these numbers anywhere:

No public, open dataset of Sewage Treatment Plant (STP) or Water Treatment
Plant (WTP) throughput exists at the facility level (it's operational data
utilities and site operators don't publish). To still cover all four lines
named in the job description -- raw water, electricity, STP, and WTP -- this
module DERIVES synthetic WTP output and STP treated-volume series from the
*real* raw water consumption series, using a fixed efficiency/return ratio
plus noise and a handful of injected anomalies.

These two series are clearly synthetic and are labelled as such everywhere
they appear (README, chart titles, Excel sheet names). They exist to show
the pipeline generalises to more than one metric, not to claim real WTP/STP
sensor data. Electricity and raw water, by contrast, are the real metered
UNICON readings (see README for the citation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_wtp_output(raw_water: pd.Series, efficiency: float = 0.92,
                       noise_std: float = 0.02, seed: int = 7) -> pd.Series:
    """
    Model WTP (Water Treatment Plant) output as a fraction of raw water
    intake, plus small day-to-day process noise, plus a few injected
    efficiency-drop anomalies (e.g. filter clogging, pump wear).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=1.0, scale=noise_std, size=len(raw_water))
    output = raw_water.values * efficiency * noise

    n_anomalies = max(1, len(raw_water) // 40)
    anomaly_idx = rng.choice(len(raw_water), size=n_anomalies, replace=False)
    output[anomaly_idx] *= rng.uniform(0.55, 0.75, size=n_anomalies)  # efficiency drop

    series = pd.Series(output, index=raw_water.index, name="value")
    return series.clip(lower=0)


def derive_stp_treated(raw_water: pd.Series, return_ratio: float = 0.80,
                        noise_std: float = 0.03, seed: int = 11) -> pd.Series:
    """
    Model STP (Sewage Treatment Plant) treated volume as a fraction of raw
    water consumption that returns as wastewater, plus process noise, plus
    a few injected overflow/backlog anomalies.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=1.0, scale=noise_std, size=len(raw_water))
    treated = raw_water.values * return_ratio * noise

    n_anomalies = max(1, len(raw_water) // 35)
    anomaly_idx = rng.choice(len(raw_water), size=n_anomalies, replace=False)
    treated[anomaly_idx] *= rng.uniform(1.3, 1.6, size=n_anomalies)  # overflow/backlog spike

    series = pd.Series(treated, index=raw_water.index, name="value")
    return series.clip(lower=0)


def validate_detector(series: pd.Series, detect_fn, n_inject: int = 12,
                       spike_range: tuple[float, float] = (0.35, 0.60),
                       seed: int = 42) -> dict:
    """
    Sanity-check the anomaly detector against KNOWN ground truth:
    take a real series, inject `n_inject` deliberate spikes at known
    positions, run the detector, and report precision/recall against
    those known positions. This is the validation step a pure "did it
    find anything" demo is missing.
    """
    rng = np.random.default_rng(seed)
    noisy = series.copy()
    eligible = np.arange(10, len(series) - 10)  # avoid the warm-up window
    injected_idx = rng.choice(eligible, size=min(n_inject, len(eligible)), replace=False)

    for idx in injected_idx:
        bump = rng.uniform(*spike_range)
        noisy.iloc[idx] = noisy.iloc[idx] * (1 + bump)

    result = detect_fn(noisy)
    flagged_idx = np.where(result["is_anomaly"].values)[0]

    true_set, flagged_set = set(injected_idx), set(flagged_idx)
    true_positives = len(true_set & flagged_set)
    precision = true_positives / len(flagged_set) if flagged_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0

    return {
        "n_injected": len(true_set),
        "n_flagged_total": len(flagged_set),
        "true_positives": true_positives,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }
