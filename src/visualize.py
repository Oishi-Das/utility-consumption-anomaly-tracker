"""
src/visualize.py

Saves one PNG per metric: the daily trend line with flagged anomalies
marked on top, plus the rolling baseline band for context.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless/server-safe backend
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_trend(result_df: pd.DataFrame, title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(result_df.index, result_df["value"], color="#2a6f97", linewidth=1.2, label="Daily value")
    ax.plot(result_df.index, result_df["rolling_mean"], color="#adb5bd", linewidth=1,
            linestyle="--", label="Rolling baseline")

    spikes = result_df[result_df["is_anomaly"] & (result_df["direction"] == "spike")]
    drops = result_df[result_df["is_anomaly"] & (result_df["direction"] == "drop")]

    ax.scatter(spikes.index, spikes["value"], color="#d62828", zorder=5, s=28, label="Flagged spike")
    ax.scatter(drops.index, drops["value"], color="#f4a261", zorder=5, s=28, label="Flagged drop")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
