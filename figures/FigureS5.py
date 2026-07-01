#!/usr/bin/env python3
"""Figure S5 - Benchmarking UniCAV against existing HIV-1 bNAb-resistance
predictors on the VRC-CH31 antibody (CH505 autologous viruses). Grouped-bar
metric plot ported from benchmark_ch505/06_compare_all.py (save_metric_barplot),
here showing AUROC, Accuracy, Balanced accuracy, F1 score and MCC per method.
Data: source_data/FigureS5.xlsx (sheet 'benchmark'). Output -> FigureS5_panels/.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hiv_mpl_cache"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
XLSX = HERE / "source_data" / "FigureS5.xlsx"
OUT = HERE / "FigureS5_panels"
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 10, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

METRIC_LABELS = {"AUC": "AUC", "Acc": "Accuracy", "Balanced_Acc": "Balanced acc.",
                 "F1": "F1 score", "MCC": "MCC"}
COLORS = ["#8EA6D9", "#F2C94C", "#F29B7A", "#A6D5C1", "#7B8491"]


def main():
    df = pd.read_excel(XLSX, sheet_name="benchmark")
    metrics = [m for m in METRIC_LABELS if m in df.columns]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(df)); width = 0.78 / len(metrics)
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, df[metric].astype(float).values, width=width,
               label=METRIC_LABELS[metric], color=COLORS[i % len(COLORS)], alpha=0.92)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("CH505 VRC-CH31 benchmark classification metrics")
    ymin = min(0.0, float(df[metrics].to_numpy().min())) - 0.03
    ax.set_ylim(ymin, 1.05)
    ax.legend(frameon=False, ncol=len(metrics), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "panel_benchmark_metrics.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Figure S5 ->", OUT / "panel_benchmark_metrics.pdf")


if __name__ == "__main__":
    main()
