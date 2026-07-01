#!/usr/bin/env python3
"""Figure 4 panel D — CH505 regression (predicted vs measured log10 IC50).

Reads ``Figure4.xlsx`` sheet ``reg_ch505``. The other Figure 4 panels are R
(run ``Figure4.R``): B confusion, C metric summary, E per-antibody heatmap,
F longitudinal sensitive fraction, G BEAST MCC tree. Panel A is the schematic
(kept in the Illustrator file). Output -> Figure4_panels/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
FIG_XLSX = HERE / "source_data" / "Figure4.xlsx"
OUT = HERE / "Figure4_panels"
OUT.mkdir(parents=True, exist_ok=True)
AXIS_COL = "#1A1A1A"; GRID_COL = "#E7EBF0"; MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.35, "axes.edgecolor": AXIS_COL, "axes.labelcolor": AXIS_COL,
    "text.color": AXIS_COL, "xtick.color": AXIS_COL, "ytick.color": AXIS_COL,
    "legend.frameon": False, "figure.dpi": 150,
})


def panel_regression_ch505():
    raw = pd.read_excel(FIG_XLSX, sheet_name="reg_ch505", header=None)
    hdr = raw.index[raw[0] == "Prediction values"][0] + 1
    cols = raw.iloc[hdr].tolist(); d = raw.iloc[hdr + 1:].copy(); d.columns = cols
    d = d.dropna(subset=["y_true_real", "y_pred_real"]).astype({"y_true_real": float, "y_pred_real": float})
    yt = d["y_true_real"].to_numpy(); yp = d["y_pred_real"].to_numpy()
    resid = yp - yt; r = float(np.corrcoef(yt, yp)[0, 1]); mae = float(np.mean(np.abs(resid))); n = len(yt)
    lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max()); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
    cmap = LinearSegmentedColormap.from_list("blues_proj", ["#EAF2FA", "#D3E2F2", "#7FB0D8", "#0A6DAF", "#06436C"])
    fig, ax = plt.subplots(figsize=(62 * MM, 62 * MM))
    ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.7, color="#ACB4BF", zorder=1, label="y = x")
    try:
        from scipy.stats import gaussian_kde
        dens = gaussian_kde(np.vstack([yt, yp]))(np.vstack([yt, yp])); order = np.argsort(dens)
        ax.scatter(yt[order], yp[order], c=dens[order], s=8, cmap=cmap, linewidths=0.0, alpha=0.9, zorder=2)
    except Exception:
        ax.scatter(yt, yp, s=8, color="#0A6DAF", linewidths=0.0, alpha=0.6, zorder=2)
    slope, intercept = np.polyfit(yt, yp, 1); xs = np.array([lo, hi])
    ax.plot(xs, slope * xs + intercept, lw=1.2, color="#DB6C50", zorder=4, label="Linear fit")
    ax.text(0.04, 0.97, f"n = {n:,}\nPearson r = {r:.3f}\nMAE = {mae:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, linespacing=1.45)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured log₁₀ IC₅₀"); ax.set_ylabel("Predicted log₁₀ IC₅₀")
    ax.xaxis.set_major_locator(MultipleLocator(1.0)); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f")); ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(length=2.5, width=0.35)
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
    ax.set_title("CH505 independent regression", fontsize=8, pad=4)
    fig.tight_layout()
    fig.savefig(OUT / "panel_D_regression_ch505.pdf", bbox_inches="tight")
    plt.close(fig); print("saved panel_D_regression_ch505")


if __name__ == "__main__":
    panel_regression_ch505()
    print("Figure 4 (panel D) ->", OUT)
