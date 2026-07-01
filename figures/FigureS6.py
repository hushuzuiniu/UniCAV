#!/usr/bin/env python3
"""Figure S6 panel D - CD4bs selection-site mutations linked to CH235 ancestral
escape. Ported from BEAST/20_plot_cd4bs_selection_escape_link.py (plot_figure,
left + right subplots): CD4bs selected-site mutation fraction vs root distance,
and predicted CH235 sensitivity vs mutation fraction. Panels A-C are R (FigureS6.R).
Data: source_data/FigureS6.xlsx (D_nodes/D_bin/D_firsttimes/D_stats).
Output -> FigureS6_panels/.
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
XLSX = HERE / "source_data" / "FigureS6.xlsx"
OUT = HERE / "FigureS6_panels"
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 7, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "axes.spines.right": False, "axes.spines.top": False})

PREFERRED_TIME_PALETTE = {"TF": "#d73027", "w4": "#FFF6F6", "w14": "#F6FFDC", "w20": "#ebe6ba",
                          "w22": "#e8c69f", "w30": "#dc8274", "w53": "#c36075", "w78": "#9e4e7a",
                          "w100": "#7f3f7a", "w136": "#483063", "w160": "#1e1030"}
TIMEPOINT_WEEKS = [("TF", 0.0), ("w4", 4.0), ("w14", 14.0), ("w20", 20.0), ("w22", 22.0),
                   ("w30", 30.0), ("w53", 53.0), ("w78", 78.0), ("w100", 100.0), ("w136", 136.0), ("w160", 160.0)]


def main():
    node_df = pd.read_excel(XLSX, sheet_name="D_nodes")
    bin_df = pd.read_excel(XLSX, sheet_name="D_bin")
    first_times = pd.read_excel(XLSX, sheet_name="D_firsttimes")
    stats = pd.read_excel(XLSX, sheet_name="D_stats").iloc[0]
    r_time = node_df[["root_distance_weeks", "mutation_fraction"]].corr().iloc[0, 1]

    fig, (ax_mut, ax_scatter) = plt.subplots(1, 2, figsize=(5.3, 2.85),
                                             gridspec_kw={"width_ratios": [1.15, 0.95], "wspace": 0.42})
    # --- left: mutation fraction vs root distance ---
    ax_mut.scatter(node_df["root_distance_weeks"], node_df["mutation_fraction"], s=10,
                   color=node_df["nearest_timepoint"].map(PREFERRED_TIME_PALETTE),
                   alpha=0.58, edgecolor="#39434D", linewidth=0.12)
    ax_mut.plot(bin_df["root_distance_weeks"], bin_df["mutation_fraction_mean"], color="#247C7B", lw=1.25)
    ax_mut.set_xlabel("Root distance (weeks)")
    ax_mut.set_ylabel("CD4bs selected-site\nmutation fraction")
    ax_mut.set_ylim(-0.06, 0.88)
    ax_mut.text(0.02, 0.12, f"Pearson r = {r_time:.3f}", transform=ax_mut.transAxes, fontsize=6.5, color="#247C7B")
    label_y = [0.80, 0.68, 0.56, 0.44, 0.32]
    ft = first_times.dropna(subset=["first_mutated_root_distance_weeks"]).reset_index(drop=True)
    for idx, row in ft.iterrows():
        x = float(row["first_mutated_root_distance_weeks"]); y = label_y[idx % len(label_y)]
        ax_mut.axvline(x, color="#B66A32", lw=0.75, ls=(0, (2, 2)), alpha=0.55)
        ax_mut.text(x + 1.5, y, f"HXB2 {int(row['hxb2_position'])}", color="#8A4B24", fontsize=5.8, va="center", ha="left")
    # --- right: CH235 score vs mutation fraction ---
    for label, _w in TIMEPOINT_WEEKS:
        part = node_df[node_df["nearest_timepoint"].eq(label)]
        ax_scatter.scatter(part["mutation_fraction"], part["predicted_sensitive_score"],
                           color=PREFERRED_TIME_PALETTE[label], s=16, alpha=0.76, edgecolor="#39434D", linewidth=0.18)
    valid = node_df.dropna(subset=["mutation_fraction", "predicted_sensitive_score"])
    if valid["mutation_fraction"].nunique() > 1:
        x = valid["mutation_fraction"].to_numpy(float); y = valid["predicted_sensitive_score"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax_scatter.plot(xs, intercept + slope * xs, color="#111111", lw=1.0)
    ax_scatter.set_xlabel("CD4bs selected-site mutation fraction")
    ax_scatter.set_ylabel("CH235 sensitive score")
    ax_scatter.set_xlim(-0.05, 0.82); ax_scatter.set_ylim(0.15, 0.92)
    bb = {"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.1}
    ax_scatter.text(0.05, 0.92, f"Pearson r = {float(stats['pearson_r_mutation_vs_score']):.3f}",
                    transform=ax_scatter.transAxes, fontsize=6.7, bbox=bb)
    ax_scatter.text(0.05, 0.84, f"Partial r = {float(stats['partial_r_control_root_distance']):.3f}",
                    transform=ax_scatter.transAxes, fontsize=6.3, color="#4B5563", bbox=bb)
    ax_scatter.text(0.05, 0.76, f"n = {len(valid)} nodes", transform=ax_scatter.transAxes,
                    fontsize=6.3, color="#4B5563", bbox=bb)
    handles = [Line2D([0], [0], marker="o", lw=0, color="#39434D", markerfacecolor=PREFERRED_TIME_PALETTE[l],
                      markeredgewidth=0.25, markersize=4.1, label=l)
               for l, _ in TIMEPOINT_WEEKS if l in set(node_df["nearest_timepoint"])]
    handles += [Line2D([0], [0], color="#247C7B", lw=1.1, label="Mutation trend"),
                Line2D([0], [0], color="#111111", lw=1.0, label="Linear fit")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=5.6,
               handlelength=1.2, columnspacing=0.72, labelspacing=0.25, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.28, top=0.95)
    fig.savefig(OUT / "panelD_cd4bs_escape_link.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Figure S6 panel D ->", OUT / "panelD_cd4bs_escape_link.pdf")


if __name__ == "__main__":
    main()
