#!/usr/bin/env python3
"""Figure S7 - CH235-lineage posterior anchor MRCA forest plot.

Data: source_data/FigureS7.xlsx (sheet 'anchor_posterior').
Output -> FigureS7_panels/panel_A_posterior_anchor_forest.pdf.
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
XLSX = HERE / "source_data" / "FigureS7.xlsx"
OUT = HERE / "FigureS7_panels"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    }
)

ANTIBODY_ORDER = [
    "UCA",
    "IA4",
    "IA3",
    "IA2",
    "IA1",
    "CH235",
    "CH236",
    "CH239",
    "CH240",
    "CH241",
]
REQUIRED_COLUMNS = {
    "anchor_name",
    "anchor_short",
    "facet_label",
    "anchor_approx_week",
    "antibody",
    "mean_p_sens",
    "ci_lo",
    "ci_hi",
}
POINT_COLOR = "#254653"
INTERVAL_COLOR = "#8B98A1"


def load_anchor_posterior(xlsx_path: Path = XLSX) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="anchor_posterior")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"FigureS7.xlsx is missing columns: {sorted(missing)}")

    df = df.copy()
    numeric_cols = ["anchor_approx_week", "mean_p_sens", "ci_lo", "ci_hi"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["anchor_short", "antibody", "mean_p_sens", "ci_lo", "ci_hi"])
    df = df[df["antibody"].isin(ANTIBODY_ORDER)].copy()
    df["antibody"] = pd.Categorical(df["antibody"], categories=ANTIBODY_ORDER, ordered=True)
    return df


def ordered_anchors(df: pd.DataFrame) -> pd.DataFrame:
    anchors = (
        df[["anchor_short", "facet_label", "anchor_approx_week"]]
        .drop_duplicates()
        .sort_values(["anchor_approx_week", "anchor_short"])
        .reset_index(drop=True)
    )
    return anchors


def plot_anchor_forest(df: pd.DataFrame, output_path: Path) -> None:
    anchors = ordered_anchors(df)
    if anchors.empty:
        raise ValueError("No anchor rows available for Figure S7")

    fig, axes = plt.subplots(
        1,
        len(anchors),
        figsize=(11.05, 6.30),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    if len(anchors) == 1:
        axes = [axes]

    y_positions = np.arange(len(ANTIBODY_ORDER))
    y_lookup = dict(zip(ANTIBODY_ORDER, y_positions))

    for ax, anchor in zip(axes, anchors.itertuples(index=False)):
        sub = (
            df[df["anchor_short"].eq(anchor.anchor_short)]
            .sort_values("antibody")
            .reset_index(drop=True)
        )
        y = sub["antibody"].map(y_lookup).astype(float).to_numpy()
        x = sub["mean_p_sens"].astype(float).to_numpy()
        lo = sub["ci_lo"].astype(float).to_numpy()
        hi = sub["ci_hi"].astype(float).to_numpy()
        xerr = np.vstack([np.clip(x - lo, 0, None), np.clip(hi - x, 0, None)])

        ax.axvline(0.5, color="#A8A8A8", linestyle=(0, (3, 2)), linewidth=0.55, zorder=1)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            markersize=3.9,
            markerfacecolor=POINT_COLOR,
            markeredgecolor=POINT_COLOR,
            ecolor=INTERVAL_COLOR,
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )
        ax.set_title(str(anchor.facet_label), fontsize=8.6, pad=7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(len(ANTIBODY_ORDER) - 0.55, -0.45)
        ax.tick_params(axis="x", labelsize=8, width=0.8, length=4, pad=2)
        ax.tick_params(axis="y", labelsize=9, width=0.8, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(ANTIBODY_ORDER)
    axes[0].set_ylabel("Antibody", fontsize=11)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.suptitle(
        "CH235-lineage posterior anchor MRCA predictions across BEAST tree samples (n=100)",
        fontsize=12.8,
        fontweight="bold",
        y=0.965,
    )
    fig.supxlabel("Sensitive score \u00b1 95% interval", fontsize=11, y=0.055)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.135, top=0.835)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    df = load_anchor_posterior()
    output_path = OUT / "panel_A_posterior_anchor_forest.pdf"
    plot_anchor_forest(df, output_path)
    print("Figure S7 ->", output_path)


if __name__ == "__main__":
    main()
