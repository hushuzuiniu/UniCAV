#!/usr/bin/env python3
"""Figure 6 — BEAST posterior ancestral propagation (CH505 / CH235 lineage).

Self-contained single file. The plotting functions from the original BEAST
scripts are inlined verbatim below:
  17 -> safe_corr, binned_median           (panel A time trend)
  14 -> plot_anchor_ribbon, plot_mcc_vs_posterior_median  (panels D / E)
  12 -> plot_lineage + heatmap helpers     (panel C)

Panels A/D/E read source_data/Figure6.xlsx; panel C reads the two raw CH505
prediction CSVs (anc_validation + independent_test ensemble). Panel B is R
(run Figure6.R). Output -> Figure6_panels/.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "hiv_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data"
FIG_XLSX = DATA / "Figure6.xlsx"
OUT = HERE / "Figure6_panels"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Constants (from BEAST scripts 12 / 14 / 17)
# ============================================================================
CONFIDENCE_COLORS = {"high": "#0072B2", "medium": "#E69F00", "low": "#8A8A8A"}

CH235_LINEAGE_ANTIBODIES = ["UCA", "IA4", "IA3", "IA2", "IA1",
                            "CH235", "CH236", "CH239", "CH240", "CH241"]

POSTERIOR_RIBBON_COLORS = ["#0A6dAF", "#A72061", "#926AAD", "#059244", "#AC5326",
                           "#848385", "#D4E5F4", "#FBB9BA", "#F5EBF4", "#C2E3D0",
                           "#F6E4D0", "#C9C9C9"]

LINEAGE_ANTIBODIES = {
    "CH103": ["CH103", "CH104", "CH106", "CH31"],
    "CH235": ["UCA", "IA4", "IA3", "IA2", "IA1",
              "CH235", "CH236", "CH239", "CH240", "CH241", "CH31"],
}
TIMEPOINT_WEEK = {
    "Week 0": (-0.5, 2), "Week 4": (2, 10), "Week 14": (10, 17), "Week 20": (17, 25),
    "Week 30": (25, 40), "Week 53": (40, 65), "Week 78": (65, 90), "Week 100": (90, 1e6),
}
TIMEPOINT_ORDER = list(TIMEPOINT_WEEK.keys())
TIMEPOINT_COLOR = {
    "Week 0": "#D55E00", "Week 4": "#E69F00", "Week 14": "#9BC53D", "Week 20": "#009E73",
    "Week 30": "#1CA3A8", "Week 53": "#56B4E9", "Week 78": "#0072B2", "Week 100": "#6A3D9A",
}
COLOR_RESISTANT = "#E76F51"
COLOR_SENSITIVE = "#2A9D8F"
COLOR_MISSING = "#FFFFFF"


# ============================================================================
# Inlined plotting functions (verbatim from BEAST scripts 12 / 14 / 17)
# ============================================================================
def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return float("nan")
    return float(data["x"].corr(data["y"], method=method))


def binned_median(x: pd.Series, y: pd.Series, n_bins: int = 8) -> pd.DataFrame:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty:
        return pd.DataFrame(columns=["x", "y"])
    bins = np.linspace(float(data["x"].min()), float(data["x"].max()), n_bins + 1)
    if len(np.unique(bins)) < 2:
        return pd.DataFrame(columns=["x", "y"])
    data["bin"] = pd.cut(data["x"], bins=bins, include_lowest=True)
    out = data.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median"))
    return out.reset_index(drop=True)


def is_threshold_centered_summary(summary):
    if "prediction_metric" not in summary.columns or summary.empty:
        return False
    metric = str(summary["prediction_metric"].dropna().iloc[0])
    return metric.startswith("threshold_centered_score")


def plot_anchor_ribbon(summary, output_path, antibodies=None):
    """时间轨迹 ribbon: 一条线 / 一条 ribbon = 一个 antibody, x = anchor week。

    同 week 多 clade 时, 按 (antibody, week) 聚合: mean 取均值, interval 取范围并集。
    """
    if antibodies is not None:
        summary = summary[summary["antibody"].isin(antibodies)].copy()
    if summary.empty:
        return
    abs_unique = sorted(summary["antibody"].unique())
    agg = (summary.groupby(["antibody", "anchor_approx_week"], dropna=False)
           .agg(mid=("mean_p_sens", "mean"),
                lo=("ci_lo", "min"),
                hi=("ci_hi", "max"))
           .reset_index()
           .sort_values(["antibody", "anchor_approx_week"]))

    if antibodies is not None:
        abs_sorted = [ab for ab in antibodies if ab in abs_unique]
    else:
        abs_sorted = sorted(abs_unique)

    threshold_centered = is_threshold_centered_summary(summary)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    if antibodies is not None:
        palette = (
            POSTERIOR_RIBBON_COLORS
            if len(abs_sorted) <= len(POSTERIOR_RIBBON_COLORS)
            else POSTERIOR_RIBBON_COLORS * math.ceil(
                len(abs_sorted) / len(POSTERIOR_RIBBON_COLORS)
            )
        )
        colors = {ab: palette[i] for i, ab in enumerate(abs_sorted)}
    else:
        cmap = plt.colormaps.get_cmap("tab20")
        colors = {ab: cmap(i % 20) for i, ab in enumerate(abs_sorted)}
    for i, ab in enumerate(abs_sorted):
        sub = agg[agg["antibody"] == ab].sort_values("anchor_approx_week")
        if len(sub) < 2:
            continue
        color = colors[ab]
        ax.plot(sub["anchor_approx_week"], sub["mid"],
                color=color, linewidth=1.5,
                marker="o", markersize=4, label=ab)
        ax.fill_between(sub["anchor_approx_week"], sub["lo"], sub["hi"],
                        color=color, alpha=0.15)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Anchor week", fontsize=10)
    y_label = (
        "Sensitive score ± 95% interval"
        if threshold_centered else
        "Mean predicted-sensitive fraction ± 95% interval"
    )
    ax.set_ylabel(y_label, fontsize=10)
    lineage_prefix = "CH235-lineage" if antibodies is not None else "CH505"
    title_prefix = (
        f"{lineage_prefix} escape trajectory across local-clade ancestors"
        if threshold_centered else
        f"{lineage_prefix} escape trajectory across local-clade ancestors"
    )
    ax.set_title(
        f"{title_prefix}\n"
        "(posterior over BEAST tree samples, n=100)",
        fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=6.5, ncol=2,
              frameon=True, framealpha=0.82, edgecolor="none",
              title="Antibody", title_fontsize=7)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_mcc_vs_posterior_median(summary, mcc_pred_csv, output_path):
    if not mcc_pred_csv.exists():
        print(f"  warn: 缺 MCC 预测 {mcc_pred_csv}, 跳过 MCC vs posterior 散点")
        return
    mcc = pd.read_csv(mcc_pred_csv, low_memory=False)
    if "mcc_p_sens" in mcc.columns:
        mcc_score_col = "mcc_p_sens"
    elif "pred_prob_sensitive_mean" in mcc.columns:
        mcc_score_col = "pred_prob_sensitive_mean"
    else:
        return
    mcc_mean = (mcc.groupby("antibody")[mcc_score_col]
                .mean().rename("mcc_mean").reset_index())
    post_mean = (summary.groupby("antibody")["mean_p_sens"]
                 .mean().rename("posterior_mean").reset_index())
    merged = mcc_mean.merge(post_mean, on="antibody", how="inner")

    ch235_order = [
        "UCA", "IA4", "IA3", "IA2", "IA1",
        "CH235", "CH236", "CH239", "CH240", "CH241",
    ]
    merged = merged[merged["antibody"].isin(ch235_order)].copy()
    merged["antibody"] = pd.Categorical(
        merged["antibody"], categories=ch235_order, ordered=True)
    merged = merged.sort_values("antibody").reset_index(drop=True)
    n = len(merged)

    fig, ax = plt.subplots(figsize=(4.10, 3.85))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=0.7, zorder=1)

    if "focal_alpha0p88" in str(output_path):
        label_positions = {
            "UCA": (0.70, 0.56, "right"),
            "IA4": (0.69, 1.02, "right"),
            "IA3": (0.70, 0.93, "right"),
            "IA2": (0.82, 0.89, "right"),
            "IA1": (0.79, 1.08, "right"),
            "CH235": (0.88, 1.08, "center"),
            "CH236": (0.97, 0.90, "left"),
            "CH239": (0.98, 1.08, "left"),
            "CH240": (1.06, 1.02, "left"),
            "CH241": (1.06, 0.96, "left"),
        }
        xlim = (-0.03, 1.11)
        ylim = (-0.03, 1.12)
    else:
        label_positions = {
            "UCA": (0.405, 0.615, "right"),
            "IA1": (0.405, 0.555, "right"),
            "CH239": (0.415, 0.690, "right"),
            "CH240": (0.548, 0.555, "left"),
            "IA2": (0.545, 0.710, "left"),
            "IA3": (0.552, 0.650, "left"),
            "CH235": (0.520, 0.835, "right"),
            "IA4": (0.665, 0.955, "left"),
            "CH236": (0.695, 0.885, "left"),
            "CH241": (0.890, 0.930, "right"),
        }
        xlim = (-0.03, 1.05)
        ylim = (-0.03, 1.07)
    color_map = {
        "CH103": "#0A6dAF",
        "CH104": "#0A6dAF",
        "CH106": "#0A6dAF",
        "CH31": "#848385",
        "UCA": "#0A6dAF",
        "IA4": "#A72061",
        "IA3": "#926AAD",
        "IA2": "#059244",
        "IA1": "#AC5326",
        "CH235": "#848385",
        "CH236": "#D4E5F4",
        "CH239": "#FBB9BA",
        "CH240": "#F5EBF4",
        "CH241": "#C2E3D0",
    }
    for _, row in merged.iterrows():
        antibody = row["antibody"]
        ax.scatter(
            row["mcc_mean"],
            row["posterior_mean"],
            color=color_map.get(antibody, "#2A9D8F"),
            s=21,
            edgecolor="black",
            linewidth=0.25,
            zorder=3,
        )
    for _, row in merged.iterrows():
        antibody = row["antibody"]
        if antibody not in label_positions:
            continue
        x_pos, y_pos, side = label_positions[antibody]
        ax.annotate(
            antibody,
            xy=(row["mcc_mean"], row["posterior_mean"]),
            xytext=(x_pos, y_pos),
            textcoords="data",
            fontsize=7.5,
            ha=side,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#A8A8A8",
                "linewidth": 0.35,
                "alpha": 0.70,
                "shrinkA": 1.0,
                "shrinkB": 2.0,
            },
            bbox={
                "boxstyle": "round,pad=0.06",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            clip_on=False,
        )

    r_text = "NA"
    if n >= 3:
        r = np.corrcoef(merged["mcc_mean"], merged["posterior_mean"])[0, 1]
        r_text = f"{r:.3f}"

    ax.text(
        0.05,
        0.94,
        f"Pearson r = {r_text}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#CFCFCF",
            "linewidth": 0.4,
            "alpha": 0.9,
        },
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("MCC mean predicted-sensitive fraction", fontsize=8.5)
    ax.set_ylabel("Posterior mean predicted-sensitive fraction", fontsize=8.5)
    ax.set_title("CH235 lineage: MCC vs posterior mean predicted-sensitive fraction",
                 fontsize=9.5)
    ax.tick_params(width=0.6, length=3)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def normalize_tip_predictions(tips: pd.DataFrame) -> pd.DataFrame:
    """Accept either legacy 04i tables or UniCAV ensemble tables."""
    tips = tips.copy()
    if "pred_binary_sensitive" not in tips.columns:
        if "average_y_pred_label" not in tips.columns:
            raise ValueError(
                "Tip predictions need pred_binary_sensitive or average_y_pred_label"
            )
        tips["pred_binary_sensitive"] = pd.to_numeric(
            tips["average_y_pred_label"], errors="coerce"
        )
    if "binary_label" not in tips.columns:
        if "binary_label_50" in tips.columns:
            tips["binary_label"] = pd.to_numeric(
                tips["binary_label_50"], errors="coerce"
            )
        elif "y_true" in tips.columns:
            tips["binary_label"] = pd.to_numeric(tips["y_true"], errors="coerce")
        else:
            raise ValueError("Tip predictions need binary_label, binary_label_50, or y_true")
    tips["pred_binary_sensitive"] = pd.to_numeric(
        tips["pred_binary_sensitive"], errors="coerce"
    )
    tips["binary_label"] = pd.to_numeric(tips["binary_label"], errors="coerce")
    return tips


def bin_ancestral_to_timepoint(years: float) -> str | None:
    """Map beast_root_distance_years to the nearest tip-timepoint label."""
    weeks = years * 52.0
    for label, (lo, hi) in TIMEPOINT_WEEK.items():
        if lo <= weeks < hi:
            return label
    return None


def build_tip_blocks(tips: pd.DataFrame, lineage: str, antibodies: list[str]):
    """Build per-timepoint row blocks for tips (measured + predicted S/R)."""
    sub = tips[tips["antibody"].isin(antibodies) & (tips["lineage"] == lineage)].copy()

    # One row per (timepoint, virus); pivot antibody. Values are 1=S, 0=R.
    meas_pivot = sub.pivot_table(
        index=["timepoint", "virus"],
        columns="antibody",
        values="binary_label",
        aggfunc="mean",
    )
    pred_pivot = sub.pivot_table(
        index=["timepoint", "virus"],
        columns="antibody",
        values="pred_binary_sensitive",
        aggfunc="mean",
    )
    meas_pivot = meas_pivot.reindex(columns=antibodies)
    pred_pivot = pred_pivot.reindex(columns=antibodies)

    # Sort rows by timepoint order then virus
    def key(idx):
        tp, v = idx
        return (TIMEPOINT_ORDER.index(tp) if tp in TIMEPOINT_ORDER else 99, v)
    ordered = sorted(meas_pivot.index, key=key)
    meas_pivot = meas_pivot.loc[ordered]
    pred_pivot = pred_pivot.loc[ordered]

    timepoint_labels = [i[0] for i in ordered]
    virus_labels = [i[1] for i in ordered]
    return meas_pivot, pred_pivot, timepoint_labels, virus_labels


def build_ancestral_block(anc: pd.DataFrame, antibodies: list[str]):
    """Predicted S/R at BEAST internal nodes, binned to tip-timepoint brackets."""
    sub = anc[anc["antibody"].isin(antibodies)].copy()
    sub["tp_bin"] = sub["beast_root_distance_years"].map(bin_ancestral_to_timepoint)
    pred_pivot = sub.pivot_table(
        index=["tp_bin", "beast_node_id", "beast_root_distance_years"],
        columns="antibody",
        values="pred_binary_sensitive",
        aggfunc="mean",
    )
    pred_pivot = pred_pivot.reindex(columns=antibodies)

    def key(idx):
        tp, node, t = idx
        return (TIMEPOINT_ORDER.index(tp) if tp in TIMEPOINT_ORDER else 99, t)
    ordered = sorted(pred_pivot.index, key=key)
    pred_pivot = pred_pivot.loc[ordered]

    timepoint_labels = [i[0] for i in ordered]
    node_labels = [i[1] for i in ordered]
    return pred_pivot, timepoint_labels, node_labels


def draw_heatmap(ax, matrix, row_timepoints, row_labels,
                 antibodies, title, show_yticklabels=True,
                 timepoint_side=None):
    """matrix: 1=sensitive, 0=resistant, NaN=missing. timepoint_side: 'left'/'right'/None."""
    arr = np.asarray(matrix, dtype=float)
    cmap = BINARY_CMAP.copy()
    cmap.set_bad(color=COLOR_MISSING)
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)

    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    n_ab = len(antibodies)
    ax.set_xticks(np.arange(n_ab))
    ax.set_xticklabels(antibodies, rotation=90, fontsize=12)
    if show_yticklabels:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7.5)
        ax.tick_params(axis="y", pad=1, length=2)
    else:
        ax.set_yticks([])

    # Separator lines + optional timepoint strip (left or right side).
    # Strip and text positioned in points-offset from the spine so the
    # physical distance from the heatmap is the same regardless of panel
    # width (4 antibodies vs 11 antibodies).
    import matplotlib.transforms as mt
    blended = mt.blended_transform_factory(ax.transAxes, ax.transData)

    n_ab_panel = len(antibodies)
    if n_ab_panel <= 4:
        # CH103 panel (narrow): more breathing room
        left_strip_offset_pts = -55
        left_text_offset_pts = -62
    else:
        # CH235 panel (wider): just enough offset
        left_strip_offset_pts = -50
        left_text_offset_pts = -62
    right_strip_offset_pts = 10
    right_text_offset_pts = 30
    strip_w_pts = 9

    prev_tp = None
    block_start = 0
    for i, tp in enumerate(row_timepoints + [None]):
        if tp != prev_tp and prev_tp is not None:
            ax.axhline(i - 0.5, color="black", linewidth=0.6)
            if timepoint_side in ("left", "right"):
                y_mid = (block_start + i - 1) / 2
                if timepoint_side == "left":
                    base_x_ax = 0.0
                    strip_off = left_strip_offset_pts
                    text_off = left_text_offset_pts
                    ha = "right"
                else:
                    base_x_ax = 1.0
                    strip_off = right_strip_offset_pts
                    text_off = right_text_offset_pts
                    ha = "left"
                strip_trans = mt.offset_copy(
                    blended, fig=ax.figure, x=strip_off, units="points")
                text_trans = mt.offset_copy(
                    blended, fig=ax.figure, x=text_off, units="points")
                # Rectangle width also in points → use a second offset for right edge
                strip_w_trans = mt.offset_copy(
                    blended, fig=ax.figure,
                    x=strip_off + strip_w_pts, units="points")
                # Build patch using axes-x for both edges via two offsets
                # (matplotlib Rectangle takes width in current transform units;
                # workaround: draw a thin vertical rectangle with width in
                # display coords by setting width in axes-fraction tiny units
                # is messy — easier: use a Line2D-like thick vertical bar.)
                ax.plot(
                    [base_x_ax, base_x_ax],
                    [block_start - 0.5, i - 0.5],
                    transform=strip_trans,
                    color=TIMEPOINT_COLOR.get(prev_tp, "grey"),
                    linewidth=strip_w_pts,
                    solid_capstyle="butt",
                    clip_on=False,
                )
                ax.text(base_x_ax, y_mid, prev_tp, ha=ha, va="center",
                        fontsize=11, fontweight="bold",
                        color=TIMEPOINT_COLOR.get(prev_tp, "black"),
                        transform=text_trans, clip_on=False)
            block_start = i
        prev_tp = tp

    ax.set_title(title, fontsize=12, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return im


def plot_lineage(tips: pd.DataFrame, anc: pd.DataFrame, lineage: str,
                 out_pdf: Path):
    antibodies = LINEAGE_ANTIBODIES[lineage]
    meas_pv, pred_pv, tp_labels_tip, virus_labels = build_tip_blocks(
        tips, lineage, antibodies)
    anc_pv, tp_labels_anc, node_labels = build_ancestral_block(anc, antibodies)

    n_tip = meas_pv.shape[0]
    n_anc = anc_pv.shape[0]

    # Figure: 3 heatmaps side by side. Legend below the figure.
    fig_h = max(7, 0.05 * max(n_tip, n_anc) + 2.0)
    # Per-antibody column width: narrower for wide panels (CH235, 11 abs)
    per_ab_w = 0.50 if len(antibodies) <= 4 else 0.32
    fig_w = 3.0 + per_ab_w * len(antibodies) * 3
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[len(antibodies), len(antibodies), len(antibodies)],
        wspace=0.30,
    )
    ax_meas = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_anc = fig.add_subplot(gs[0, 2])

    draw_heatmap(ax_meas, meas_pv.values, tp_labels_tip, virus_labels, antibodies,
                 title=f"Measured tips\n(n={n_tip})",
                 show_yticklabels=True,
                 timepoint_side="left")
    draw_heatmap(ax_pred, pred_pv.values, tp_labels_tip, virus_labels, antibodies,
                 title=f"Predicted tips\n(n={n_tip})",
                 show_yticklabels=False,
                 timepoint_side=None)
    draw_heatmap(ax_anc, anc_pv.values, tp_labels_anc, node_labels, antibodies,
                 title=f"Predicted ancestors\n(n={n_anc})",
                 show_yticklabels=False,
                 timepoint_side="right")

    # Only include "No measurement" patch if any matrix actually has NaN
    has_missing = (
        np.isnan(meas_pv.values).any()
        or np.isnan(pred_pv.values).any()
        or np.isnan(anc_pv.values).any()
    )
    legend_handles = [
        Patch(facecolor=COLOR_SENSITIVE, edgecolor="black",
              label="Sensitive (IC50 < 50 µg/mL)"),
        Patch(facecolor=COLOR_RESISTANT, edgecolor="black",
              label="Resistant (IC50 ≥ 50 µg/mL)"),
    ]
    if has_missing:
        legend_handles.append(
            Patch(facecolor=COLOR_MISSING, edgecolor="black",
                  label="No measurement")
        )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.08, right=0.92)
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.03))

    fig.suptitle(
        f"{lineage} lineage — CH505 neutralization S/R: tips vs BEAST ancestors",
        fontsize=15, fontweight="bold", y=0.965,
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

BINARY_CMAP = mcolors.ListedColormap([COLOR_RESISTANT, COLOR_SENSITIVE])


# ============================================================================
# Panel wrappers
# ============================================================================
def panel_A() -> None:
    """A: CH235 ancestral sensitive-score vs root time (script 17 transforms)."""
    sub = pd.read_excel(FIG_XLSX, sheet_name="A_time_trend")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for conf, color in CONFIDENCE_COLORS.items():
        part = sub[sub["confidence"].astype(str) == conf]
        ax.scatter(part["root_time_weeks"], part["score"], s=14, alpha=0.60,
                   color=color, edgecolor="none", label=conf)
    med = binned_median(sub["root_time_weeks"], sub["score"])
    if not med.empty:
        ax.plot(med["x"], med["y"], color="#111111", lw=1.4)
    ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", alpha=0.85)
    r = safe_corr(sub["root_time_weeks"], sub["score"], method="pearson")
    ax.text(0.04, 0.10, f"Pearson r = {r:.2f}", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold")
    ax.set_title("CH235 Antibody", fontsize=11)
    ax.set_xlabel("Root distance (weeks)", fontsize=10)
    ax.set_ylabel("Sensitive score", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8,
              handletextpad=0.2, columnspacing=1.0)
    fig.savefig(OUT / "panel_A_time_trend.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved panel_A_time_trend (CH235, Pearson r={r:.2f})")


def panel_C() -> None:
    """C: CH235 lineage tip/ancestor S/R heatmap (script 12 plot_lineage)."""
    tips = normalize_tip_predictions(
        pd.read_csv(DATA / "independent_test_predictions_ensemble.csv"))
    anc = pd.read_csv(DATA / "anc_validation_predictions.csv")
    lineage = "CH235"
    present = [a for a in LINEAGE_ANTIBODIES[lineage] if a in tips["antibody"].unique()]
    LINEAGE_ANTIBODIES[lineage] = present
    plot_lineage(tips, anc, lineage, OUT / "panel_C_ch235_heatmap.pdf")
    print("  saved panel_C_ch235_heatmap")


def panels_DE() -> None:
    """D: escape-trajectory ribbon; E: MCC vs posterior (script 14)."""
    centered = pd.read_excel(FIG_XLSX, sheet_name="D_ribbon")
    summary = pd.read_excel(FIG_XLSX, sheet_name="E_posterior")
    plot_anchor_ribbon(centered, OUT / "panel_D_anchor_ribbon.pdf",
                       antibodies=CH235_LINEAGE_ANTIBODIES)
    print("  saved panel_D_anchor_ribbon")
    tmp = OUT / "_e_mcc.csv"
    pd.read_excel(FIG_XLSX, sheet_name="E_mcc").to_csv(tmp, index=False)
    plot_mcc_vs_posterior_median(summary, tmp, OUT / "panel_E_mcc_vs_posterior.pdf")
    tmp.unlink(missing_ok=True)
    print("  saved panel_E_mcc_vs_posterior")


if __name__ == "__main__":
    print("Figure 6 panel A"); panel_A()
    print("Figure 6 panel C"); panel_C()
    print("Figure 6 panels D/E"); panels_DE()
    print("Figure 6 (Python panels A/C/D/E) ->", OUT, "  [panel B: run Figure6.R]")
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".pdf":
            print("  ", f.name)
