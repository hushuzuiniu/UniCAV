#!/usr/bin/env python3
"""Figure S2 - Cross-validation training, per-antibody performance and feature
ablation of UniCAV. Panels A-D are ported verbatim (styling and plotting logic)
from the original scripts under original_analysis/draw/:
  A training_history_styled.py   validation metrics vs epoch, per fold
  B roc_auroc_cv_styled.py       pooled OOF + per-fold ROC
  C lineage_auc.py               out-of-fold AUC per antibody lineage
  D feature_ablation_dumbbell.py leave-one-out auxiliary-feature ablation
Data: source_data/FigureS2.xlsx. Output -> FigureS2_panels/.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

CACHE = Path(tempfile.gettempdir()) / "hiv_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
XLSX = HERE / "source_data" / "FigureS2.xlsx"
OUT = HERE / "FigureS2_panels"
OUT.mkdir(parents=True, exist_ok=True)

AXIS_COL, GRID_COL = "#1A1A1A", "#E7EBF0"
FOLD_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"]
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.4, "axes.edgecolor": AXIS_COL, "axes.labelcolor": AXIS_COL,
    "text.color": AXIS_COL, "xtick.color": AXIS_COL, "ytick.color": AXIS_COL,
    "legend.frameon": False, "figure.dpi": 150,
})


def _save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# ============================================================ A: training history
def panel_A():
    METRICS = [("test_roc_auc", "ROC-AUC"), ("test_f1_macro", "F1 (macro)"),
               ("test_mcc", "MCC"), ("test_accuracy", "Accuracy")]
    d = pd.read_excel(XLSX, sheet_name="A_training")
    folds = sorted(d["fold"].dropna().unique())
    emax = float(d["epoch"].max())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(210 * MM, 62 * MM))
    for ax, (col, title) in zip(axes, METRICS):
        for fi, f in enumerate(folds):
            s = d[d["fold"] == f].sort_values("epoch")
            ax.plot(s["epoch"], s[col], color=FOLD_COLORS[fi % len(FOLD_COLORS)], lw=0.9, alpha=0.9, zorder=3)
        finals = [d[d["fold"] == f].sort_values("epoch")[col].iloc[-1] for f in folds]
        mu = float(np.mean(finals))
        ax.axhline(mu, color="#9AA0A6", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(emax, mu, f" {mu:.2f}", color="#6B7280", fontsize=5.6, ha="right", va="bottom")
        ax.set_title(title, fontsize=8, pad=3); ax.set_xlabel("Epoch")
        ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_xlim(0.5, emax + 0.5); ax.set_xticks(np.arange(0, emax + 1, 10))
        ax.tick_params(length=2.3, width=0.4)
    axes[0].set_ylabel("Score")
    handles = [Line2D([0], [0], color=FOLD_COLORS[i % len(FOLD_COLORS)], lw=1.2, label=f"Fold {int(f)}")
               for i, f in enumerate(folds)]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=5.6, handlelength=1.3,
                    labelspacing=0.25, borderaxespad=0.3)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.20, wspace=0.30)
    _save(fig, "panelA_training.pdf")


# ============================================================ B: CV ROC
def panel_B():
    HERO, CHANCE = "#B2182B", "#B6BCC6"
    FOLDS = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]; POOLED = "Pooled OOF"
    su = pd.read_excel(XLSX, sheet_name="B_roc_auroc")
    auroc = {r["curve"]: float(r["AUROC"]) for _, r in su.iterrows()}
    co = pd.read_excel(XLSX, sheet_name="B_roc_coords")
    fold_aucs = np.array([auroc[f] for f in FOLDS]); mu, sd = fold_aucs.mean(), fold_aucs.std(ddof=1)
    fig, ax = plt.subplots(figsize=(74 * MM, 74 * MM))
    ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.6, color=CHANCE, zorder=1)
    for i, f in enumerate(FOLDS):
        dd = co[co["curve"] == f].sort_values("fpr")
        ax.plot(dd["fpr"], dd["tpr"], color=FOLD_COLORS[i], lw=0.9, alpha=0.9, zorder=2 + i)
    dp = co[co["curve"] == POOLED].sort_values("fpr")
    ax.plot(dp["fpr"], dp["tpr"], color=HERO, lw=1.9, zorder=8, solid_capstyle="round")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(length=2.5, width=0.35)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Cross-validation ROC (group 5-fold)", fontsize=8, pad=4)
    handles = [Line2D([0], [0], color=CHANCE, lw=0.8, ls=(0, (4, 3)), label="Chance   0.500")]
    handles += [Line2D([0], [0], color=FOLD_COLORS[i], lw=1.1, label=f"{f}   {auroc[f]:.3f}") for i, f in enumerate(FOLDS)]
    handles.append(Line2D([0], [0], color=HERO, lw=1.9, label=f"Pooled OOF   {auroc[POOLED]:.3f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=5.9, handlelength=1.4, labelspacing=0.32,
              borderaxespad=0.5, handletextpad=0.6, title=f"AUROC   (mean {mu:.3f} ± {sd:.3f})", title_fontsize=6.0)
    fig.tight_layout(); _save(fig, "panelB_cv_roc.pdf")


# ============================================================ C: per-lineage AUC
TYPE_ORDER = ["CD4bs", "V1V2", "V3", "MPER", "fusion_peptide", "interface", "CD4i",
              "silent_face", "glycan_other", "multi", "other", "unknown"]
TYPE_BG = {"CD4bs": "#DDEAF4", "V1V2": "#E4F1DF", "V3": "#FFF0D7", "MPER": "#F0E6F6",
           "fusion_peptide": "#FBE3E0", "interface": "#E3F2F2", "CD4i": "#EEEAE3",
           "silent_face": "#F3E6EF", "glycan_other": "#E9EEF8", "multi": "#F1F1D8",
           "other": "#ECECEC", "unknown": "#F6F6F6"}
TYPE_COLOR = {"CD4bs": "#4C78A8", "V1V2": "#59A14F", "V3": "#E8A33D", "MPER": "#8C6BB1",
              "fusion_peptide": "#C0504D", "interface": "#3C9D9B", "CD4i": "#A6884F",
              "silent_face": "#B07AA1", "glycan_other": "#7F9BC0", "multi": "#BAB23C",
              "other": "#9AA0A6", "unknown": "#BFC4CA"}
OVERALL_AUC, N_LIN_TOTAL = 0.913991, 62   # original_analysis/draw/source_data_lineage_auc_overall.txt


def panel_C():
    df = pd.read_excel(XLSX, sheet_name="C_lineage_auc")
    lin_type = df.groupby("lineage")["type"].agg(lambda s: s.value_counts().index[0]).to_dict()
    type_rank = {t: i for i, t in enumerate(TYPE_ORDER)}
    order = sorted(lin_type, key=lambda lin: (type_rank.get(lin_type[lin], len(TYPE_ORDER)), lin_type[lin], lin))
    data = [df.loc[df.lineage == lin, "auc"].to_numpy() for lin in order]
    colors = [TYPE_COLOR.get(lin_type[lin], "#9AA0A6") for lin in order]
    x = np.arange(len(order))
    fig = plt.figure(figsize=(260 * MM, 100 * MM)); ax = fig.add_subplot(111)
    ax.axhline(OVERALL_AUC, color="#B0392B", lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(len(order) - 0.5, OVERALL_AUC, f" overall {OVERALL_AUC:.3f}", color="#B0392B",
            fontsize=6.0, va="bottom", ha="right", zorder=5)
    spans = []; start = 0
    while start < len(order):
        typ = lin_type[order[start]]; end = start
        while end + 1 < len(order) and lin_type[order[end + 1]] == typ:
            end += 1
        spans.append((typ, start, end))
        ax.axvspan(start - 0.5, end + 0.5, color=TYPE_BG.get(typ, "#F2F2F2"), alpha=0.50, lw=0, zorder=0)
        start = end + 1
    bp = ax.boxplot(data, positions=x, widths=0.62, patch_artist=True, showfliers=False, zorder=2,
                    medianprops=dict(color="#222", lw=1.0), whiskerprops=dict(color="#555", lw=0.8),
                    capprops=dict(color="#555", lw=0.8), boxprops=dict(lw=0.7, edgecolor="#444"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lin}  (med {np.median(v):.2f})" for lin, v in zip(order, data)],
                       rotation=90, fontsize=4.4, ha="center")
    ax.tick_params(axis="x", length=2, width=0.5); ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Antibody lineage (grouped by epitope type)", labelpad=2)
    ax.set_ylim(0.40, 1.005); ax.set_yticks(np.arange(0.40, 1.0001, 0.1))
    ax.set_ylabel("Out-of-fold test ROC-AUC")
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=2.5)
    ax.set_title(f"Per-lineage neutralization-prediction AUC (all {len(order)} bNAb lineages; grouped by antibody type)",
                 fontsize=8.0, fontweight="bold", pad=18)
    tax = ax.secondary_xaxis("top"); tax.set_xlim(ax.get_xlim())
    tax.set_xticks([(s + e) / 2 for _, s, e in spans])
    tax.set_xticklabels([t for t, _, _ in spans], rotation=45, ha="left", fontsize=5.6, color="#444444")
    tax.tick_params(axis="x", length=0, pad=1)
    for sp in tax.spines.values():
        sp.set_visible(False)
    used = list(dict.fromkeys(lin_type[lin] for lin in order))
    handles = [Patch(facecolor=TYPE_COLOR.get(t, "#9AA0A6"), edgecolor="#444", lw=0.4, label=t) for t in used]
    ax.legend(handles=handles, loc="lower left", ncol=len(used), fontsize=5.6, handlelength=1.0,
              handleheight=1.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3,
              title="Epitope type", title_fontsize=6.2)
    _save(fig, "panelC_lineage_auc.pdf")


# ============================================================ D: feature ablation
def panel_D():
    METRICS = ["ACC", "PRE", "REC", "F1-Score"]; FULL = "feat_all"
    PANELS = [("feat_no_env_aux", "Env aux($-$)", "#5B8FC9"),
              ("feat_no_epitope_pos", "Epitope pos($-$)", "#E07B91"),
              ("feat_no_epitope_token", "Epitope token($-$)", "#6FB07A"),
              ("feat_none", "Embedding only", "#E1A04B")]
    REF_LBL = "Full feature set"
    VAR_LBL = {"feat_no_env_aux": "no Env aux", "feat_no_epitope_pos": "no Epitope pos",
               "feat_no_epitope_token": "no Epitope token", "feat_none": "Embedding only"}
    GREY, GREY_EDGE = "#8C8C8C", "#5A5A5A"
    df = pd.read_excel(XLSX, sheet_name="D_feature_ablation")
    full = {m: df[(df.config == FULL) & (df.metric == m)]["value"].iloc[0] for m in METRICS}
    lo, hi = df["value"].min(), df["value"].max(); pad = 0.012
    xlim = (np.floor((lo - pad) / 0.025) * 0.025, np.ceil((hi + pad) / 0.025) * 0.025)
    yv = np.arange(len(METRICS))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 100 * MM), sharex=True)
    for ax, (cfg, title, col) in zip(axes.ravel(), PANELS):
        sub = {m: df[(df.config == cfg) & (df.metric == m)]["value"].iloc[0] for m in METRICS}
        for y, m in zip(yv, METRICS):
            vf, va = full[m], sub[m]
            ax.plot([va, vf], [y, y], color="#C4C4C4", lw=1.0, zorder=1)
            ax.scatter([vf], [y], s=160, color=GREY, alpha=0.18, lw=0, zorder=2)
            ax.scatter([va], [y], s=160, color=col, alpha=0.20, lw=0, zorder=2)
            ax.scatter([va], [y], s=34, color=col, edgecolor="white", lw=0.5, zorder=3)
            ax.scatter([vf], [y], s=44, color=GREY, edgecolor=GREY_EDGE, lw=0.7, zorder=4)
        ax.set_yticks(yv); ax.set_yticklabels(METRICS); ax.set_ylim(-0.6, len(METRICS) - 0.4)
        ax.set_xlim(*xlim); ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.025))
        ax.grid(axis="x", color=GRID_COL, lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.4); ax.set_title(title, fontsize=8, pad=4)
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor=GREY_EDGE,
                          markersize=5.2, label=REF_LBL),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor=col, markeredgecolor="white",
                          markersize=4.8, label=VAR_LBL[cfg])]
        ax.legend(handles=handles, loc="lower left", fontsize=5.8, handletextpad=0.3,
                  labelspacing=0.3, borderaxespad=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Score (pooled out-of-fold, threshold 0.5, n = 69,083)")
    fig.tight_layout(); _save(fig, "panelD_feature_ablation.pdf")


if __name__ == "__main__":
    panel_A(); panel_B(); panel_C(); panel_D()
    print("Figure S2 panels ->", OUT)
    for f in sorted(OUT.glob("*.pdf")):
        print("  ", f.name)
