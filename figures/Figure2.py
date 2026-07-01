#!/usr/bin/env python3
"""Figure 2 — UniCAV performance, benchmark and ablations (consolidated).

One runnable script for all Figure 2 panels (A-J). Every panel reads its source
data from the bundled workbook ``Figure2.xlsx`` (one sheet per source), so no
model run directory is needed. Panels are written to ``Figure2_panels/`` as PDF.

  A panel_metrics          CV held-out test metric summary (Fig04_training + pooled)
  B panel_confusion        CV pooled confusion matrix
  C panel_group            split-strategy generalisation (cold virus / antibody)
  D panel_plm              Env encoder ablation
  E panel_module           architecture-module SHAP ablation
  F panel_arch             architecture ablation ROC
  G panel_baseline_values  benchmark vs bNAb-ReP/LBUM/BRAVE (per-antibody)
  H panel_baseline_ranks   benchmark top-1 wins
  I panel_regression       regression predicted vs measured log10 IC50 (CV)
  J panel_error_tolerance  regression error-tolerance curve (CV vs CH505)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
FIG_XLSX = HERE / "source_data" / "Figure2.xlsx"
OUT = HERE / "Figure2_panels"
OUT.mkdir(parents=True, exist_ok=True)

AXIS_COL = "#1A1A1A"
GRID_COL = "#E7EBF0"
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.35, "axes.edgecolor": AXIS_COL,
    "axes.labelcolor": AXIS_COL, "text.color": AXIS_COL,
    "xtick.color": AXIS_COL, "ytick.color": AXIS_COL,
    "legend.frameon": False, "figure.dpi": 150,
})

# Excel truncates sheet names to 31 chars
SH_CONF_METRIC = "confusion_matrix_cv_test_metric"
SH_MODULE = "source_data_shap_module_ablatio"


def _raw(sheet):
    return pd.read_excel(FIG_XLSX, sheet_name=sheet, header=None)


def _table(sheet):
    return pd.read_excel(FIG_XLSX, sheet_name=sheet)


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


FOLDS = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
POOLED = "Pooled OOF"


# ----------------------------------------------------- A: metric summary
def panel_metrics():
    raw = _raw("Fig04_training")
    h = raw.index[raw[0] == "fold"][0]
    cols = raw.iloc[h].tolist(); d = raw.iloc[h + 1:].copy(); d.columns = cols
    d = d.dropna(subset=["fold", "epoch"])
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    last = d.sort_values("epoch").groupby("fold").tail(1).sort_values("fold")
    r1 = _raw("Fig01_CV_ROC"); h1 = r1.index[r1[0] == "curve"][0]
    c1 = r1.iloc[h1].tolist(); d1 = r1.iloc[h1 + 1:].copy(); d1.columns = c1
    auroc = float(d1.loc[d1["curve"] == POOLED, "AUROC"].iloc[0])
    r2 = _raw("Fig02_CV_PR"); h2 = r2.index[r2[0] == "curve"][0]
    c2 = r2.iloc[h2].tolist(); d2 = r2.iloc[h2 + 1:].copy(); d2.columns = c2
    auprc = float(pd.to_numeric(d2.loc[d2["curve"] == POOLED, "auprc"], errors="coerce").iloc[0])
    mm = _table(SH_CONF_METRIC).iloc[0]
    cm = _table("confusion_matrix_cv_test")
    g = {(t, p): int(c) for t, p, c in zip(cm.true_label, cm.predicted_label, cm["count"])}
    TP = g[("Sensitive", "Sensitive")]; TN = g[("Resistant", "Resistant")]
    FP = g[("Resistant", "Sensitive")]; FN = g[("Sensitive", "Resistant")]
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    pooled = {"AUROC": auroc, "AUPRC": auprc, "Accuracy": float(mm["accuracy"]),
              "F1": float(mm["f1"]), "Recall": float(mm["recall"]), "MCC": mcc}
    METR = [("test_roc_auc", "AUROC"), ("test_avg_precision", "AUPRC"),
            ("test_accuracy", "Accuracy"), ("test_f1_binary", "F1"),
            ("test_recall", "Recall"), ("test_mcc", "MCC")]
    keys = [k for k, _ in METR]; labels = [l for _, l in METR]
    means = [last[k].mean() for k in keys]; folds = {k: last[k].to_numpy() for k in keys}
    pvals = [pooled[l] for l in labels]
    BAR = ["#0A6DAF", "#3C9D9B", "#6FB07A", "#C49A48", "#DB6C50", "#9B7FB5"]
    x = np.arange(len(METR)); fig, ax = plt.subplots(figsize=(72 * MM, 62 * MM))
    ax.grid(axis="y", color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.bar(x, means, width=0.66, color=[mpl.colors.to_rgba(c, 0.55) for c in BAR],
           edgecolor=BAR, linewidth=0.7, zorder=2)
    rng = np.random.default_rng(0)
    for xi, k in zip(x, keys):
        jit = (rng.random(len(folds[k])) - 0.5) * 0.26
        ax.scatter(xi + jit, folds[k], s=4.5, color="#23303A", edgecolor="white", linewidth=0.2, alpha=0.95, zorder=4)
    ax.scatter(x, pvals, marker="D", s=11, color="#B2182B", edgecolor="white", linewidth=0.4, zorder=6)
    for xi, pv in zip(x, pvals):
        ax.text(xi, pv + 0.018, f"{pv:.3f}", ha="center", va="bottom", fontsize=6.0, color=AXIS_COL)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=6.4)
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score (5-fold CV held-out test)"); ax.tick_params(length=2.5, width=0.35)
    ax.set_title("UniCAV cross-validation test performance", fontsize=8, pad=4)
    h_ = [Line2D([0], [0], marker="D", color="none", markerfacecolor="#B2182B", markeredgecolor="white", markersize=3.6, label="Pooled OOF"),
          Line2D([0], [0], marker="o", color="none", markerfacecolor="#23303A", markeredgecolor="white", markersize=3.0, label="Per-fold")]
    ax.legend(handles=h_, loc="lower left", fontsize=5.8, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3)
    fig.tight_layout(); _save(fig, "panel_A_metrics")


# ----------------------------------------------------- B: confusion
def panel_confusion():
    cm = _table("confusion_matrix_cv_test")
    g = {(t, p): int(c) for t, p, c in zip(cm.true_label, cm.predicted_label, cm["count"])}
    total = sum(g.values())
    grid = [[g[("Resistant", "Resistant")], g[("Resistant", "Sensitive")]],
            [g[("Sensitive", "Resistant")], g[("Sensitive", "Sensitive")]]]
    vmax = max(max(r) for r in grid); norm = Normalize(0, vmax)
    cmap = LinearSegmentedColormap.from_list("slate", ["#E7ECF4", "#C3CFE2", "#9DAECB", "#6B7DA8", "#4F608F"])
    fig, ax = plt.subplots(figsize=(58 * MM, 58 * MM)); gap = 0.04
    for r in range(2):
        for c in range(2):
            val = grid[r][c]; x, y = c, 1 - r; col = cmap(norm(val))
            ax.add_patch(Rectangle((x + gap / 2, y + gap / 2), 1 - gap, 1 - gap, facecolor=col, edgecolor="none"))
            tc = "white" if norm(val) > 0.6 else "#243352"
            ax.text(x + 0.5, y + 0.56, f"{val:,}", ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
            ax.text(x + 0.5, y + 0.40, f"{100 * val / total:.1f}%", ha="center", va="center", fontsize=8.5, fontweight="bold", color=tc)
    ax.text(-0.10, 1.5, "True\nResistant", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(-0.10, 0.5, "True\nSensitive", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(0.5, -0.10, "Pred\nResistant", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(1.5, -0.10, "Pred\nSensitive", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); _save(fig, "panel_B_confusion")


# ----------------------------------------------------- C: split ablation
def panel_group():
    df = _table("source_data_group_ablation")
    METRICS = [("AUC", "AUROC"), ("Accuracy", "Accuracy"), ("F1", "F1"), ("MCC", "MCC")]
    MCOLOR = {"AUROC": "#0A6DAF", "Accuracy": "#6FB07A", "F1": "#C49A48", "MCC": "#9B7FB5"}
    SPLIT_ORDER = [s for s in ["group_random", "group_virus_id", "group_ab_id"] if (df.split == s).any()]
    SPLIT_LABEL = {"group_random": "Random (i.i.d.)", "group_virus_id": "Cold virus", "group_ab_id": "Cold antibody"}

    def get(split, metric, eval_set, col):
        sub = df[(df.eval_set == eval_set) & (df.split == split) & (df.metric == metric)]
        return float(sub[col].iloc[0])
    dx = 0.19; offs = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * dx; block = np.arange(len(SPLIT_ORDER))
    fig, ax = plt.subplots(figsize=(64 * MM, 70 * MM))
    ax.grid(axis="y", color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for i in block:
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="#F6F8FA", zorder=0)
    for i, split in enumerate(SPLIT_ORDER):
        for j, (mkey, disp) in enumerate(METRICS):
            xp = i + offs[j]; c = MCOLOR[disp]
            cv_m = get(split, mkey, "cv_test", "mean"); ex_m = get(split, mkey, "ch505", "ensemble")
            ax.plot([xp, xp], [ex_m, cv_m], color=c, lw=1.3, alpha=0.5, solid_capstyle="round", zorder=2)
            ax.plot(xp, ex_m, "o", ms=3.6, mfc="white", mec=c, mew=1.0, zorder=4)
            ax.plot(xp, cv_m, "o", ms=3.6, mfc=c, mec="white", mew=0.5, zorder=5)
    ax.set_xticks(block); ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLIT_ORDER])
    ax.set_xlim(-0.5, len(SPLIT_ORDER) - 0.5); ax.set_ylim(0.30, 1.00); ax.set_yticks(np.arange(0.3, 1.001, 0.1))
    ax.tick_params(length=2.5, width=0.35); ax.set_ylabel("Test score")
    ax.set_title("Internal vs external generalisation", fontsize=8, pad=4)
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=MCOLOR[disp], markeredgecolor="white", markersize=4, label=disp) for _, disp in METRICS]
    h += [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555", markeredgecolor="white", markersize=4, label="Internal CV"),
          Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555", markeredgewidth=1.0, markersize=4, label="External CH505")]
    ax.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), handlelength=1.0, columnspacing=1.2, handletextpad=0.3, fontsize=5.8)
    fig.tight_layout(); _save(fig, "panel_C_group")


# ----------------------------------------------------- D: PLM ablation
def panel_plm():
    df = _table("source_data_plm_ablation")
    env_df = df[(df.vh == "Antiberta2-H") & (df.vl == "Antiberta2-L")]
    METRICS = [("AUC", "AUROC"), ("Accuracy", "Accuracy"), ("F1", "F1"), ("MCC", "MCC")]
    MCOLOR = {"AUROC": "#0A6DAF", "Accuracy": "#6FB07A", "F1": "#C49A48", "MCC": "#9B7FB5"}
    ENV_ORDER = [e for e in ["ESM1b-650M", "ESM2-8M", "ESM2-35M", "ESM2-150M", "ESM2-650M", "ESM2-3B", "ESMC-300M", "ESMC-600M", "ESMC-6B"] if (env_df.env == e).any()]
    SELECTED = "ESMC-600M"
    ENV_FAMILY = [(0, 0, "ESM-1b", "#FFFFFF"), (1, 5, "ESM-2", "#F2F5F9"), (6, len(ENV_ORDER) - 1, "ESM-C", "#E3EDF6")]

    def get(env, mkey):
        r = env_df[(env_df.env == env) & (env_df.metric == mkey)].iloc[0]
        return float(r["mean"]), float(r["std"])
    x = np.arange(len(ENV_ORDER)); dx = 0.16
    fig, ax = plt.subplots(figsize=(115 * MM, 68 * MM))
    for s, e, lab, shade in ENV_FAMILY:
        if e < s:
            continue
        if shade != "#FFFFFF":
            ax.axvspan(s - 0.5, e + 0.5, color=shade, zorder=0)
        ax.text((s + e) / 2, 0.952, lab, ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#555")
    ax.grid(axis="y", color=GRID_COL, lw=0.5, zorder=1); ax.set_axisbelow(True)
    for j, (mkey, disp) in enumerate(METRICS):
        off = (j - 1.5) * dx
        means = [get(env, mkey)[0] for env in ENV_ORDER]; stds = [get(env, mkey)[1] for env in ENV_ORDER]
        c = MCOLOR[disp]
        ax.plot(x + off, means, color=c, lw=0.7, alpha=0.45, zorder=2)
        ax.errorbar(x + off, means, yerr=stds, fmt="o", ms=3.6, mfc=c, mec="white", mew=0.4, ecolor=c, elinewidth=0.7, capsize=1.4, zorder=3, label=disp)
    if SELECTED in ENV_ORDER:
        ax.scatter(ENV_ORDER.index(SELECTED), 0.935, marker="*", s=46, color="#B11116", zorder=5, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels([e.split("-", 1)[1] if "-" in e else e for e in ENV_ORDER], fontsize=6.6)
    ax.set_xlim(-0.5, len(ENV_ORDER) - 0.5); ax.set_ylim(0.45, 0.95); ax.set_yticks(np.arange(0.5, 0.951, 0.1))
    ax.tick_params(length=2.5, width=0.35); ax.set_ylabel("CV test score"); ax.set_xlabel("Env protein language model")
    ax.set_title("Env encoder ablation", fontsize=8, pad=14)
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=MCOLOR[d], markeredgecolor="white", markersize=4.5, label=d) for _, d in METRICS]
    h.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#B11116", markeredgecolor="none", markersize=8, label="Selected (ESM-C 600M)"))
    ax.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), handlelength=1.0, columnspacing=1.4, handletextpad=0.3, fontsize=6.2)
    fig.tight_layout(); _save(fig, "panel_D_plm")


# ----------------------------------------------------- E: module ablation
def panel_module():
    df = _table(SH_MODULE)
    METRICS = ["ACC", "PRE", "REC", "F1-Score"]
    MCOL = {"ACC": "#1F4E79", "PRE": "#3C78B4", "REC": "#7BA8D6", "F1-Score": "#B8D2EC"}
    ORDER = ["drop_pair", "drop_z_ab", "drop_z_env", "drop_gate", "drop_bilinear"]
    LABEL = {"drop_pair": "pair (−)", "drop_z_ab": "z_ab (−)", "drop_z_env": "z_env (−)",
             "drop_gate": "attention-gate (−)", "drop_bilinear": "prediction-summary (−)"}
    full = {m: df[(df.config == "full") & (df.metric == m)]["value"].iloc[0] for m in METRICS}
    delta = {c: {m: df[(df.config == c) & (df.metric == m)]["value"].iloc[0] - full[m] for m in METRICS} for c in ORDER}
    bar_h = 0.18; yc = np.arange(len(ORDER))[::-1]
    fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
    for i, m in enumerate(METRICS):
        off = (i - (len(METRICS) - 1) / 2) * bar_h
        ax.barh(yc + off, [delta[c][m] for c in ORDER], height=bar_h, color=MCOL[m], edgecolor="white", linewidth=0.3, label=m, zorder=3)
    ax.axvline(0, color="#444", lw=0.9, zorder=4)
    ax.set_yticks(yc); ax.set_yticklabels([LABEL[c] for c in ORDER])
    ax.set_xlabel("Δ decision quality vs full surrogate\n(ablated − full; held-out 20%, threshold 0.5)")
    ax.tick_params(length=2.5); ax.grid(axis="x", color="#ECECEC", lw=0.7, zorder=0); ax.set_axisbelow(True)
    xlo = min(-0.065, min(min(delta[c].values()) for c in ORDER) - 0.01)
    xhi = max(0.015, max(max(delta[c].values()) for c in ORDER) + 0.006)
    ax.set_xlim(xlo, xhi); ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6.3, handlelength=1.1, columnspacing=1.2, handletextpad=0.4)
    ax.set_title("Architecture-module ablation of the UniCAV SHAP surrogate", fontsize=8, pad=18)
    fig.tight_layout(); _save(fig, "panel_E_module_ablation")


# ----------------------------------------------------- F: architecture ROC
ARCH = {"Token-level": ("Token-level", "#9B7FB5", 1.0, 3),
        "Sequence-level": ("Sequence-level", "#3C9D9B", 1.0, 4),
        "Token-level + Sequence-level": ("Token + Sequence (UniCAV)", "#0A6DAF", 1.7, 6)}


def panel_arch():
    raw = _raw("Fig08_arch_ROC")
    hs, hc = list(raw.index[raw[0] == "architecture"])[:2]
    cs = raw.iloc[hs].tolist(); su = raw.iloc[hs + 1:hc].copy(); su.columns = cs
    su = su[su["architecture"].isin(ARCH)]
    auroc = {(r["architecture"], r["dataset"]): float(r["AUROC"]) for _, r in su.iterrows()}
    cc = raw.iloc[hc].tolist(); co = raw.iloc[hc + 1:].copy(); co.columns = cc
    co = co.dropna(subset=["architecture", "fpr", "tpr"])
    co["fpr"] = pd.to_numeric(co["fpr"], errors="coerce"); co["tpr"] = pd.to_numeric(co["tpr"], errors="coerce")
    co = co.dropna(subset=["fpr", "tpr"])
    fig, ax = plt.subplots(figsize=(72 * MM, 70 * MM))
    ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.6, color="#B6BCC6", zorder=1)
    style = {"CV pooled OOF": "-", "CH505 ensemble": (0, (3.2, 2.2))}
    for akey, (lab, col, lw, z) in ARCH.items():
        for dskey, ls in style.items():
            d = co[(co.architecture == akey) & (co.dataset == dskey)].sort_values("fpr")
            ax.plot(d["fpr"], d["tpr"], color=col, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=2.5, width=0.35)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Architecture ablation ROC", fontsize=8, pad=4)
    arch_h = [Line2D([0], [0], color=col, lw=lw, label=f"{lab}  {auroc[(akey,'CV pooled OOF')]:.3f} / {auroc[(akey,'CH505 ensemble')]:.3f}") for akey, (lab, col, lw, z) in ARCH.items()]
    style_h = [Line2D([0], [0], color="#555", lw=1.1, ls="-", label="CV pooled OOF"),
               Line2D([0], [0], color="#555", lw=1.1, ls=(0, (3.2, 2.2)), label="External CH505")]
    leg1 = ax.legend(handles=arch_h, loc="lower right", fontsize=5.6, handlelength=1.4, labelspacing=0.35, borderaxespad=0.3, title="Architecture (CV / CH505 AUROC)", title_fontsize=5.8)
    ax.add_artist(leg1)
    ax.legend(handles=style_h, loc="upper left", fontsize=5.8, handlelength=1.6, labelspacing=0.3, borderaxespad=0.4)
    fig.tight_layout(); _save(fig, "panel_F_arch")


# ----------------------------------------------------- G/H: baseline comparison
METHODS = ["UniCAV", "BRAVE", "LBUM", "bNAb-ReP"]
METHOD_FROM_CSV = {"PAVE": "UniCAV", "bNAb-ReP": "bNAb-ReP", "LBUM": "LBUM", "BRAVE": "BRAVE"}
EDGE = {"UniCAV": "#C0392B", "bNAb-ReP": "#B0CCE6", "LBUM": "#6FA3CE", "BRAVE": "#0A6DAF"}
FILL = {m: mpl.colors.to_rgba(c, 0.45) for m, c in EDGE.items()}
BMETRICS = [("AUC", "AUROC"), ("Accuracy", "Accuracy"), ("F1", "F1")]


def panel_baseline():
    d = _table("source_data_values_and_ranks"); d["method"] = d["method"].map(METHOD_FROM_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 62 * MM)); rng = np.random.default_rng(0)
    for ax, (mkey, disp) in zip(axes, BMETRICS):
        sub = d[d["metric"] == mkey]
        data = [sub.loc[sub["method"] == m, "value"].values for m in METHODS]
        pos = np.arange(len(METHODS))
        ax.grid(axis="y", color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=AXIS_COL, linewidth=0.9), whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(FILL[m]); patch.set_edgecolor(EDGE[m]); patch.set_linewidth(0.9)
        for i, ln in enumerate(bp["whiskers"]):
            ln.set_color(EDGE[METHODS[i // 2]])
        for i, ln in enumerate(bp["caps"]):
            ln.set_color(EDGE[METHODS[i // 2]])
        for i, m in enumerate(METHODS):
            y = data[i]; x = pos[i] + rng.uniform(-0.15, 0.15, size=len(y))
            ax.scatter(x, y, s=4.5, color=EDGE[m], alpha=0.55, linewidths=0, zorder=3)
            ax.scatter(pos[i], np.mean(y), marker="D", s=13, facecolor="white", edgecolor=EDGE[m], linewidths=1.0, zorder=4)
        ax.set_xticks(pos); ax.set_xticklabels(METHODS, rotation=30, ha="right")
        ax.set_title(disp, fontsize=8, pad=3); ax.set_ylim(0.10 if disp == "F1" else 0.45, 1.0)
        ax.tick_params(length=2.5, width=0.35)
    axes[0].set_ylabel("Per-antibody score")
    leg = [Patch(facecolor=FILL[m], edgecolor=EDGE[m], label=m) for m in METHODS]
    leg.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#555", markersize=4.5, label="Mean"))
    fig.legend(handles=leg, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.12), handlelength=1.2, columnspacing=1.3, fontsize=6.4)
    fig.subplots_adjust(wspace=0.26); _save(fig, "panel_G_baseline_values")

    r = _table("source_data_values_and_ranks")
    r["antibody"] = r["antibody"].replace({"4.00E+10": "4E10"})
    baseval = {(x.antibody, x.metric, x.method): float(x.value) for x in r.itertuples() if x.method != "PAVE"}
    uni = {(x.antibody, x.metric): float(x.value) for x in r.itertuples() if x.method == "PAVE"}
    abts = list(dict.fromkeys(r["antibody"])); mkeys = [k for k, _ in BMETRICS]
    wins = {m: {k: 0 for k in mkeys} for m in METHODS}
    for ab in abts:
        for mk in mkeys:
            vals = {"UniCAV": uni[(ab, mk)], "bNAb-ReP": baseval[(ab, mk, "bNAb-ReP")],
                    "LBUM": baseval[(ab, mk, "LBUM")], "BRAVE": baseval[(ab, mk, "BRAVE")]}
            best = max(round(v, 2) for v in vals.values())
            for m, v in vals.items():
                if round(v, 2) == best:
                    wins[m][mk] += 1
    wins = pd.DataFrame(wins).T.reindex(METHODS)[mkeys].astype(int); n_ab = len(abts)
    fig, ax = plt.subplots(figsize=(78 * MM, 64 * MM))
    ax.grid(axis="y", color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    x = np.arange(len(BMETRICS)); w = 0.19
    for j, m in enumerate(METHODS):
        xpos = x + (j - 1.5) * w
        ax.bar(xpos, wins.loc[m], width=w, color=FILL[m], edgecolor=EDGE[m], linewidth=0.8, zorder=2)
        for xp, v in zip(xpos, wins.loc[m]):
            ax.text(xp, v + 0.4, str(int(v)), ha="center", va="bottom", fontsize=5.6, color=EDGE[m])
    ax.set_xticks(x); ax.set_xticklabels([disp for _, disp in BMETRICS])
    ax.set_ylabel(f"No. antibodies ranked #1 (of {n_ab})"); ax.set_ylim(0, n_ab); ax.tick_params(length=2.5, width=0.35)
    ax.set_title("Top-1 wins per metric", fontsize=8, pad=3)
    leg = [Patch(facecolor=FILL[m], edgecolor=EDGE[m], label=m) for m in METHODS]
    ax.legend(handles=leg, ncol=2, loc="upper center", handlelength=1.2, columnspacing=1.2, handletextpad=0.4, fontsize=6.0)
    _save(fig, "panel_H_baseline_ranks")


# ----------------------------------------------------- I: regression scatter
def _load_pred(sheet):
    raw = _raw(sheet)
    hdr = raw.index[raw[0] == "Prediction values"][0] + 1
    cols = raw.iloc[hdr].tolist(); data = raw.iloc[hdr + 1:].copy(); data.columns = cols
    data = data.dropna(subset=["y_true_real", "y_pred_real"])
    return data.astype({"y_true_real": float, "y_pred_real": float})


def panel_regression():
    df = _load_pred("Fig15_reg_CV")
    yt = df["y_true_real"].to_numpy(); yp = df["y_pred_real"].to_numpy()
    resid = yp - yt
    r = float(np.corrcoef(yt, yp)[0, 1]); mae = float(np.mean(np.abs(resid)))
    n = len(yt)
    lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max()); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
    cmap = LinearSegmentedColormap.from_list("blues_proj", ["#EAF2FA", "#D3E2F2", "#7FB0D8", "#0A6DAF", "#06436C"])
    fig, ax = plt.subplots(figsize=(62 * MM, 62 * MM))
    ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.7, color="#ACB4BF", zorder=1, label="y = x")
    mappable = ax.hexbin(yt, yp, gridsize=48, bins="log", cmap=cmap, mincnt=1, linewidths=0.0, zorder=2)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Count (log)", fontsize=6.3); cb.ax.tick_params(labelsize=5.8, width=0.35, length=2); cb.outline.set_linewidth(0.35)
    slope, intercept = np.polyfit(yt, yp, 1); xs = np.array([lo, hi])
    ax.plot(xs, slope * xs + intercept, lw=1.2, color="#DB6C50", zorder=4, label="Linear fit")
    ax.text(0.04, 0.97, f"n = {n:,}\nPearson r = {r:.3f}\nMAE = {mae:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, linespacing=1.45)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured log₁₀ IC₅₀"); ax.set_ylabel("Predicted log₁₀ IC₅₀")
    ax.xaxis.set_major_locator(MultipleLocator(1.0)); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f")); ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(length=2.5, width=0.35)
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
    ax.set_title("Cross-validation regression", fontsize=8, pad=4)
    fig.tight_layout(); _save(fig, "panel_I_regression_cv")


# ----------------------------------------------------- J: error tolerance
def panel_error_tolerance():
    def err(sheet):
        d = _load_pred(sheet)
        return np.abs(d["y_pred_real"].to_numpy() - d["y_true_real"].to_numpy())
    cv = err("Fig15_reg_CV"); ch = err("Fig16_reg_CH505")
    x = np.linspace(0, 2.5, 400)
    cdf = lambda e: np.array([(e <= t).mean() for t in x])
    fcv, fch = cdf(cv), cdf(ch); frac_at = lambda e, t: float((e <= t).mean())
    fig, ax = plt.subplots(figsize=(62 * MM, 66 * MM))
    ax.grid(True, color=GRID_COL, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for t, lab in ((np.log10(2), "2×"), (np.log10(3), "3×"), (1.0, "10×")):
        ax.axvline(t, color="#B6BCC6", lw=0.6, ls=(0, (3, 2.5)), zorder=1)
        ax.text(t, 1.02, lab, ha="center", va="bottom", fontsize=5.8, color="#777")
    ax.plot(x, fcv, color="#0A6DAF", lw=1.6, zorder=4, label=f"CV (pooled OOF), n={len(cv):,}")
    ax.plot(x, fch, color="#C49A48", lw=1.6, zorder=4, label=f"CH505 (ensemble), n={len(ch):,}")
    for f, e, col, dy in ((fcv, cv, "#0A6DAF", -0.055), (fch, ch, "#C49A48", 0.03)):
        y = frac_at(e, 1.0)
        ax.scatter([1.0], [y], s=16, color=col, edgecolor="white", lw=0.5, zorder=5)
        ax.text(1.06, y + dy, f"{y*100:.0f}%", color=col, fontsize=6.0, va="center", fontweight="bold")
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.0); ax.set_xticks(np.arange(0, 2.51, 0.5)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(length=2.5, width=0.35)
    ax.set_xlabel("Absolute error tolerance (log₁₀ IC₅₀)"); ax.set_ylabel("Fraction of pairs within tolerance")
    ax.set_title("Error-tolerance curve", fontsize=8, pad=8)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.4, labelspacing=0.35, borderaxespad=0.4)
    fig.tight_layout(); _save(fig, "panel_J_error_tolerance")


def main():
    for fn in (panel_metrics, panel_confusion, panel_group, panel_plm, panel_module,
               panel_arch, panel_baseline, panel_regression, panel_error_tolerance):
        try:
            fn()
        except Exception as e:
            print(f"[WARN] {fn.__name__} failed: {e}")
    print("Figure 2 panels ->", OUT)


if __name__ == "__main__":
    main()
