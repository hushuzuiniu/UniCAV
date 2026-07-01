#!/usr/bin/env python3
"""Figure S3 - Interpretability of UniCAV's paired representation (panels B-E).
Ported verbatim (styling + plotting) from original_analysis/draw/:
  B latent_branch_probes_styled.py  linear-probe accuracy per branch/target
  C latent_by_antibody_styled.py    fold-1 pair-embedding t-SNE by antibody lineage
  D explain_shap_styled.py          top internal features by mean|SHAP|, by module
  E latent_neut_top3_pc_styled.py   within-antibody neutralisation (PCA + probe)
Panel A (Env 3D attention render) is a structural image and is not reproduced here.
Data: source_data/FigureS3.xlsx. Output -> FigureS3_panels/.
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
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
XLSX = HERE / "source_data" / "FigureS3.xlsx"
OUT = HERE / "FigureS3_panels"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
AXIS_COL = "#1A1A1A"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS_COL, "axes.labelcolor": AXIS_COL,
    "text.color": AXIS_COL, "xtick.color": AXIS_COL, "ytick.color": AXIS_COL,
    "legend.frameon": False, "figure.dpi": 150,
})


def _save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)


# ================================================================ B: branch probes
def panel_B():
    REP_COL = {"z_ab": "#6FB07A", "z_env": "#C49A48", "pair": "#0A6DAF"}
    REP_LBL = {"z_ab": "z$_{ab}$ (antibody)", "z_env": "z$_{env}$ (virus)", "pair": "pair"}
    LEAK, CHANCE = "#222222", "#888888"
    HELD = {"ab_id": "antibody-out", "virus_id": "virus-out"}
    raw = pd.read_excel(XLSX, sheet_name="B_branch_probes", header=None)
    h = raw.index[raw[0] == "target"][0]
    d = raw.iloc[h + 1:].copy(); d.columns = raw.iloc[h].tolist()
    d = d.dropna(subset=["target", "representation"])
    for c in ("cv_accuracy_random", "cv_accuracy_grouped", "chance"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    targets = list(dict.fromkeys(d["target"])); reps = ["z_ab", "z_env", "pair"]
    x = np.arange(len(targets)); w = 0.26
    fig, ax = plt.subplots(figsize=(110 * MM, 74 * MM))
    for i, rep in enumerate(reps):
        sub = d[d["representation"] == rep].set_index("target").loc[targets]
        xx = x + (i - 1) * w
        ax.bar(xx, sub["cv_accuracy_grouped"], width=w, color=REP_COL[rep], zorder=3)
        for xi, r in zip(xx, sub["cv_accuracy_random"]):
            ax.plot([xi - w * 0.46, xi + w * 0.46], [r, r], color=LEAK, lw=1.4, zorder=5, solid_capstyle="butt")
        for xi, g in zip(xx, sub["cv_accuracy_grouped"]):
            ax.text(xi, g + 0.018, f"{g:.2f}", ha="center", va="bottom", fontsize=5.8, color="#222")
    for xi, t in zip(x, targets):
        ch = d[d["target"] == t]["chance"].iloc[0]
        ax.plot([xi - 1.7 * w, xi + 1.7 * w], [ch, ch], color=CHANCE, lw=1.0, ls=(0, (3, 2)), zorder=6)
        held = d[d["target"] == t]["held_out"].iloc[0]
        ax.text(xi, -0.085, f"({HELD.get(held, held)})", ha="center", va="top", fontsize=5.6, color="#999")
    ax.set_xticks(x); ax.set_xticklabels([str(t).replace("\n", " ") for t in targets])
    ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Linear-probe accuracy"); ax.tick_params(length=2.5, width=0.6)
    ax.set_title("What each branch encodes — solid bar = grouped CV (leakage-free)", fontsize=7.8, pad=6)
    handles = ([Patch(facecolor=REP_COL[r], label=REP_LBL[r]) for r in reps]
               + [Line2D([0], [0], color=LEAK, lw=1.4, label="random CV (leaked)"),
                  Line2D([0], [0], color=CHANCE, lw=0.9, ls=(0, (3, 2)), label="chance (majority)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.0,
              handlelength=1.4, labelspacing=0.4, borderaxespad=0.0)
    fig.tight_layout(); _save(fig, "panelB_branch_probes.pdf")


# ================================================================ C: t-SNE by antibody
def panel_C():
    SPINE, GREY = "#C8CDD3", "#D8D8D8"; N_TOP, PROBE_ACC = 30, 0.995
    PAL30 = ["#3B6FB6", "#E8853A", "#4FA363", "#C0392B", "#8E6FB3", "#7A5C3E", "#D46BA8",
             "#2BA0A8", "#B6B12C", "#1F4E79", "#E6AB02", "#1B9E77", "#E7298A", "#66A61E",
             "#A6761D", "#5E3C99", "#D7301F", "#3690C0", "#80CDC1", "#8FB0DB", "#C51B7D",
             "#35978F", "#BF812D", "#762A83", "#01665E", "#DE77AE", "#4D9221", "#8C510A",
             "#542788", "#F46D43"]
    raw = pd.read_excel(XLSX, sheet_name="C_tsne", header=None)
    h = raw.index[raw[0] == "id"][0]
    d = raw.iloc[h + 1:].copy(); d.columns = raw.iloc[h].tolist()
    d = d.dropna(subset=["tsne_x", "tsne_y", "ab_lineage"])
    d["tsne_x"] = d["tsne_x"].astype(float); d["tsne_y"] = d["tsne_y"].astype(float)
    n_lin = d["ab_lineage"].nunique()
    top = d["ab_lineage"].value_counts().head(N_TOP).index.tolist()
    col_of = {lin: PAL30[i] for i, lin in enumerate(top)}
    fig, ax = plt.subplots(figsize=(124 * MM, 92 * MM))
    rest = d[~d["ab_lineage"].isin(top)]
    ax.scatter(rest["tsne_x"], rest["tsne_y"], s=2.0, c=GREY, linewidths=0, alpha=0.5)
    for lin in top:
        s = d[d["ab_lineage"] == lin]
        ax.scatter(s["tsne_x"], s["tsne_y"], s=2.6, color=col_of[lin], linewidths=0, alpha=0.9)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.tick_params(length=2.5, width=0.35)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE)
    ax.set_title("Pair embedding by antibody lineage", fontsize=8, pad=8)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=col_of[l],
                      markeredgecolor="none", markersize=4.0, label=l) for l in top]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
                          markeredgecolor="none", markersize=4.0, label=f"other ({n_lin - N_TOP})"))
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=5.0,
                    handletextpad=0.3, labelspacing=0.30, borderaxespad=0.0,
                    title=f"Antibody lineage (top {N_TOP} of {n_lin})", title_fontsize=6.0)
    leg._legend_box.align = "left"
    ax.text(0.015, 0.015, f"Linear probe (lineage): {PROBE_ACC:.3f}", transform=ax.transAxes,
            fontsize=6, color="#444", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    fig.tight_layout(); _save(fig, "panelC_tsne_by_antibody.pdf")


# ================================================================ D: SHAP top features
def panel_D():
    MOD = {"pair": ("pair", "#0A6DAF"), "z_ab": ("z_ab", "#6FB07A"),
           "attention_gate_summary": ("attention-gate", "#9B7FB5"),
           "z_env": ("z_env", "#C49A48"), "prediction_summary": ("prediction-summary", "#DB6C50")}
    TOP_N = 15

    def feat_label(f):
        return (str(f).replace("_PC", " PC").replace("bilinear_score", "bilinear")
                .replace("gate_l2", "gate L2").replace("gate_mean", "gate mean"))
    raw = pd.read_excel(XLSX, sheet_name="D_shap_top", header=None)
    h = raw.index[raw[0] == "feature"][0]
    d = raw.iloc[h + 1:].copy(); d.columns = raw.iloc[h].tolist()
    d = d.dropna(subset=["feature"])
    d["mean_abs_shap"] = pd.to_numeric(d["mean_abs_shap"], errors="coerce")
    d = d.dropna(subset=["mean_abs_shap"]).head(TOP_N).iloc[::-1]
    y = np.arange(len(d)); vals = d["mean_abs_shap"].to_numpy()
    cols = [MOD[g][1] for g in d["group"]]; vmax = vals.max()
    fig, ax = plt.subplots(figsize=(92 * MM, 92 * MM))
    ax.barh(y, vals, color=cols, edgecolor="white", linewidth=0.5, height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + vmax * 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.8, color="#1a1a1a")
    ax.set_xlim(0, vmax * 1.12); ax.set_yticks(y)
    ax.set_yticklabels([feat_label(f) for f in d["feature"]])
    ax.set_xlabel("mean |SHAP|  (contribution to UniCAV logit)", fontsize=8)
    ax.tick_params(length=2.5, width=0.6); ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", labelsize=7.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    used = list(dict.fromkeys(d["group"]))
    handles = [Patch(facecolor=MOD[g][1], edgecolor="white", label=MOD[g][0]) for g in MOD if g in used]
    ax.legend(handles=handles, loc="lower right", fontsize=5.8, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.4, title="Module", title_fontsize=6.0)
    fig.tight_layout(); _save(fig, "panelD_shap_top_features.pdf")


# ================================================================ E: within-antibody neut
def panel_E():
    NEUT, NONNEUT, ENVCOL = "#2C6E91", "#C0504D", "#C49A48"
    ORDER = ["TRNM", "CH235", "10E8"]
    EPI = {"fusion_peptide": "Fusion peptide", "CD4bs": "CD4bs", "MPER": "MPER"}
    df = pd.read_excel(XLSX, sheet_name="E_neut_persample")
    raw = pd.read_excel(XLSX, sheet_name="E_neut_summary", header=None)
    h = raw.index[raw[0] == "panel"][0]
    meta = raw.iloc[h + 1:].copy(); meta.columns = raw.iloc[h].tolist()
    meta = meta.dropna(subset=["lineage"]).set_index("lineage")

    def style(ax):
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#1A1A1A"); ax.spines[s].set_linewidth(0.6)
        ax.tick_params(length=2.5, width=0.6, colors=AXIS_COL)
    fig, axes = plt.subplots(2, 3, figsize=(176 * MM, 104 * MM),
                             gridspec_kw=dict(height_ratios=[1.05, 0.85], hspace=0.42, wspace=0.30))
    for j, lin in enumerate(ORDER):
        s = df[df["lineage"] == lin]
        neut = s[s["y_true"] == 1]; non = s[s["y_true"] == 0]
        epi = EPI.get(str(meta.loc[lin, "epitope"]), str(meta.loc[lin, "epitope"]))
        n = len(s); frac = s["y_true"].mean()
        auc = roc_auc_score(s["y_true"], s["cv_probe_score"]); env_auc = float(meta.loc[lin, "z_env_auc"])
        axp = axes[0, j]
        for gi, (grp, col) in enumerate(sorted([(non, NONNEUT), (neut, NEUT)], key=lambda g: -len(g[0]))):
            axp.scatter(grp["pc1"], grp["pc2"], s=7, c=col, alpha=0.72, linewidths=0, zorder=2 + gi)
        axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
        axp.set_title(f"{lin}", fontsize=8.5, fontweight="bold", pad=10)
        axp.text(0.5, 1.012, f"{epi} · n={n} · neut frac {frac:.2f}", transform=axp.transAxes,
                 ha="center", va="bottom", fontsize=6.2, color="#555")
        axp.tick_params(labelleft=False, labelbottom=False); style(axp)
        axh = axes[1, j]
        sc = s["cv_probe_score"].to_numpy(); bins = np.linspace(sc.min(), sc.max(), 26)
        maj, mcol = (non, NONNEUT) if len(non) >= len(neut) else (neut, NEUT)
        mino, ocol = (neut, NEUT) if len(non) >= len(neut) else (non, NONNEUT)
        axh.hist(maj["cv_probe_score"], bins=bins, color=mcol, alpha=0.62, linewidth=0, zorder=2)
        axh.hist(mino["cv_probe_score"], bins=bins, color=ocol, alpha=0.62, linewidth=0, zorder=3)
        axh.set_xlabel("Linear-probe score (virus-out CV)")
        if j == 0:
            axh.set_ylabel("Count")
        bbox = dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.7, edgecolor="none")
        axh.text(0.97, 0.95, f"pair AUC {auc:.2f}", transform=axh.transAxes, ha="right", va="top",
                 fontsize=7.2, color=NEUT, fontweight="bold", bbox=bbox)
        axh.text(0.97, 0.79, f"Env-only AUC {env_auc:.2f}", transform=axh.transAxes, ha="right", va="top",
                 fontsize=6.8, color=ENVCOL, fontweight="bold", bbox=bbox)
        style(axh)
    handles = [Line2D([0], [0], marker="o", ls="", ms=4.5, mec="none", mfc=NEUT, label="neutralising"),
               Line2D([0], [0], marker="o", ls="", ms=4.5, mec="none", mfc=NONNEUT, label="non-neutralising")]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=6.0, handletextpad=0.3,
                      labelspacing=0.3, borderaxespad=0.3)
    fig.suptitle("Pair representation separates neutralisation within each antibody (top-3 by separability)",
                 fontsize=8.6, y=0.99)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.90, bottom=0.095, hspace=0.45, wspace=0.28)
    fig.text(0.022, 0.715, "PCA of pair embedding", rotation=90, va="center", ha="center", fontsize=7.2, color="#444")
    fig.text(0.022, 0.27, "Probe-score distribution", rotation=90, va="center", ha="center", fontsize=7.2, color="#444")
    _save(fig, "panelE_within_antibody_neut.pdf")


if __name__ == "__main__":
    panel_B(); panel_C(); panel_D(); panel_E()
    print("Figure S3 panels ->", OUT)
    for f in sorted(OUT.glob("*.pdf")):
        print("  ", f.name)
