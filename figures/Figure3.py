#!/usr/bin/env python3
"""Figure 3 — Model interpretation and attention analysis (consolidated).

One runnable script for all Figure 3 panels (A-E), reading bundled data from
``Figure3.xlsx``. Panels are written to ``Figure3_panels/`` as PDF.

  A panel_attention_hxb2      global Env cross-attention vs HXB2 position
  B panel_attention_validation per-epitope contact-ranking AUROC
  C panel_shap_modules        SHAP module share (mean |SHAP| %)
  D panel_latent_outcome      fold-1 pair-embedding t-SNE by neutralization outcome
  E panel_latent_epitope      same t-SNE by antibody epitope class
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
FIG_XLSX = HERE / "source_data" / "Figure3.xlsx"
OUT = HERE / "Figure3_panels"
OUT.mkdir(parents=True, exist_ok=True)

AXIS_COL = "#1A1A1A"
GRID_COL = "#E7EBF0"
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.35, "axes.edgecolor": AXIS_COL,
    "axes.labelcolor": AXIS_COL, "text.color": AXIS_COL,
    "xtick.color": AXIS_COL, "ytick.color": AXIS_COL,
    "legend.frameon": False, "figure.dpi": 150,
})

SH_GLYCAN = "explain_ch505_annotated__profil"  # 31-char truncated


def _raw(sheet):
    return pd.read_excel(FIG_XLSX, sheet_name=sheet, header=None)


def _table(sheet):
    return pd.read_excel(FIG_XLSX, sheet_name=sheet)


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


# ----------------------------------------------------- A: attention vs HXB2
COL_ATTN = "#2C6E91"; BAND = "#F3C98B"; GLYCAN = "#5C9E3F"; TEXTC = "#26313D"
GP_SPLIT = 511.5; TOP_FRAC = 0.85; WINDOW = 9


def _smooth(values, window=WINDOW):
    y = np.asarray(values, float); half = window // 2; out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = float(np.nanmean(y[max(0, i - half):min(len(y), i + half + 1)]))
    return out


def _contiguous(positions, gap=2):
    runs, s, p = [], None, None
    for x in sorted(positions):
        if s is None:
            s = p = x
        elif x <= p + gap:
            p = x
        else:
            runs.append((s, p)); s = p = x
    if s is not None:
        runs.append((s, p))
    return runs


def panel_attention_hxb2():
    prof = _table("epitope_profiles")
    prof["hxb2_pos"] = pd.to_numeric(prof["hxb2_pos"], errors="coerce")
    prof["mean_attn"] = pd.to_numeric(prof["mean_attn"], errors="coerce")
    has_norm = "attention_over_mean" in prof.columns
    if has_norm:
        prof["attention_over_mean"] = pd.to_numeric(prof["attention_over_mean"], errors="coerce")
    prof = prof.dropna(subset=["ab_epitope_simple", "hxb2_pos", "mean_attn"])
    rows = []
    for cls, g in prof.groupby("ab_epitope_simple", sort=True):
        g = g.sort_values("hxb2_pos")
        if has_norm:
            raw = g["attention_over_mean"].to_numpy(float)
        else:
            denom = float(g["mean_attn"].mean())
            if not np.isfinite(denom) or denom <= 0:
                continue
            raw = g["mean_attn"].to_numpy(float) / denom
        if not np.isfinite(raw).any():
            continue
        rows.append(pd.DataFrame({"hxb2_pos": g["hxb2_pos"].astype(int).to_numpy(),
                                  "smoothed": _smooth(raw)}))
    per = pd.concat(rows, ignore_index=True)
    glob = per.groupby("hxb2_pos")["smoothed"].mean().reset_index().sort_values("hxb2_pos")
    x = glob["hxb2_pos"].to_numpy(); y = glob["smoothed"].to_numpy(); xmax = float(x.max())
    thr = np.quantile(y, TOP_FRAC); hi = _contiguous(x[y >= thr].astype(int))
    reg = _table("env_regions_in_alignment")
    gp = _table(SH_GLYCAN)
    glycans = gp.loc[gp["is_N_glycosylation_site"].astype(str).str.lower().isin(["true", "1", "1.0"]), "hxb2_pos"].astype(float).to_numpy()

    fig = plt.figure(figsize=(183 * MM, 76 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 0.5], hspace=0.08)
    ax = fig.add_subplot(gs[0]); axg = fig.add_subplot(gs[1], sharex=ax)
    for s, e in hi:
        ax.axvspan(s - 0.5, e + 0.5, color=BAND, alpha=0.40, zorder=0)
    for _, r in reg.iterrows():
        ax.axvline(r.hxb2_end + 0.5, color="#E2E6EB", lw=0.4, zorder=0)
        ax.text((r.hxb2_start + r.hxb2_end) / 2, 1.005, r.region, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.6, fontweight="bold", color=TEXTC, clip_on=False)
    ax.axvline(GP_SPLIT, color="#AAB0B7", lw=0.8, zorder=1)
    ax.text(GP_SPLIT / 2, 1.26, "gp120", style="italic", fontsize=7, color="#9AA0A6", ha="center", va="center")
    ax.text((GP_SPLIT + xmax) / 2, 1.26, "gp41", style="italic", fontsize=7, color="#9AA0A6", ha="center", va="center")
    ax.axhline(1.0, ls="--", color="#999999", lw=0.4, zorder=1)
    ax.plot(x, y, color=COL_ATTN, lw=0.9, zorder=4)
    ax.set_xlim(1, xmax); ax.set_ylim(0.80, max(1.42, y.max() * 1.08))
    ax.set_yticks([0.8, 1.0, 1.2, 1.4]); ax.set_yticklabels([f"{v:.1f}×" for v in [0.8, 1.0, 1.2, 1.4]])
    ax.set_ylabel("Cross-attention / mean"); ax.tick_params(labelbottom=False, length=2.5, width=0.35)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=BAND, alpha=0.40, label="High-attention region (top 15%)"))
    ax.plot([], [], color=COL_ATTN, lw=1.2, label="Smoothed attention / mean")
    ax.legend(loc="upper right", fontsize=5.8, borderaxespad=0.3)
    ax.set_title("Global Env cross-attention with HXB2 region annotations (shaded = top-15% attention)", fontsize=8, pad=14)
    for gx in glycans:
        axg.plot([gx, gx], [0, 1], color=GLYCAN, lw=0.6)
    axg.set_ylim(0, 1); axg.set_yticks([0.5]); axg.set_yticklabels(["N-glycan"], fontsize=6.2)
    axg.set_xlim(1, xmax); axg.axvline(GP_SPLIT, color="#AAB0B7", lw=0.8)
    axg.set_xlabel("HXB2 Env residue position  (gp120: 1–511 | gp41: 512–856)"); axg.tick_params(length=2.5, width=0.35)
    for sp in ("left", "right", "top"):
        axg.spines[sp].set_visible(False)
    _save(fig, "panel_A_attention_hxb2")


# ----------------------------------------------------- B: attention validation
DISPLAY = {"V1V2": "V1V2", "V3": "V3", "CD4i": "CD4i", "CD4bs": "CD4bs",
           "interface": "Interface", "fusion_peptide": "Fusion peptide", "MPER": "MPER"}
SUBUNIT = {"V1V2": "gp120", "V3": "gp120", "CD4i": "gp120", "CD4bs": "gp120",
           "interface": "gp120/gp41", "fusion_peptide": "gp41", "MPER": "gp41"}


def panel_attention_validation():
    raw = _raw("Fig17_attention_hxb2")
    h = next(i for i in range(len(raw)) if "AUROC_mean" in [str(v) for v in raw.iloc[i]])
    cols = [str(c) for c in raw.iloc[h].tolist()]
    blk = raw.iloc[h + 1:].copy()
    blk.columns = cols + [f"_{k}" for k in range(raw.shape[1] - len(cols))]
    keep = ["ab_epitope_simple", "n_samples", "EAE_mean", "AUROC_mean", "AUROC_sem", "AUROC_rand_mean"]
    blk = blk[keep].dropna(subset=["AUROC_mean"])
    blk = blk[blk["ab_epitope_simple"].astype(str) != "nan"]
    for c in keep[1:]:
        blk[c] = pd.to_numeric(blk[c], errors="coerce")
    df = blk.sort_values("AUROC_mean").reset_index(drop=True)
    y = np.arange(len(df)); auroc = df["AUROC_mean"].to_numpy(); sem = df["AUROC_sem"].to_numpy()
    names = [DISPLAY.get(e, e) for e in df["ab_epitope_simple"]]
    colors = ["#2C6E91" if a >= 0.5 else "#9AA0A6" for a in auroc]
    fig, ax = plt.subplots(figsize=(92 * MM, 60 * MM))
    ax.barh(y, auroc, height=0.66, color=colors, zorder=2, xerr=sem, error_kw=dict(elinewidth=0.6, ecolor="#555", capsize=1.5))
    ax.axvline(0.5, color="#777", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.5, 0.985, "chance", transform=blended_transform_factory(ax.transData, ax.transAxes), color="#777", fontsize=6.2, ha="center", va="top")
    for yi, a in zip(y, auroc):
        out = a >= 0.5
        ax.text(a + (0.012 if out else -0.012), yi, f"{a:.2f}", ha="left" if out else "right", va="center", fontsize=6.8, color="#1a1a1a")
    ax.set_yticks(y); ax.set_yticklabels(names)
    for yi, e in zip(y, df["ab_epitope_simple"]):
        ax.text(1.005, yi, SUBUNIT.get(e, ""), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.6, color="#9AA0A6", fontstyle="italic", clip_on=False)
    ax.set_xlim(0, 0.86); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Epitope contact-ranking AUROC  (chance = 0.50)"); ax.margins(y=0.04)
    _save(fig, "panel_B_attention_validation")


# ----------------------------------------------------- C: SHAP module share
MOD = {"pair": ("pair", "#0A6DAF"), "z_ab": ("z_ab", "#6FB07A"),
       "attention_gate_summary": ("attention-gate", "#9B7FB5"),
       "z_env": ("z_env", "#C49A48"), "prediction_summary": ("prediction-summary", "#DB6C50")}


def panel_shap_modules():
    raw = _raw("Fig22_shap_modules")
    h = raw.index[raw[0] == "group"][0]
    c = raw.iloc[h].tolist(); d = raw.iloc[h + 1:].copy(); d.columns = c
    d = d.dropna(subset=["group"])
    d["fraction"] = pd.to_numeric(d["fraction"], errors="coerce")
    d = d.dropna(subset=["fraction"]).sort_values("fraction")
    y = np.arange(len(d))
    cols = [MOD.get(g, (g, "#999999"))[1] for g in d["group"]]
    labs = [MOD.get(g, (g, ""))[0] for g in d["group"]]
    fig, ax = plt.subplots(figsize=(92 * MM, 60 * MM))
    ax.barh(y, d["fraction"] * 100, color=cols, edgecolor="white", linewidth=0.5, height=0.66, zorder=2)
    for yi, fr in zip(y, d["fraction"]):
        ax.text(fr * 100 + 1.4, yi, f"{fr*100:.1f}%", va="center", ha="left", fontsize=6.8, color="#1a1a1a")
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total mean |SHAP| (%)", fontsize=8)
    ax.tick_params(length=2.5, width=0.6)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    _save(fig, "panel_C_shap_modules")


# ----------------------------------------------------- D/E: latent t-SNE
def _load_latent():
    raw = _raw("Fig24_latent_by_antibody")
    h = raw.index[raw[0] == "id"][0]
    d = raw.iloc[h + 1:].copy(); d.columns = raw.iloc[h].tolist()
    return d


COL_SENSITIVE = "#2C7FB8"
COL_RESISTANT = "#D6604D"


def panel_latent_outcome():
    d = _load_latent().dropna(subset=["tsne_x", "tsne_y", "y_true"]).copy()
    for col in ("tsne_x", "tsne_y", "y_true"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["tsne_x", "tsne_y", "y_true"])
    d = d[d["y_true"].isin([0, 1])].copy()
    d["y_true"] = d["y_true"].astype(int)

    # Shuffle the draw order so neither outcome class systematically overlays the other.
    d = d.sample(frac=1.0, random_state=0).reset_index(drop=True)
    colors = np.where(d["y_true"].to_numpy() == 1, COL_SENSITIVE, COL_RESISTANT)
    n_sensitive = int((d["y_true"] == 1).sum())
    n_resistant = int((d["y_true"] == 0).sum())

    fig, ax = plt.subplots(figsize=(78 * MM, 78 * MM))
    ax.scatter(d["tsne_x"], d["tsne_y"], s=2.2, c=colors, linewidths=0, alpha=0.85)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.tick_params(length=2.5, width=0.35)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C8CDD3")
    ax.set_title("Pair embedding by neutralization outcome", fontsize=8, pad=8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_SENSITIVE,
               markeredgecolor="none", markersize=4.5,
               label=f"Sensitive  n={n_sensitive:,}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_RESISTANT,
               markeredgecolor="none", markersize=4.5,
               label=f"Resistant  n={n_resistant:,}"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
              ncol=1, fontsize=6.0, handletextpad=0.3, labelspacing=0.35,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.75)
    fig.tight_layout(); _save(fig, "panel_D_latent_outcome")


EPI_ORDER = ["CD4bs", "V3", "V1V2", "interface", "fusion_peptide", "MPER", "CD4i", "silent_face", "glycan_other", "multi", "other", "unknown"]
EPI_PALETTE = ["#2E6E8E", "#E8A33D", "#7BA6C2", "#C0504D", "#6A9A5B", "#8C6BB1", "#C9A66B", "#4F81BD", "#9BBB59", "#CC79A7", "#D99694", "#999999"]
EPI_DISPLAY = {"fusion_peptide": "Fusion peptide", "silent_face": "Silent face", "glycan_other": "Glycan (other)"}


def panel_latent_epitope():
    d = _load_latent().dropna(subset=["tsne_x", "tsne_y", "ab_epitope_simple"])
    d["tsne_x"] = d["tsne_x"].astype(float); d["tsne_y"] = d["tsne_y"].astype(float)
    classes = [c for c in EPI_ORDER if c in set(d["ab_epitope_simple"])]
    col_of = {c: EPI_PALETTE[EPI_ORDER.index(c)] for c in classes}
    draw_order = [c for c in d["ab_epitope_simple"].value_counts().index.tolist() if c in classes]
    fig, ax = plt.subplots(figsize=(78 * MM, 78 * MM))
    for c in draw_order:
        s = d[d["ab_epitope_simple"] == c]
        ax.scatter(s["tsne_x"], s["tsne_y"], s=2.2, color=col_of[c], linewidths=0, alpha=0.85)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.tick_params(length=2.5, width=0.35)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C8CDD3")
    ax.set_title("Pair embedding by epitope", fontsize=8, pad=8)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=col_of[c], markeredgecolor="none", markersize=4.0, label=EPI_DISPLAY.get(c, c)) for c in classes]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, fontsize=5.0,
              handletextpad=0.2, columnspacing=0.6, labelspacing=0.25, frameon=True, facecolor="white", edgecolor="none", framealpha=0.8)
    ax.text(0.015, 0.015, "Linear probe (epitope): 0.98", transform=ax.transAxes, fontsize=6, color="#444", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    fig.tight_layout(); _save(fig, "panel_E_latent_epitope")


def main():
    for fn in (panel_attention_hxb2, panel_attention_validation, panel_shap_modules,
               panel_latent_outcome, panel_latent_epitope):
        try:
            fn()
        except Exception as e:
            print(f"[WARN] {fn.__name__} failed: {e}")
    print("Figure 3 panels ->", OUT)


if __name__ == "__main__":
    main()
