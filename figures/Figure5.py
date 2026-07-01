#!/usr/bin/env python3
"""Figure 5 (Python panels) — CH505 attention interpretation.

Reads ``Figure5.xlsx``; panels written to ``Figure5_panels/`` as PDF.
  A panel_attention   CH505 Env cross-attention vs HXB2 + glycan/selection rails
  B panel_latent      CH505 pair embedding (PCA) by sensitive/resistant
  C panel_probe       per-representation neutralisation linear probe
Panels D/E (attention-vs-selection violin + scatter) are R: run Figure5.R.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG_XLSX = HERE / "source_data" / "Figure5.xlsx"
OUT = HERE / "Figure5_panels"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
AXIS_COL = "#1A1A1A"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150,
})


def _table(s):
    return pd.read_excel(FIG_XLSX, sheet_name=s)


def _raw(s):
    return pd.read_excel(FIG_XLSX, sheet_name=s, header=None)


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", stem)


def _smooth(y, w=9):
    y = np.asarray(y, float); half = w // 2; out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = float(np.nanmean(y[max(0, i - half):min(len(y), i + half + 1)]))
    return out


def _contiguous(positions, gap=2):
    runs, s, p = [], None, None
    for x in sorted(int(v) for v in positions):
        if s is None:
            s = p = x
        elif x <= p + gap:
            p = x
        else:
            runs.append((s, p)); s = p = x
    if s is not None:
        runs.append((s, p))
    return runs


# ----------------------------------------------------- A: attention + rails
ACCENT = "#2E6E8E"; SHADE = "#E8A33D"; GLYCO = "#6A9A5B"; POSSEL = "#DB6C50"; TEXTC = "#26313D"


def panel_attention():
    prof = _table("profile").sort_values("hxb2_pos")
    prof["hxb2_pos"] = pd.to_numeric(prof["hxb2_pos"], errors="coerce").astype(int)
    x = prof["hxb2_pos"].to_numpy(int)
    raw = pd.to_numeric(prof["attention_over_mean"], errors="coerce").to_numpy(float)
    y = _smooth(raw)
    thr = float(np.quantile(y, 0.85)); bands = _contiguous(x[y >= thr])
    gly = prof.loc[prof["is_N_glycosylation_site"].astype(float) == 1, "hxb2_pos"].to_numpy()
    possel = prof.loc[prof["is_positively_selected"].astype(float) == 1, "hxb2_pos"].to_numpy()
    env = _table("env_regions"); ymax = float(np.nanmax(y))
    fig, (ax, axg, axp) = plt.subplots(3, 1, figsize=(183 * MM, 82 * MM), sharex=True,
                                       gridspec_kw=dict(height_ratios=[6.0, 0.55, 0.55], hspace=0.08))
    for s, e in bands:
        ax.axvspan(s - 0.5, e + 0.5, color=SHADE, alpha=0.40, lw=0, zorder=0)
    for _, r in env.iterrows():
        ax.axvline(r.hxb2_end + 0.5, color="#E2E6EB", lw=0.4, zorder=0)
        ax.text((r.hxb2_start + r.hxb2_end) / 2, 1.005, r.region, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=TEXTC, clip_on=False)
    ax.axhline(1.0, color="#aaaaaa", lw=0.5, ls=(0, (3, 3)), zorder=1)
    ax.axvline(511.5, color="#cccccc", lw=0.7, zorder=1)
    ax.plot(x, y, color=ACCENT, lw=1.0, zorder=3)
    ax.set_ylim(0.80, max(1.32, ymax * 1.08)); ax.set_xlim(1, 856)
    ax.set_yticks([0.8, 1.0, 1.2]); ax.set_yticklabels(["0.8×", "1.0×", "1.2×"], fontsize=6)
    ax.set_ylabel("Cross-attention / mean")
    ax.set_title("CH505 Env cross-attention with HXB2 region annotations (shaded = top-15% attention)", fontsize=7.7, pad=14)
    ax.annotate("gp120", xy=(256, 1.26), ha="center", fontsize=6.2, color="#888", fontstyle="italic")
    ax.annotate("gp41", xy=(684, 1.26), ha="center", fontsize=6.2, color="#888", fontstyle="italic")
    axg.vlines(gly, 0, 1, color=GLYCO, lw=0.6); axg.set_ylim(0, 1); axg.set_yticks([0.5]); axg.set_yticklabels(["N-glycan"], fontsize=5.8)
    axg.tick_params(left=False, labelbottom=False, length=0)
    for s in ("left", "right", "top", "bottom"):
        axg.spines[s].set_visible(False)
    axg.axvline(511.5, color="#cccccc", lw=0.7)
    axp.vlines(possel, 0, 1, color=POSSEL, lw=0.55); axp.set_ylim(0, 1); axp.set_yticks([0.5])
    axp.set_yticklabels(["Positively-\nselected"], fontsize=5.8, linespacing=0.9)
    axp.tick_params(left=False, length=2.5, width=0.35)
    for s in ("left", "right", "top"):
        axp.spines[s].set_visible(False)
    axp.axvline(511.5, color="#cccccc", lw=0.7)
    axp.set_xlabel("HXB2 Env residue position  (gp120: 1–511 | gp41: 512–856)")
    h = [plt.Rectangle((0, 0), 1, 1, color=SHADE, alpha=0.40), plt.Line2D([0], [0], color=ACCENT, lw=1.4)]
    ax.legend(h, ["High-attention region (top 15%)", "Smoothed attention / mean"], loc="upper right", ncol=1, handletextpad=0.4)
    _save(fig, "panel_A_attention")


# ----------------------------------------------------- B/C: latent + probe
def _load_latent():
    raw = _raw("ch505_latent")
    hr = next(i for i in range(raw.shape[0]) if "pc1" in [str(v) for v in raw.iloc[i].tolist()])
    cols = [str(v) for v in raw.iloc[hr].tolist()]
    emb = raw.iloc[hr + 1:].copy(); emb.columns = cols + [f"_{k}" for k in range(raw.shape[1] - len(cols))]
    emb = emb[["y_true", "y_prob", "pc1", "pc2"]].dropna(subset=["pc1", "pc2"])
    for c in emb.columns:
        emb[c] = pd.to_numeric(emb[c], errors="coerce")
    pr = next(i for i in range(len(raw)) if "cv_accuracy" in [str(v) for v in raw.iloc[i]])
    probe = raw.iloc[pr + 1:].copy()
    probe.columns = ["representation", "target", "cv_accuracy", "shuffled_baseline"] + [f"_{k}" for k in range(raw.shape[1] - 4)]
    probe = probe[["representation", "target", "cv_accuracy", "shuffled_baseline"]]
    probe = probe[probe["target"] == "neutralization"].dropna(subset=["cv_accuracy"])
    for c in ("cv_accuracy", "shuffled_baseline"):
        probe[c] = pd.to_numeric(probe[c], errors="coerce")
    return emb, probe.set_index("representation")


def panel_latent():
    emb, probe = _load_latent()
    sel = probe.loc["pair"]
    acc = float(sel["cv_accuracy"]); base = float(sel["shuffled_baseline"])
    ACC, NEG = "#2E6E8E", "#C0504D"
    classes = [(1.0, ACC, "Sensitive"), (0.0, NEG, "Resistant")]
    counts = {lab: int((emb["y_true"] == lab).sum()) for lab, _, _ in classes}
    fig = plt.figure(figsize=(96 * MM, 94 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[6.5, 1.0], height_ratios=[1.0, 6.5], wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax); axr = fig.add_subplot(gs[1, 1], sharey=ax)
    for lab, col, _ in classes:
        s = emb[emb["y_true"] == lab]
        ax.scatter(s["pc1"], s["pc2"], s=6.0, c=col, lw=0, alpha=0.6, zorder=3 if lab == 1.0 else 2)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.margins(0.04); ax.relim(); ax.autoscale_view()
    try:
        from scipy.stats import gaussian_kde
        for lab, col, _ in classes:
            s = emb[emb["y_true"] == lab]
            xy = np.vstack([s["pc1"].to_numpy(), s["pc2"].to_numpy()]); k = gaussian_kde(xy)
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            xx, yy = np.mgrid[xmin:xmax:160j, ymin:ymax:160j]
            zz = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=[0.30 * zz.max(), 0.65 * zz.max()], colors=[col], linewidths=(0.6, 0.9), alpha=0.85, zorder=4)
        for lab, col, name in classes:
            s = emb[emb["y_true"] == lab]
            kx = gaussian_kde(s["pc1"].to_numpy()); gx = np.linspace(*ax.get_xlim(), 200)
            axt.fill_between(gx, kx(gx), color=col, alpha=0.30, lw=0); axt.plot(gx, kx(gx), color=col, lw=0.9)
            ky = gaussian_kde(s["pc2"].to_numpy()); gy = np.linspace(*ax.get_ylim(), 200)
            axr.fill_betweenx(gy, ky(gy), color=col, alpha=0.30, lw=0); axr.plot(ky(gy), gy, color=col, lw=0.9)
    except Exception as e:
        print("  [latent] KDE skipped:", e)
    for lab, col, name in classes:
        s = emb[emb["y_true"] == lab]
        ax.annotate(f"{name}\n(n = {counts[lab]})", (s["pc1"].median(), s["pc2"].median()), color=col, fontsize=7.8,
                    fontweight="semibold", ha="center", va="center", linespacing=1.05,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.65, edgecolor="none"), zorder=6)
    for a in (axt, axr):
        a.axis("off")
    axt.set_ylim(bottom=0); axr.set_xlim(left=0)
    axt.set_title("CH505 pair embedding separates sensitive vs resistant", fontsize=8.6, pad=4)
    ax.text(0.5, -0.155, f"sensitive/resistant linear-probe accuracy = {acc:.2f}  (shuffled baseline {base:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.7, color="#555")
    _save(fig, "panel_B_latent")


def panel_probe():
    _, p = _load_latent()
    reps = ["z_env", "z_ab", "pair"]; labels = ["Env\n(z$_{env}$)", "Antibody\n(z$_{ab}$)", "Env + Ab\n(pair)"]
    acc = p.loc[reps, "cv_accuracy"].to_numpy(); base = p.loc[reps, "shuffled_baseline"].to_numpy()
    SIGNAL, NOSIG, MAJOR = "#2E6E8E", "#B9C0C7", "#C0504D"; MAJORITY = 667 / 947
    colors = [NOSIG if a <= b + 1e-9 else SIGNAL for a, b in zip(acc, base)]
    x = np.arange(len(reps))
    fig, ax = plt.subplots(figsize=(74 * MM, 70 * MM))
    ax.axhline(MAJORITY, color=MAJOR, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.bar(x, acc, width=0.62, color=colors, zorder=2)
    for xi, b, c in zip(x, base, colors):
        ax.plot([xi - 0.31, xi + 0.31], [b, b], color="white" if c == SIGNAL else "#3A3A3A", lw=1.3, ls=(0, (2.2, 1.6)), zorder=4)
    for xi, a in zip(x, acc):
        ax.annotate(f"{a:.2f}", xy=(xi, a), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, fontweight="semibold")
    ax.annotate("≈ chance", xy=(0, acc[0]), xytext=(0, -16), textcoords="offset points", ha="center", va="top", fontsize=6.0, color="#6b6b6b", fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Neutralisation probe accuracy  (5-fold CV)")
    ax.set_title("Antibody embedding carries the\nCH505 neutralisation signal", fontsize=8.4, pad=6)
    ax.plot([], [], color="#444", lw=1.1, ls=(0, (2.2, 1.6)), label="shuffled baseline")
    ax.plot([], [], color=MAJOR, lw=0.8, ls=(0, (1, 1.5)), label="majority class")
    ax.legend(loc="lower left", handlelength=1.5, borderaxespad=0.35, labelspacing=0.3, fontsize=6.3)
    _save(fig, "panel_C_probe")


def main():
    for fn in (panel_attention, panel_latent, panel_probe):
        try:
            fn()
        except Exception as e:
            print(f"[WARN] {fn.__name__} failed: {e}")
    print("Figure 5 (python panels) ->", OUT)


if __name__ == "__main__":
    main()
