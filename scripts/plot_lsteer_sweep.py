"""Accuracy vs steerability visualization for the L_steer weight sweep.

The 1-D analogue of plot_steer_matrix.py: all cells share r128 + IND, the only
moving variable is lambda_steer (the CB-SAE-style intervention-consistency loss
weight). Reads each gen_regression_steer_r128_ind_lsteer_l<tag>/intervention_curve.json
and produces the SAME four-page style as steer_matrix_viz.pdf, keyed on
lambda_steer instead of residual_dim × method:

  Page 1 — Scatter: baseline L2 (accuracy) vs Δon / Δmean (steerability), colour = λ
  Page 2 — Steer-gain bar chart (Δon + Δmean), ordered by λ
  Page 3 — Overlaid intervention curves: % relative improvement vs fraction corrected
  Page 4 — One panel per λ: RAND / IMP / LCP curves, independent tight y-axis

Loading + the JSON schema are reused from plot_steer_matrix.py (load_cells, _style,
palette constants) so the two stay in lockstep.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_steer_matrix import (
    load_cells, _style, _BG, _PANEL, _GRID, _ORD_COLOR, _ORD_LABEL)

# Sweep cells, ascending λ. Suffix tag l<int>p<frac> ↔ lambda_steer value.
SWEEP = [
    ("r128_ind_lsteer_l0p0", 0.0),
    ("r128_ind_lsteer_l0p5", 0.5),
    ("r128_ind_lsteer_l1p0", 1.0),
    ("r128_ind_lsteer_l2p0", 2.0),
    ("r128_ind_lsteer_l4p0", 4.0),
]
_LAMBDA = dict(SWEEP)
CELLS = [c for c, _ in SWEEP]


def _parse(cell: str) -> dict:
    """Cell suffix → metadata; the sweep varies lambda_steer at fixed r128+IND."""
    return {"rdim": 128, "method": "ind", "lambda_steer": _LAMBDA[cell],
            "tag": cell.split("_")[-1]}


def _lambda_colors(rows: dict) -> dict:
    """Evenly-sampled viridis colour per cell, ordered by λ (0 = dark → high = bright)."""
    order = sorted(rows, key=lambda c: rows[c]["lambda_steer"])
    n = max(len(order) - 1, 1)
    return {c: cm.viridis(0.12 + 0.80 * i / n) for i, c in enumerate(order)}


def _lbl(r: dict) -> str:
    return f"λ={r['lambda_steer']:g}"


# ── Page 1: scatter ───────────────────────────────────────────────────────────

def _scatter_page(rows: dict, colors: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(_BG)

    for ax, gain_key, sub in zip(
        axes, ["steer_gain_on", "steer_gain_mean"],
        ["residual ON  (realistic)", "residual = train-mean  (fair concepts-only)"],
    ):
        ax.set_facecolor(_PANEL)
        for cell, r in rows.items():
            ax.scatter(r["baseline"], r[gain_key], s=200, color=colors[cell],
                       edgecolors="white", linewidths=0.9, zorder=3)
            ax.annotate(_lbl(r), (r["baseline"], r[gain_key]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=8, color="white", zorder=4)
        xs = [r["baseline"] for r in rows.values()]
        ys = [r[gain_key] for r in rows.values()]
        xspan = (max(xs) - min(xs)) or 0.01
        yspan = (max(ys) - min(ys)) or 0.001
        ax.set_xlim(min(xs) - xspan * 0.15, max(xs) + xspan * 0.15)
        ax.set_ylim(min(ys) - yspan * 0.25, max(ys) + yspan * 0.35)
        ax.invert_xaxis()
        ax.axhline(0, color="white", lw=0.7, ls=":", alpha=0.5)
        _style(ax, title=f"[{sub}]",
               xlabel="Baseline L2_avg (m)   ← better accuracy",
               ylabel="Steer gain Δ (m)   ↑ more steerable")

    fig.suptitle("L_steer sweep — accuracy vs steerability  (r128+IND, "
                 "IMP ordering, all concepts → GT)", color="white", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Page 2: bar chart ─────────────────────────────────────────────────────────

def _bar_page(rows: dict, colors: dict):
    cells = sorted(rows, key=lambda c: rows[c]["lambda_steer"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.patch.set_facecolor(_BG)
    x = np.arange(len(cells))

    for ax, gain_key, ylabel in [
        (ax1, "steer_gain_on",   "Δon  (m)  [residual ON]"),
        (ax2, "steer_gain_mean", "Δmean  (m)  [residual = train-mean]"),
    ]:
        ax.set_facecolor(_PANEL)
        vals = [rows[c][gain_key] for c in cells]
        bars = ax.bar(x, vals, color=[colors[c] for c in cells],
                      edgecolor="white", linewidth=0.6)
        vrange = max(abs(v) for v in vals) or 0.001
        for bar, v in zip(bars, vals):
            off = vrange * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + off if v >= 0 else v - off * 2, f"{v:+.4f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([_lbl(rows[c]) for c in cells], fontsize=9, color="white")
        ax.axhline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_ylim(min(min(vals) - vrange * 0.15, -0.001),
                    max(max(vals) + vrange * 0.20, 0.001))
        _style(ax, ylabel=ylabel)

    fig.suptitle("Steerability gain vs L_steer weight  (r128+IND, IMP ordering)\n"
                 "Δon = realistic steer gain  |  Δmean = fair concepts-only gain",
                 color="white", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Page 3: overlaid % relative improvement ──────────────────────────────────

def _overlay_page(rows: dict, colors: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor(_BG)

    for ax, ckey, sub in [
        (axes[0], "curves_on",   "residual ON  (realistic)"),
        (axes[1], "curves_mean", "residual = train-mean  (fair concepts-only)"),
    ]:
        ax.set_facecolor(_PANEL)
        ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.5)
        for cell in sorted(rows, key=lambda c: rows[c]["lambda_steer"]):
            r = rows[cell]
            frac = [m / r["n_concepts"] for m in r["x"]]
            base = r[ckey]["imp"][0]
            pct = [(base - v) / base * 100 for v in r[ckey]["imp"]]
            ax.plot(frac, pct, color=colors[cell], lw=2.2,
                    label=f"{_lbl(r)}  (base={base:.3f}m)")
        all_pcts = [(rows[c][ckey]["imp"][0] - v) / rows[c][ckey]["imp"][0] * 100
                    for c in rows for v in rows[c][ckey]["imp"]]
        ax.set_ylim(min(min(all_pcts) - 0.1, -0.2), max(max(all_pcts) + 0.15, 0.1))
        ax.legend(loc="lower right", fontsize=7.5, facecolor=_BG,
                  edgecolor=_GRID, labelcolor="white")
        _style(ax, title=sub,
               xlabel="Fraction of concepts intervened  (IMP, 0 = none → 1 = all GT)",
               ylabel="% relative L2 improvement   ↑ more steerable")

    fig.suptitle("Intervention curves — % relative improvement  (IMP, residual_on)\n"
                 "colour = L_steer weight  — normalised so different absolute L2 are comparable",
                 color="white", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ── Page 4: one panel per λ, tight independent y-axes ────────────────────────

def _grid_page(rows: dict):
    cells = sorted(rows, key=lambda c: rows[c]["lambda_steer"])
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.6 * nrow), squeeze=False)
    fig.patch.set_facecolor(_BG)

    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_facecolor(_PANEL)
        if idx >= len(cells):
            ax.axis("off"); continue
        r = rows[cells[idx]]
        all_y = [v for o in ("rand", "imp", "lcp") for v in r["curves_on"][o]]
        span = max(all_y) - min(all_y)
        pad = max(span * 0.12, 0.005)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        for o in ("rand", "imp", "lcp"):
            ax.plot(r["x"], r["curves_on"][o], color=_ORD_COLOR[o], lw=1.8,
                    label=_ORD_LABEL[o])
        delta = r["curves_on"]["imp"][0] - r["curves_on"]["imp"][-1]
        pct = delta / r["baseline"] * 100
        ax.text(0.97, 0.96, f"Δon={delta:+.4f}m\n({pct:+.2f}%)",
                transform=ax.transAxes, ha="right", va="top", color="white",
                fontsize=8, bbox=dict(facecolor=_BG, alpha=0.75, edgecolor="none", pad=2))
        ax.axhline(r["baseline"], color="white", lw=0.7, ls=":", alpha=0.4)
        _style(ax, title=f"{_lbl(r)}  [base={r['baseline']:.3f}m]",
               xlabel="# concepts intervened" if idx // ncol == nrow - 1 else "",
               ylabel="L2_avg (m)" if idx % ncol == 0 else "")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7, facecolor=_BG,
                      edgecolor=_GRID, labelcolor="white")

    fig.suptitle("L_steer sweep — RAND / IMP / LCP per λ  (residual ON, independent tight y-axes)\n"
                 "Δon annotated per panel; dotted = no-intervention baseline",
                 color="white", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/runs"))
    ap.add_argument("--out", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/runs/lsteer_sweep_viz.pdf"))
    args = ap.parse_args()

    print(f"Loading L_steer sweep cells from {args.runs_dir} ...")
    rows = load_cells(args.runs_dir, cells=CELLS, parse=_parse)
    if not rows:
        print("No cells found — exiting"); return
    print(f"  loaded {len(rows)}/{len(CELLS)}: {list(rows)}")

    colors = _lambda_colors(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        pdf.savefig(_scatter_page(rows, colors), facecolor=_BG); plt.close()
        pdf.savefig(_bar_page(rows, colors),     facecolor=_BG); plt.close()
        pdf.savefig(_overlay_page(rows, colors), facecolor=_BG); plt.close()
        pdf.savefig(_grid_page(rows),            facecolor=_BG); plt.close()
    print(f"wrote {args.out}  (4 pages)")


if __name__ == "__main__":
    main()
