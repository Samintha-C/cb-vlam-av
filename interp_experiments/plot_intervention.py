"""Render the intervention-curve JSON to a PDF.

Page 1: trajectory L2 vs #concepts intervened — one line per selection ordering
        (RAND / IMP / LCP), solid = residual ON, dashed = residual OFF. The
        RAND→IMP→LCP spread shows how much "most-important-first" helps; the
        ON↔OFF gap shows how much the residual explains away the concepts.
Page 2: the "most important concepts" linear-weight ranking.

Usage: python interp_experiments/plot_intervention.py intervention_curve.json --out curve.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

_BG, _PANEL, _GRID = "#1a1a2e", "#16213e", "#444466"
_COLORS = {"rand": "#9aa0b5", "imp": "tomato", "lcp": "mediumseagreen"}
_LABELS = {"rand": "RAND (baseline)", "imp": "IMP (most-important-first)",
           "lcp": "LCP (GT-error oracle)"}


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color="white", fontsize=10)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(True, color=_GRID, lw=0.4, alpha=0.5)
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)


def _curve_page(data, metric):
    x = data["x"]
    curves = data["curves"]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(_BG)
    for ordering in ("rand", "imp", "lcp"):
        if ordering not in curves:
            continue
        c = _COLORS[ordering]
        on = curves[ordering]["residual_on"][metric]
        off = curves[ordering]["residual_off"][metric]
        ax.plot(x, on, "-", color=c, lw=2.2, label=f"{_LABELS[ordering]} · resid ON")
        ax.plot(x, off, "--", color=c, lw=1.6, alpha=0.85,
                label=f"{_LABELS[ordering]} · resid OFF")
    base = curves["imp"]["residual_on"][metric][0]
    ax.axhline(base, color="white", lw=0.8, ls=":", alpha=0.6)
    ax.text(x[-1], base, "  no-intervention baseline", color="white",
            fontsize=8, va="bottom", ha="right")
    _style(ax, f"Concept intervention — {metric} vs #concepts corrected (GT)",
           "# concepts intervened (most-important-first)", f"{metric}  (m)")
    n = data["meta"]["n"]
    ax.legend(loc="upper right", fontsize=8.5, facecolor=_BG, edgecolor="none",
              labelcolor="white", ncol=1)
    ax.text(0.01, 0.01, f"split={data['meta']['split']}  n={n}  "
            f"rand_seeds={data['meta']['rand_seeds']}  imp={data['meta']['imp_kind']}",
            transform=ax.transAxes, color="#9aa0b5", fontsize=8)
    plt.tight_layout()
    return fig


def _importance_page(data):
    table = data["importance_table"]
    names = [r["concept"] for r in table][::-1]
    vals = [r["weight_norm"] for r in table][::-1]
    kinds = [r["kind"] for r in table][::-1]
    kcol = {"continuous": "#5dade2", "binary": "tomato", "categorical": "gold"}
    fig, ax = plt.subplots(figsize=(11, max(7, len(names) * 0.32)))
    fig.patch.set_facecolor(_BG)
    ax.barh(np.arange(len(names)), vals,
            color=[kcol.get(k, "white") for k in kinds], height=0.7)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color="white")
    _style(ax, "Concept importance — FinalPredictor linear-weight norm",
           "‖W[:, concept]‖  (contribution to trajectory)", "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=kcol[k]) for k in kcol]
    ax.legend(handles, list(kcol), loc="lower right", fontsize=8,
              facecolor=_BG, edgecolor="none", labelcolor="white")
    plt.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("curve_json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("./intervention_curve.pdf"))
    ap.add_argument("--metrics", nargs="+", default=["l2_avg", "ade_m"])
    args = ap.parse_args()

    data = json.loads(args.curve_json.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        for metric in args.metrics:
            pdf.savefig(_curve_page(data, metric), facecolor=_BG)
            plt.close()
        pdf.savefig(_importance_page(data), facecolor=_BG)
        plt.close()
    print(f"wrote {args.out}  ({len(args.metrics)} curve pages + importance)")


if __name__ == "__main__":
    main()
