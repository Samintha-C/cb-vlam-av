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


# residual modes present in the JSON → (panel title, key). residual_mean is the
# fair concepts-only ablation (train-mean r); residual_off (r=0) is off-distribution.
_MODES = [("residual_on", "residual ON (realistic)"),
          ("residual_mean", "residual = train-mean  (fair concepts-only)"),
          ("residual_off", "residual OFF = 0  (off-distribution)")]


def _panel(ax, x, curves, metric, mode, show_baseline):
    for ordering in ("rand", "imp", "lcp"):
        if ordering not in curves or mode not in curves[ordering]:
            continue
        ax.plot(x, curves[ordering][mode][metric], "-", color=_COLORS[ordering],
                lw=2.2, label=_LABELS[ordering])
    if show_baseline:
        base = curves["imp"][mode][metric][0]
        ax.axhline(base, color="white", lw=0.8, ls=":", alpha=0.6)
        ax.text(x[-1], base, " baseline", color="white", fontsize=8,
                va="bottom", ha="right")


def _curve_page(data, metric):
    x = data["x"]
    curves = data["curves"]
    # Only plot residual modes that are actually present in this run.
    modes = [(k, t) for (k, t) in _MODES
             if any(k in curves[o] for o in curves)]
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6.5))
    if len(modes) == 1:
        axes = [axes]
    fig.patch.set_facecolor(_BG)
    # Each panel auto-scales independently — on (~0.4) and off (~6.5) no longer
    # share a y-axis, so the near-flat steering signal is actually visible.
    for ax, (key, title) in zip(axes, modes):
        _panel(ax, x, curves, metric, key, show_baseline=True)
        _style(ax, title, "# concepts intervened (most-important-first)", f"{metric}  (m)")
        ax.legend(loc="best", fontsize=8.5, facecolor=_BG, edgecolor="none",
                  labelcolor="white")
    m = data["meta"]
    fig.suptitle(f"Concept intervention — {metric} vs #concepts corrected (GT)   "
                 f"[split={m['split']}  n={m['n']}  rand_seeds={m['rand_seeds']}  "
                 f"imp={m['imp_kind']}]", color="white", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


_KCOL = {"continuous": "#5dade2", "binary": "tomato", "categorical": "gold"}


def _importance_page(data):
    # Column-fair ‖W‖/√k (so the 3-slot categorical isn't √3-inflated).
    table = sorted(data["importance_table"], key=lambda r: r["weight_norm_per_col"])
    names = [r["concept"] for r in table]
    vals = [r["weight_norm_per_col"] for r in table]
    kinds = [r["kind"] for r in table]
    fig, ax = plt.subplots(figsize=(11, max(7, len(names) * 0.32)))
    fig.patch.set_facecolor(_BG)
    ax.barh(np.arange(len(names)), vals,
            color=[_KCOL.get(k, "white") for k in kinds], height=0.7)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color="white")
    _style(ax, "Concept importance — column-fair linear-weight norm  ‖W‖/√k",
           "‖W[:, concept]‖ / √(#cols)  (contribution to trajectory)", "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_KCOL[k]) for k in _KCOL]
    ax.legend(handles, list(_KCOL), loc="lower right", fontsize=8,
              facecolor=_BG, edgecolor="none", labelcolor="white")
    plt.tight_layout()
    return fig


def _per_concept_page(data):
    """Single-intervention ΔL2 per concept + ΔL2-vs-prediction-error scatter."""
    pc = data.get("per_concept")
    if not pc:
        return None
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(18, max(7, len(pc) * 0.32)),
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor(_BG)

    # left: ΔL2 bars (sorted helpful→harmful); green = helps (ΔL2<0), red = hurts.
    rows = sorted(pc, key=lambda r: r["delta_l2"], reverse=True)
    names = [r["concept"] for r in rows]
    dl = [r["delta_l2"] for r in rows]
    axb.barh(np.arange(len(rows)), dl,
             color=["mediumseagreen" if v < 0 else "tomato" for v in dl], height=0.7)
    axb.set_yticks(np.arange(len(rows)))
    axb.set_yticklabels(names, fontsize=7.5, color="white")
    axb.axvline(0, color="white", lw=0.8, alpha=0.6)
    _style(axb, "Per-concept single intervention (residual on)",
           "ΔL2_avg vs no-intervention  (m)   [<0 helps, >0 hurts]", "")

    # right: does prediction error predict steering effect? (expect NO correlation
    # if the bottleneck is non-causal — that's the finding).
    pe = np.array([r["pred_error"] for r in pc])
    dla = np.array([r["delta_l2"] for r in pc])
    kinds = [r["kind"] for r in pc]
    axs.scatter(pe, dla, c=[_KCOL.get(k, "white") for k in kinds], s=40)
    axs.axhline(0, color="white", lw=0.8, alpha=0.6)
    if len(pe) > 1 and pe.std() > 0:
        rho = float(np.corrcoef(pe, dla)[0, 1])
        axs.set_title(f"ΔL2 vs prediction error   (corr={rho:+.2f})",
                      color="white", fontsize=11, pad=8)
    _style(axs, axs.get_title() or "ΔL2 vs prediction error",
           "prediction error |GT−pred| (activation space)", "ΔL2_avg  (m)")
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
        pc_page = _per_concept_page(data)
        if pc_page is not None:
            pdf.savefig(pc_page, facecolor=_BG)
            plt.close()
        pdf.savefig(_importance_page(data), facecolor=_BG)
        plt.close()
    print(f"wrote {args.out}  ({len(args.metrics)} curve pages + per-concept + importance)")


if __name__ == "__main__":
    main()
