"""
Accuracy vs steerability visualization for the 4×2 steer-matrix experiment.

Reads each gen_regression_steer_<cell>/intervention_curve.json and produces:
  Page 1 — Scatter: baseline L2 (accuracy) vs Δon / Δmean (steerability)
  Page 2 — Steer-gain bar chart (Δon + Δmean), sorted, both methods visible
  Page 3 — Overlaid intervention curves: % relative improvement vs fraction of concepts
            corrected (normalises away absolute L2 differences between cells)
  Page 4 — 4×2 grid (residual_dim rows × method cols): IMP + RAND + LCP curves,
            independent tight y-axis per panel so the signal is visible even when Δ is small
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
_METHOD_COLOR = {"jnt": "#5dade2", "ind": "#e67e22"}
_METHOD_LABEL = {"jnt": "JNT (standard)", "ind": "IND (teacher-forced)"}
_ORD_COLOR = {"rand": "#9aa0b5", "imp": "tomato", "lcp": "mediumseagreen"}
_ORD_LABEL = {"rand": "RAND (baseline)", "imp": "IMP (weight-norm)", "lcp": "LCP (GT-error oracle)"}
_DIMS = [8, 32, 64, 128]
CELLS = ["r8_jnt", "r8_ind", "r32_jnt", "r32_ind",
         "r64_jnt", "r64_ind", "r128_jnt", "r128_ind"]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_PANEL)
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=_GRID, lw=0.4, alpha=0.6)
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)


def _default_parse(cell: str) -> dict:
    """Cell name → {rdim, method} for the r{dim}_{method} steer-matrix naming."""
    rdim_str, method = cell.rsplit("_", 1)
    return {"rdim": int(rdim_str[1:]), "method": method}


def load_cells(runs_dir: Path, cells=None, parse=None,
               prefix: str = "gen_regression_steer_") -> dict:
    """Load per-cell intervention_curve.json into the row schema the pages read.

    Generic over cell naming: ``cells`` is the list of run-dir suffixes and
    ``parse`` maps each suffix to the metadata keys the pages need (default:
    the r{dim}_{method} steer-matrix scheme). Reused by plot_lsteer_sweep.py,
    which passes its own cells + a lambda_steer parse.
    """
    cells = CELLS if cells is None else cells
    parse = parse or _default_parse
    rows = {}
    for cell in cells:
        p = runs_dir / f"{prefix}{cell}" / "intervention_curve.json"
        if not p.exists():
            print(f"  SKIP {cell} — {p} not found")
            continue
        d = json.loads(p.read_text())
        meta = parse(cell)
        c = d["curves"]
        on   = c["imp"]["residual_on"]["l2_avg"]
        mean = c["imp"]["residual_mean"]["l2_avg"]
        off  = c["imp"]["residual_off"]["l2_avg"]
        rows[cell] = {
            "cell": cell, **meta,
            "x": d["x"],
            "n_concepts": d["meta"]["n_concepts"],
            "n": d["meta"]["n"],
            "curves_on":   {o: c[o]["residual_on"]["l2_avg"]   for o in ("rand", "imp", "lcp")},
            "curves_mean": {o: c[o]["residual_mean"]["l2_avg"] for o in ("rand", "imp", "lcp")},
            "baseline":         on[0],
            "steer_gain_on":    on[0]  - on[-1],
            "steer_gain_mean":  mean[0] - mean[-1],
            "concepts_only_l2": off[-1],
        }
    return rows


# ── Page 1: scatter ───────────────────────────────────────────────────────────

def _scatter_page(rows: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(_BG)
    _DIM_SZ = {8: 80, 32: 140, 64: 200, 128: 280}

    for ax, gain_key, sub in zip(
        axes,
        ["steer_gain_on",   "steer_gain_mean"],
        ["residual ON  (realistic)",
         "residual = train-mean  (fair concepts-only)"],
    ):
        ax.set_facecolor(_PANEL)
        for cell, r in rows.items():
            mc = _METHOD_COLOR[r["method"]]
            ax.scatter(r["baseline"], r[gain_key],
                       s=_DIM_SZ[r["rdim"]], color=mc,
                       edgecolors="white", linewidths=0.8, zorder=3, alpha=0.9)
            ax.annotate(cell, (r["baseline"], r[gain_key]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, color="white", zorder=4)

        # Axis limits: give 10% padding each side of the data
        xs = [r["baseline"]   for r in rows.values()]
        ys = [r[gain_key]     for r in rows.values()]
        xspan = max(xs) - min(xs) or 0.01
        yspan = max(ys) - min(ys) or 0.001
        ax.set_xlim(min(xs) - xspan * 0.15, max(xs) + xspan * 0.15)
        ax.set_ylim(min(ys) - yspan * 0.25, max(ys) + yspan * 0.35)
        ax.invert_xaxis()   # lower L2 = better accuracy → rightward

        # Legends: method (color) + size (rdim)
        for m, mc in _METHOD_COLOR.items():
            ax.scatter([], [], s=120, color=mc, edgecolors="white", lw=0.8,
                       label=_METHOD_LABEL[m])
        for dim, sz in _DIM_SZ.items():
            ax.scatter([], [], s=sz, color="grey", edgecolors="white", lw=0.8,
                       label=f"r={dim}", alpha=0.6)
        ax.legend(loc="upper left", fontsize=7.5, facecolor=_BG,
                  edgecolor=_GRID, labelcolor="white")
        _style(ax, title=f"[{sub}]",
               xlabel="Baseline L2_avg (m)   ← better accuracy",
               ylabel="Steer gain Δ (m)   ↑ more steerable")

    fig.suptitle("Accuracy vs Steerability — all 8 cells  "
                 "(IMP ordering, all concepts → GT)",
                 color="white", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Page 2: bar chart ─────────────────────────────────────────────────────────

def _bar_page(rows: dict):
    cells_sorted = sorted(rows, key=lambda c: rows[c]["steer_gain_on"], reverse=True)
    n = len(cells_sorted)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.patch.set_facecolor(_BG)
    x = np.arange(n)

    for ax, gain_key, ylabel in [
        (ax1, "steer_gain_on",   "Δon  (m)  [residual ON]"),
        (ax2, "steer_gain_mean", "Δmean  (m)  [residual = train-mean]"),
    ]:
        ax.set_facecolor(_PANEL)
        vals   = [rows[c][gain_key] for c in cells_sorted]
        colors = [_METHOD_COLOR[rows[c]["method"]] for c in cells_sorted]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)

        # Value labels on bars; scale offset to data range
        vrange = max(abs(v) for v in vals) or 0.001
        for bar, v in zip(bars, vals):
            offset = vrange * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + offset if v >= 0 else v - offset * 2,
                    f"{v:+.4f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(cells_sorted, rotation=25, ha="right",
                           fontsize=9, color="white")
        ax.axhline(0, color="white", lw=0.8, alpha=0.5)

        # Tight y with headroom
        ylo = min(min(vals) - vrange * 0.15, -0.001)
        yhi = max(max(vals) + vrange * 0.20,  0.001)
        ax.set_ylim(ylo, yhi)

        for m, mc in _METHOD_COLOR.items():
            ax.bar([], [], color=mc, label=_METHOD_LABEL[m])
        ax.legend(loc="upper right", fontsize=8, facecolor=_BG,
                  edgecolor=_GRID, labelcolor="white")
        _style(ax, ylabel=ylabel)

    fig.suptitle("Steerability gain per cell  (sorted by Δon, IMP ordering)\n"
                 "Δon = realistic steer gain  |  Δmean = fair concepts-only gain",
                 color="white", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Page 3: overlaid % relative improvement ──────────────────────────────────

def _overlay_page(rows: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor(_BG)
    _LW = {8: 1.0, 32: 1.6, 64: 2.2, 128: 2.8}

    for ax, ckey, sub in [
        (axes[0], "curves_on",   "residual ON  (realistic)"),
        (axes[1], "curves_mean", "residual = train-mean  (fair concepts-only)"),
    ]:
        ax.set_facecolor(_PANEL)
        ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.5)

        for cell, r in rows.items():
            method = r["method"]
            C      = r["n_concepts"]
            frac   = [m / C for m in r["x"]]
            # Normalise each curve by ITS OWN m=0 value (residual-on and
            # residual-mean have different baselines — using on[0] for the mean
            # curve gives a meaningless ~-300% offset).
            base   = r[ckey]["imp"][0]
            pct    = [(base - v) / base * 100 for v in r[ckey]["imp"]]
            ax.plot(frac, pct,
                    ls="-" if method == "ind" else "--",
                    color=_METHOD_COLOR[method], lw=_LW[r["rdim"]],
                    label=f"{cell}  (base={base:.3f}m)")

        # Tight y: zoom around [-0.5%, max_pct+0.3%], floor at -0.5
        all_pcts = [(rows[c][ckey]["imp"][0] - v) / rows[c][ckey]["imp"][0] * 100
                    for c in rows for v in rows[c][ckey]["imp"]]
        ylo = min(min(all_pcts) - 0.1, -0.2)
        yhi = max(max(all_pcts) + 0.15, 0.1)
        ax.set_ylim(ylo, yhi)

        for m, mc in _METHOD_COLOR.items():
            ax.plot([], [], color=mc,
                    ls="-" if m == "ind" else "--",
                    lw=2, label=_METHOD_LABEL[m])
        ax.legend(loc="lower right", fontsize=6.5, facecolor=_BG,
                  edgecolor=_GRID, labelcolor="white", ncol=2)
        _style(ax, title=sub,
               xlabel="Fraction of concepts intervened  (IMP ordering, 0 = none → 1 = all GT)",
               ylabel="% relative L2 improvement   ↑ more steerable")

    fig.suptitle("Intervention curves — % relative improvement  (IMP, residual_on)\n"
                 "solid=IND  dashed=JNT  thickness ∝ residual_dim  "
                 "— normalised so cells with different absolute L2 are comparable",
                 color="white", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ── Page 4: 4×2 grid, tight independent y-axes ───────────────────────────────

def _grid_page(rows: dict):
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.patch.set_facecolor(_BG)

    for ri, rdim in enumerate(_DIMS):
        for ci, method in enumerate(["jnt", "ind"]):
            cell = f"r{rdim}_{method}"
            ax = axes[ri][ci]
            ax.set_facecolor(_PANEL)
            if cell not in rows:
                ax.text(0.5, 0.5, f"{cell}\n(no data)",
                        transform=ax.transAxes, ha="center", va="center",
                        color="white", fontsize=10)
                continue

            r = rows[cell]
            all_y = [v for o in ("rand", "imp", "lcp")
                       for v in r["curves_on"][o]]
            ylo, yhi = min(all_y), max(all_y)
            span = yhi - ylo
            # At least 5mm of range so a completely flat line still has room
            pad = max(span * 0.12, 0.005)
            ax.set_ylim(ylo - pad, yhi + pad)

            for ordering in ("rand", "imp", "lcp"):
                ax.plot(r["x"], r["curves_on"][ordering],
                        color=_ORD_COLOR[ordering], lw=1.8,
                        label=_ORD_LABEL[ordering])

            # Annotate Δon (IMP) in top-right corner
            delta = r["curves_on"]["imp"][0] - r["curves_on"]["imp"][-1]
            pct   = delta / r["baseline"] * 100
            ax.text(0.97, 0.96, f"Δon={delta:+.4f}m\n({pct:+.2f}%)",
                    transform=ax.transAxes, ha="right", va="top",
                    color="white", fontsize=8,
                    bbox=dict(facecolor=_BG, alpha=0.75, edgecolor="none", pad=2))
            # Baseline dotted line
            ax.axhline(r["baseline"], color="white", lw=0.7, ls=":", alpha=0.4)

            _style(ax,
                   title=f"{cell}  [base={r['baseline']:.3f}m]",
                   xlabel="# concepts intervened" if ri == 3 else "",
                   ylabel="L2_avg (m)" if ci == 0 else "")
            if ri == 0 and ci == 0:
                ax.legend(loc="upper right", fontsize=7, facecolor=_BG,
                          edgecolor=_GRID, labelcolor="white")

    fig.suptitle("4×2 steer matrix — RAND / IMP / LCP  (residual ON, independent tight y-axes)\n"
                 "Δon annotated per panel; dotted = no-intervention baseline",
                 color="white", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/runs"))
    ap.add_argument("--out", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/runs/steer_matrix_viz.pdf"))
    args = ap.parse_args()

    print(f"Loading cells from {args.runs_dir} ...")
    rows = load_cells(args.runs_dir)
    if not rows:
        print("No cells found — exiting"); return
    print(f"  loaded {len(rows)}/{len(CELLS)}: {list(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        pdf.savefig(_scatter_page(rows),  facecolor=_BG); plt.close()
        pdf.savefig(_bar_page(rows),      facecolor=_BG); plt.close()
        pdf.savefig(_overlay_page(rows),  facecolor=_BG); plt.close()
        pdf.savefig(_grid_page(rows),     facecolor=_BG); plt.close()
    print(f"wrote {args.out}  (4 pages)")


if __name__ == "__main__":
    main()
