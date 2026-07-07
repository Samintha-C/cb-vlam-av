"""
CB-VLAM-AV results visualisation.

Pages:
  1 — Binary concept projection: AUROC + F1 (all 18)
  2 — Continuous concept projection: R² + MAE (all 8) + categorical
  3 — Trajectory accuracy: L2 curve vs baselines + ADE/FDE
  4 — Steerability: head-weight importance + per-concept ΔL2 diagnostic
  5 — Steer matrix: accuracy vs Δon scatter + Δon bar (optional, needs --steer_summary)

Usage:
  python scripts/plot_results.py \
    --eval    outputs/genreg/evals/eval_stp3.json \
    --interp  outputs/genreg/evals/intervention_curve_new.json \
    [--steer_summary  outputs/genreg/evals/intervention_steer_matrix_summary.json] \
    --out     outputs/genreg/evals/results.pdf
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
_KIND_COLOR = {"binary": "#5dade2", "continuous": "#e67e22", "categorical": "#a29bfe"}
_POS = "#2ecc71"   # positive R² / good
_NEG = "#e74c3c"   # negative R² / bad


def _style(ax, title="", xlabel="", ylabel="", legend=False):
    ax.set_facecolor(_PANEL)
    if title:
        ax.set_title(title, color="white", fontsize=11, pad=8, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=_GRID, lw=0.4, alpha=0.6, axis="x")
    ax.grid(False, axis="y")
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)
    if legend:
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor="white")


def _hbar(ax, names, vals, colors, vline=None, xlabel="", title="", annotations=None):
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.65, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color="white")
    if vline is not None:
        ax.axvline(vline, color="white", lw=0.8, ls="--", alpha=0.5)
    if annotations:
        for i, (v, ann) in enumerate(zip(vals, annotations)):
            ax.text(max(v, 0) + 0.005, i, f" {ann}", va="center",
                    fontsize=7, color="white", alpha=0.8)
    _style(ax, title=title, xlabel=xlabel)


# ── Page 1: Binary concepts ───────────────────────────────────────────────────

def _page_binary(binary_data, fig_kw):
    pc = binary_data["per_concept"]
    rows = sorted(pc.items(), key=lambda x: x[1]["auroc"])
    names   = [r[0].replace("_", " ") for r in rows]
    aurocs  = [r[1]["auroc"] for r in rows]
    f1s     = [r[1]["f1"]    for r in rows]
    pos     = [r[1]["pos_rate"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(17, 8), **fig_kw)
    fig.patch.set_facecolor(_BG)

    # AUROC
    colors = [_POS if v >= 0.8 else (_KIND_COLOR["binary"] if v >= 0.7 else _NEG)
              for v in aurocs]
    _hbar(axes[0], names, aurocs, colors, vline=0.8,
          xlabel="AUROC", title="Binary — AUROC (all 18)",
          annotations=[f"{v:.3f}" for v in aurocs])
    axes[0].set_xlim(0, 1.08)

    # F1
    f1_colors = [_POS if v >= 0.5 else (_KIND_COLOR["binary"] if v >= 0.3 else _NEG)
                 for v in f1s]
    _hbar(axes[1], names, f1s, f1_colors, vline=0.5,
          xlabel="F1", title="Binary — F1",
          annotations=[f"{v:.3f}" for v in f1s])
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels([""] * len(names))
    axes[1].set_xlim(0, 1.08)

    # Positive rate (class imbalance context)
    _hbar(axes[2], names, pos, [_GRID] * len(names), vline=0.5,
          xlabel="Positive rate", title="Class balance (pos rate)",
          annotations=[f"{v:.2f}" for v in pos])
    axes[2].set_yticks(np.arange(len(names)))
    axes[2].set_yticklabels([""] * len(names))
    axes[2].set_xlim(0, 1.08)

    macro = binary_data.get("macro_auroc", 0)
    fig.suptitle(f"Concept Projection — Binary (18 concepts)   macro AUROC = {macro:.3f}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Page 2: Continuous + categorical ─────────────────────────────────────────

def _page_continuous(cont_data, cat_data, fig_kw):
    pc = cont_data["per_concept"]
    rows = sorted(pc.items(), key=lambda x: x[1]["r2"], reverse=True)
    names = [r[0].replace("_", " ") for r in rows]
    r2s   = [r[1]["r2"]  for r in rows]
    maes  = [r[1]["mae"] for r in rows]

    fig = plt.figure(figsize=(16, 7), **fig_kw)
    fig.patch.set_facecolor(_BG)
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    ax_r2  = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])
    ax_cat = fig.add_subplot(gs[0, 2])

    # R²
    r2_colors = [_POS if v > 0 else _NEG for v in r2s]
    _hbar(ax_r2, names, r2s, r2_colors, vline=0,
          xlabel="R²", title="Continuous — R²",
          annotations=[f"{v:+.3f}" for v in r2s])
    xmax = max(max(r2s) * 1.15, 0.05)
    xmin = min(min(r2s) * 1.15, -0.05)
    ax_r2.set_xlim(xmin, xmax)

    # MAE
    _hbar(ax_mae, names, maes, [_KIND_COLOR["continuous"]] * len(names),
          xlabel="MAE", title="Continuous — MAE",
          annotations=[f"{v:.3f}" for v in maes])
    ax_mae.set_yticks(np.arange(len(names)))
    ax_mae.set_yticklabels([""] * len(names))

    # Categorical
    cat_pc = cat_data["per_concept"]
    cat_names  = [k.replace("_", " ") for k in cat_pc]
    cat_acc    = [v["accuracy"]  for v in cat_pc.values()]
    cat_f1     = [v["macro_f1"]  for v in cat_pc.values()]
    ax_cat.set_facecolor(_PANEL)
    x = np.arange(len(cat_names))
    w = 0.35
    ax_cat.bar(x - w/2, cat_acc, w, color=_KIND_COLOR["categorical"],
               label="Accuracy", edgecolor="none")
    ax_cat.bar(x + w/2, cat_f1,  w, color="#fd79a8",
               label="Macro F1", edgecolor="none")
    ax_cat.axhline(1/3, color="white", lw=0.8, ls="--", alpha=0.5,
                   label="Chance (0.333)")
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels(cat_names, color="white", fontsize=8)
    ax_cat.set_ylim(0, 0.75)
    for xi, (a, f) in enumerate(zip(cat_acc, cat_f1)):
        ax_cat.text(xi - w/2, a + 0.015, f"{a:.3f}", ha="center",
                    fontsize=8.5, color="white")
        ax_cat.text(xi + w/2, f + 0.015, f"{f:.3f}", ha="center",
                    fontsize=8.5, color="white")
    _style(ax_cat, title="Categorical (3-class)", ylabel="Score", legend=True)
    ax_cat.grid(True, color=_GRID, lw=0.4, alpha=0.6, axis="y")
    ax_cat.grid(False, axis="x")

    macro_r2 = cont_data.get("macro_r2", 0)
    fig.suptitle(f"Concept Projection — Continuous (8 concepts)  macro R² = {macro_r2:.3f}  "
                 f"(excl. low-variance)   +  Categorical",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Page 3: Trajectory accuracy — comparison table ───────────────────────────

_OURS_COLOR = "#e67e22"

# (section, name, ade, fde, l2_1s, l2_2s, l2_3s, l2_avg)
# ade/fde = None where not reported / convention-incompatible
_TABLE_DATA = [
    ("Closed-source API-only",     None, None, None, None, None, None, None),
    (None, "GPT-4o",               None, None, 0.28, 0.93, 2.02, 1.07),
    ("Open-source Generalist VLMs",None, None, None, None, None, None, None),
    (None, "Qwen-2.5-VL-7B-Instruct", None, None, 0.46, 1.33, 2.55, 1.45),
    ("Training-based Driving Specialists", None, None, None, None, None, None, None),
    (None, "UniAD",                None, None, 0.42, 0.64, 0.91, 0.66),
    (None, "VAD",                  None, None, 0.17, 0.34, 0.60, 0.37),
    ("Specialized Driving Models", None, None, None, None, None, None, None),
    (None, "DriveVLM",             None, None, 0.18, 0.34, 0.68, 0.40),
    (None, "Impromptu 7B +Impromptu+nuScenes", None, None, 0.13, 0.27, 0.53, 0.30),
]


def _page_trajectory(traj_data, fig_kw):
    from matplotlib.patches import Rectangle

    rows = list(_TABLE_DATA)
    rows.append((None, "CB-VLAM-AV (ours)",
                 traj_data["ade_m"], traj_data["fde_m"],
                 traj_data["l2_1s"], traj_data["l2_2s"],
                 traj_data["l2_3s"], traj_data["l2_avg"]))
    ours_idx = len(rows) - 1   # track which row is ours (0-based in rows list)

    n_rows = len(rows) + 1     # +1 for column header
    row_h  = 0.44
    fig_h  = n_rows * row_h + 0.9
    fig, ax = plt.subplots(figsize=(13, fig_h), **fig_kw)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n_rows + 0.5)

    # Column x positions (right-aligned numbers)
    CX = {"method": 0.15, "ade": 5.2, "fde": 6.2,
          "l2_1s": 7.1, "l2_2s": 8.0, "l2_3s": 8.9, "l2_avg": 9.85}
    TABLE_RIGHT = 10.0
    NUM_COLS = ["ade", "fde", "l2_1s", "l2_2s", "l2_3s", "l2_avg"]
    NUM_DIVIDER = CX["ade"] - 0.25   # vertical line separating name from numbers

    def _row_y(i):
        return n_rows - i - 0.5

    # ── Column header ─────────────────────────────────────────────────────────
    hdr_y = _row_y(0)
    ax.add_patch(Rectangle((0, hdr_y - 0.5), TABLE_RIGHT, 1.0,
                            color="#0a0f1a", zorder=1))
    ax.text(CX["method"], hdr_y, "Method", color="#ccccdd", fontsize=9,
            fontweight="bold", va="center", ha="left")
    # Sub-headers
    for cx, lbl in [(CX["ade"], "ADE"), (CX["fde"], "FDE"),
                    (CX["l2_1s"], "1s"), (CX["l2_2s"], "2s"),
                    (CX["l2_3s"], "3s"), (CX["l2_avg"], "Avg.")]:
        ax.text(cx, hdr_y, lbl, color="#ccccdd", fontsize=9,
                fontweight="bold", va="center", ha="right")
    # Group labels above sub-headers
    ax.text((CX["ade"] + CX["fde"]) / 2, hdr_y + 0.62,
            "Displacement (m)  ↓", color="#aaaacc", fontsize=7.5,
            va="center", ha="center")
    ax.text((CX["l2_1s"] + CX["l2_avg"]) / 2, hdr_y + 0.62,
            "L2 Error (m)  ↓", color="#aaaacc", fontsize=7.5,
            va="center", ha="center")
    ax.axhline(hdr_y - 0.5, color="#555577", lw=1.2, zorder=2)

    # ── Data rows ─────────────────────────────────────────────────────────────
    ri = 1
    data_ri = 0   # count only data rows (not section headers)
    for row in rows:
        sec = row[0]
        y   = _row_y(ri)

        if sec is not None:
            ax.add_patch(Rectangle((0, y - 0.5), TABLE_RIGHT, 1.0,
                                   color="#0d1117", zorder=1))
            ax.text(CX["method"], y, sec, color="#9999bb", fontsize=8,
                    fontstyle="italic", va="center", ha="left")
            ax.axhline(y - 0.5, color="#333355", lw=0.6, zorder=2)
        else:
            _, name, ade, fde, v1, v2, v3, vavg = row
            is_ours = (ri - 1 == ours_idx)   # ri-1 accounts for header offset

            # Background: slightly lighter for ours, subtle stripe for others
            if is_ours:
                ax.add_patch(Rectangle((0, y - 0.5), TABLE_RIGHT, 1.0,
                                       color="#1f2d1f", zorder=1))
            elif data_ri % 2 == 0:
                ax.add_patch(Rectangle((0, y - 0.5), TABLE_RIGHT, 1.0,
                                       color="#0e1420", alpha=0.6, zorder=1))

            tc = "white"
            fs = 8.5

            ax.text(CX["method"], y, name, color=tc, fontsize=fs,
                    va="center", ha="left")

            vals = [ade, fde, v1, v2, v3, vavg]
            for col, v in zip(NUM_COLS, vals):
                txt = f"{v:.2f}" if v is not None else "—"
                ax.text(CX[col], y, txt, color=tc, fontsize=fs,
                        va="center", ha="right")

            ax.axhline(y - 0.5, color="#222240", lw=0.3, zorder=2)
            data_ri += 1

        ri += 1

    # Outer border + vertical dividers
    ax.add_patch(Rectangle((0, -0.5), TABLE_RIGHT, n_rows + 1.0,
                            fill=False, edgecolor="#555577", lw=1.0, zorder=3))
    ax.axvline(NUM_DIVIDER, color="#333355", lw=0.6, zorder=2)
    # Light divider between ADE/FDE and L2 columns
    ax.axvline(CX["l2_1s"] - 0.2, color="#222240", lw=0.4, zorder=2)

    n = traj_data["n"]
    fig.suptitle(
        f"Trajectory Accuracy — Open-loop nuScenes ST-P3  (n={n}, Impromptu q7 L2 convention)",
        color="white", fontsize=11, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


# ── Page 4: Steerability — importance + per-concept ΔL2 ─────────────────────

def _page_steerability(interp_data, fig_kw):
    imp_table = sorted(interp_data["importance_table"],
                       key=lambda r: r["weight_norm_per_col"])
    pc_table  = sorted(interp_data["per_concept"],
                       key=lambda r: r["delta_l2"])  # most helpful first

    fig, (ax_imp, ax_dl, ax_sc) = plt.subplots(1, 3, figsize=(19, 9), **fig_kw)
    fig.patch.set_facecolor(_BG)

    # ① Importance (column-fair ‖W‖/√k)
    names_imp = [r["concept"].replace("_", " ") for r in imp_table]
    vals_imp  = [r["weight_norm_per_col"] for r in imp_table]
    kinds_imp = [r["kind"] for r in imp_table]
    _hbar(ax_imp, names_imp, vals_imp,
          [_KIND_COLOR.get(k, "white") for k in kinds_imp],
          xlabel="‖W‖ / √k", title="Head concept importance  (‖W‖/√k)",
          annotations=[f"{v:.3f}" for v in vals_imp])
    # Kind legend
    for kind, col in _KIND_COLOR.items():
        ax_imp.barh([], [], color=col, label=kind)
    ax_imp.legend(fontsize=7.5, facecolor=_BG, edgecolor=_GRID, labelcolor="white",
                  loc="lower right")

    # ② Per-concept ΔL2 (single intervention, residual on)
    names_dl = [r["concept"].replace("_", " ") for r in pc_table]
    vals_dl  = [r["delta_l2"] for r in pc_table]
    colors_dl = [_POS if v < 0 else _NEG for v in vals_dl]
    y = np.arange(len(names_dl))
    ax_dl.barh(y, vals_dl, color=colors_dl, height=0.65, edgecolor="none")
    ax_dl.set_yticks(y)
    ax_dl.set_yticklabels(names_dl, fontsize=8, color="white")
    ax_dl.axvline(0, color="white", lw=0.8, alpha=0.6)
    for i, v in enumerate(vals_dl):
        ax_dl.text(v + (0.00005 if v >= 0 else -0.00005), i,
                   f" {v:+.4f}", va="center", fontsize=6.5,
                   ha="left" if v >= 0 else "right", color="white", alpha=0.85)
    _style(ax_dl, title="Per-concept ΔL2 (single GT intervention, residual on)",
           xlabel="ΔL2  (m)   [< 0 helps, > 0 hurts]")
    ax_dl.set_yticks(y); ax_dl.set_yticklabels(names_dl, fontsize=8, color="white")

    # ③ ΔL2 vs prediction error scatter
    pe   = np.array([r["pred_error"] for r in pc_table])
    dl   = np.array([r["delta_l2"]   for r in pc_table])
    ks   = [r["kind"] for r in pc_table]
    nms  = [r["concept"] for r in pc_table]
    ax_sc.set_facecolor(_PANEL)
    ax_sc.scatter(pe, dl, c=[_KIND_COLOR.get(k, "white") for k in ks], s=55,
                  edgecolors="white", lw=0.4, zorder=3)
    ax_sc.axhline(0, color="white", lw=0.7, alpha=0.5, ls="--")
    if pe.std() > 0:
        rho = float(np.corrcoef(pe, dl)[0, 1])
        # Trend line
        m, b = np.polyfit(pe, dl, 1)
        xs = np.linspace(pe.min(), pe.max(), 100)
        ax_sc.plot(xs, m * xs + b, color="#fdcb6e", lw=1.5, ls="--", alpha=0.7)
        ax_sc.set_title(f"ΔL2 vs prediction error   corr = {rho:+.2f}",
                        color="white", fontsize=10, pad=7, fontweight="bold")
    for xi, yi, nm in zip(pe, dl, nms):
        ax_sc.annotate(nm.replace("_", " "), (xi, yi),
                       textcoords="offset points", xytext=(4, 3),
                       fontsize=5.5, color="white", alpha=0.75)
    for kind, col in _KIND_COLOR.items():
        ax_sc.scatter([], [], color=col, s=40, label=kind)
    ax_sc.legend(fontsize=7.5, facecolor=_BG, edgecolor=_GRID, labelcolor="white")
    _style(ax_sc, xlabel="Prediction error |GT − pred|  (activation space)",
           ylabel="ΔL2  (m)")

    # Overall baseline annotation
    curves = interp_data["curves"]
    base = curves["imp"]["residual_on"]["l2_avg"][0]
    delta_all = curves["imp"]["residual_on"]["l2_avg"][-1] - base
    fig.suptitle(
        f"Steerability — Test-time Concept Intervention  "
        f"(baseline L2={base:.3f}m,  all-27-GT Δ={delta_all:+.4f}m,  residual on)",
        color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Page 5: Steer matrix (optional) ──────────────────────────────────────────

_METHOD_COLOR = {"jnt": "#5dade2", "ind": "#e67e22"}
_METHOD_LABEL = {"jnt": "JNT (standard)", "ind": "IND (teacher-forced)"}


def _page_steer_matrix(summary_data, fig_kw):
    rows = summary_data
    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(16, 7), **fig_kw)
    fig.patch.set_facecolor(_BG)
    _DIM_SZ = {8: 80, 32: 130, 64: 190, 128: 260}

    # Scatter: accuracy vs steer gain
    ax_sc.set_facecolor(_PANEL)
    for r in rows:
        mc = _METHOD_COLOR[r["method"]]
        sz = _DIM_SZ.get(r["residual_dim"], 100)
        ax_sc.scatter(r["baseline_l2"], r["steer_gain_on"],
                      s=sz, color=mc, edgecolors="white", lw=0.8, zorder=3, alpha=0.9)
        ax_sc.annotate(r["cell"], (r["baseline_l2"], r["steer_gain_on"]),
                       textcoords="offset points", xytext=(6, 4),
                       fontsize=7.5, color="white")
    ax_sc.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax_sc.invert_xaxis()
    for m, mc in _METHOD_COLOR.items():
        ax_sc.scatter([], [], s=120, color=mc, edgecolors="white", lw=0.8,
                      label=_METHOD_LABEL[m])
    for dim, sz in _DIM_SZ.items():
        ax_sc.scatter([], [], s=sz, color="grey", edgecolors="white", lw=0.8,
                      label=f"r={dim}", alpha=0.6)
    ax_sc.legend(fontsize=7.5, facecolor=_BG, edgecolor=_GRID, labelcolor="white",
                 loc="lower left")
    _style(ax_sc, title="Accuracy vs Steerability",
           xlabel="Baseline L2_avg (m)  ← better accuracy",
           ylabel="Steer gain Δon (m)  ↑ more steerable")

    # Bar: Δon per cell sorted
    rows_s = sorted(rows, key=lambda r: r["steer_gain_on"], reverse=True)
    x = np.arange(len(rows_s))
    vals   = [r["steer_gain_on"] for r in rows_s]
    colors = [_METHOD_COLOR[r["method"]] for r in rows_s]
    labels = [r["cell"] for r in rows_s]
    ax_bar.set_facecolor(_PANEL)
    bars = ax_bar.bar(x, vals, color=colors, edgecolor="none", width=0.6)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    v - 0.001 if v < 0 else v + 0.0005,
                    f"{v:+.4f}", ha="center",
                    va="top" if v < 0 else "bottom",
                    fontsize=8, color="white")
    ax_bar.axhline(0, color="white", lw=0.8, alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=25, ha="right", color="white", fontsize=9)
    ylo = min(vals) * 1.2
    yhi = max(max(vals) * 1.3, 0.005)
    ax_bar.set_ylim(ylo, yhi)
    for m, mc in _METHOD_COLOR.items():
        ax_bar.bar([], [], color=mc, label=_METHOD_LABEL[m])
    ax_bar.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor="white")
    _style(ax_bar, title="Steer gain Δon per cell  (IMP, all GT concepts)",
           ylabel="Δon (m)  [= baseline − steered L2]")
    ax_bar.grid(True, color=_GRID, lw=0.4, alpha=0.6, axis="y")
    ax_bar.grid(False, axis="x")

    fig.suptitle("Steer Matrix: 4×2  (residual_dim ∈ {8,32,64,128} × {JNT, IND})",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval",   type=Path, required=True,
                    help="eval_stp3.json from eval_gen.py")
    ap.add_argument("--interp", type=Path, required=True,
                    help="intervention_curve.json from run_intervention.py")
    ap.add_argument("--steer_summary", type=Path, default=None,
                    help="intervention_steer_matrix_summary.json (optional)")
    ap.add_argument("--out",    type=Path, default=Path("./results.pdf"))
    args = ap.parse_args()

    eval_d   = json.loads(args.eval.read_text())
    interp_d = json.loads(args.interp.read_text())

    concepts = eval_d["concepts"]
    traj     = eval_d["trajectory"]
    fig_kw   = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages = 0
    with PdfPages(args.out) as pdf:
        pdf.savefig(_page_binary(concepts["binary"], fig_kw),
                    facecolor=_BG); plt.close(); pages += 1
        pdf.savefig(_page_continuous(concepts["continuous"],
                                     concepts["categorical"], fig_kw),
                    facecolor=_BG); plt.close(); pages += 1
        pdf.savefig(_page_trajectory(traj, fig_kw),
                    facecolor=_BG); plt.close(); pages += 1
        pdf.savefig(_page_steerability(interp_d, fig_kw),
                    facecolor=_BG); plt.close(); pages += 1
        if args.steer_summary and args.steer_summary.exists():
            summary = json.loads(args.steer_summary.read_text())
            pdf.savefig(_page_steer_matrix(summary, fig_kw),
                        facecolor=_BG); plt.close(); pages += 1
        else:
            print("(skipping steer matrix page — no --steer_summary provided)")
    print(f"wrote {args.out}  ({pages} pages)")


if __name__ == "__main__":
    main()
