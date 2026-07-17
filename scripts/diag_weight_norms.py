"""Diagnostic B — weight-norm audit across all saved checkpoints.

Answers: did the concept block W_c of the linear FinalPredictor actually collapse
(‖W_c‖ → 0), and did L_steer / IND / residual_dim ever move it? Discriminates the
same two hypotheses as Diagnostic A from the *trained weights* side.

Tier 1 (state_dict only — required, CPU, no backbone, no HF dataset):
  For each checkpoint's final_predictor.pt, split fc.weight into
    W_c = first `activation_dim` columns   (the 27-concept / 29-slot block)
    W_u = remaining `residual_dim` columns  (the unsupervised residual block)
  and report Frobenius norms, a per-column-fair norm ratio (the residual has many
  more columns, so raw Frobenius is misleading), per-concept column norms via
  `CBL.activation_slices`, dead-weight rates, and the bias norm.

Only manifest.json (for the concept layout) + each best/{final_predictor.pt,
config.json} are read — no `datasets` load, no GPU.

Usage:
  python scripts/diag_weight_norms.py \
    --runs_dir /sc-rwx-vol/cbvlam/outputs/runs \
    --concept_store /sc-rwx-vol/cbvlam/outputs/impromptu_concepts_ab/mined \
    [--ridge_json outputs/diagnostics/ridge_ceiling.json]   # → also assemble REPORT.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cb_vlam.models.cbl import ConceptBottleneckLayer
from interp_experiments import intervention as IV

DEAD = 1e-4   # |w| below this counts as a dead weight


def _load_layout(concept_store: Path):
    """Concept layout from manifest.json only (no HF dataset load)."""
    manifest = json.loads((concept_store / "manifest.json").read_text())
    cbl = ConceptBottleneckLayer(in_dim=1, layout=manifest["per_type"])
    return cbl, IV.build_catalog(cbl), cbl.activation_dim


def _discover(runs_dir: Path, cells):
    """Return [(cell_name, best_dir)] for every checkpoint with a final_predictor.pt."""
    if cells:
        found = [(c, runs_dir / c / "best") for c in cells]
    else:
        found = sorted((p.parent.parent.name, p.parent)
                       for p in runs_dir.glob("*/best/final_predictor.pt"))
    return [(name, d) for name, d in found if (d / "final_predictor.pt").exists()]


def _audit_one(best_dir: Path, catalog, A):
    cfg = json.loads((best_dir / "config.json").read_text()) if (best_dir / "config.json").exists() else {}
    sd = torch.load(best_dir / "final_predictor.pt", map_location="cpu")
    W = sd["fc.weight"].to(torch.float64).numpy()          # (out, A + residual_dim)
    b = sd["fc.bias"].to(torch.float64).numpy() if "fc.bias" in sd else np.zeros(W.shape[0])
    rdim = int(cfg.get("residual_dim", W.shape[1] - A))
    assert W.shape[1] == A + rdim, f"{best_dir}: weight cols {W.shape[1]} != A({A})+r({rdim})"

    Wc, Wu = W[:, :A], W[:, A:]
    col_norms_c = np.linalg.norm(Wc, axis=0)               # per-column L2
    col_norms_u = np.linalg.norm(Wu, axis=0)
    per_concept = sorted(
        [{"concept": m["name"], "kind": m["kind"],
          "norm": float(np.linalg.norm(Wc[:, m["col0"]:m["col0"] + m["k"]])),
          "norm_per_col": float(np.linalg.norm(Wc[:, m["col0"]:m["col0"] + m["k"]]) / m["k"] ** 0.5)}
         for m in catalog],
        key=lambda d: -d["norm_per_col"])
    return {
        "residual_dim": rdim,
        "method": "ind" if cfg.get("teacher_force") else "jnt",
        "lambda_steer": cfg.get("lambda_steer", 0.0),
        "steer_margin": cfg.get("steer_margin"),
        "lambda_reg": cfg.get("lambda_reg"),
        "fro_Wc": float(np.linalg.norm(Wc)), "fro_Wu": float(np.linalg.norm(Wu)),
        "fro_ratio": float(np.linalg.norm(Wc) / max(np.linalg.norm(Wu), 1e-12)),
        "mean_col_Wc": float(col_norms_c.mean()), "mean_col_Wu": float(col_norms_u.mean()),
        "col_ratio": float(col_norms_c.mean() / max(col_norms_u.mean(), 1e-12)),
        "dead_Wc": float((np.abs(Wc) < DEAD).mean()), "dead_Wu": float((np.abs(Wu) < DEAD).mean()),
        "bias_norm": float(np.linalg.norm(b)),
        "per_concept": per_concept,
    }


def _cell_sort_key(name, a):
    return (a["residual_dim"], 0 if a["method"] == "jnt" else 1, a["lambda_steer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("/sc-rwx-vol/cbvlam/outputs/runs"))
    ap.add_argument("--concept_store", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/impromptu_concepts_ab/mined"))
    ap.add_argument("--cells", nargs="*", default=None,
                    help="explicit run-dir names; default = auto-discover *//best/final_predictor.pt")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/diagnostics"))
    ap.add_argument("--ridge_json", type=Path, default=None,
                    help="if given, assemble the merged REPORT.md from both diagnostics")
    args = ap.parse_args()

    cbl, catalog, A = _load_layout(args.concept_store)
    print(f"activation_dim={A}  concepts={len(catalog)}")

    found = _discover(args.runs_dir, args.cells)
    if not found:
        print(f"No checkpoints under {args.runs_dir} (*/best/final_predictor.pt)");
    audit = {}
    for name, best in found:
        try:
            audit[name] = _audit_one(best, catalog, A)
            a = audit[name]
            print(f"  {name:40} r={a['residual_dim']:<4} {a['method']} λs={a['lambda_steer']:<4} "
                  f"‖Wc‖={a['fro_Wc']:.3f} ‖Wu‖={a['fro_Wu']:.3f} col_ratio={a['col_ratio']:.3f}")
        except Exception as e:  # keep going; a corrupt cell shouldn't sink the audit
            print(f"  SKIP {name}: {e}")

    cells_sorted = sorted(audit, key=lambda n: _cell_sort_key(n, audit[n]))

    # Tier 2 (contribution decomposition) needs cached c/u activations on disk;
    # none exist (run_intervention caches in-memory only). Intermediate weight
    # snapshots for a collapse curve also don't exist (train_gen._save writes only
    # best/). Both are recorded as skipped in the report.
    payload = {"activation_dim": A, "n_concepts": len(catalog), "dead_threshold": DEAD,
               "runs_dir": str(args.runs_dir), "cells_sorted": cells_sorted, "audit": audit,
               "tier2": "skipped — no cached c/u activation dump on disk",
               "collapse_curve": "skipped — only best/ saved (train_gen._save), no step snapshots"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "weight_norms.json").write_text(json.dumps(payload, indent=2))
    (args.out_dir / "weight_norms.md").write_text(_render_md(payload))
    print(f"wrote {args.out_dir/'weight_norms.json'} and .md")

    if args.ridge_json and args.ridge_json.exists():
        rj = json.loads(args.ridge_json.read_text())
        (args.out_dir / "REPORT.md").write_text(_render_report(rj, payload, args.out_dir))
        print(f"assembled {args.out_dir/'REPORT.md'}")
    elif args.ridge_json:
        print(f"  (ridge_json {args.ridge_json} not found — REPORT.md not assembled)")


def _render_md(p):
    a = p["audit"]
    L = ["## Diagnostic B — weight-norm audit\n",
         f"activation_dim {p['activation_dim']} (W_c cols) · dead-weight |w|<{p['dead_threshold']}\n",
         "| cell | r | method | λ_steer | ‖W_c‖_F | ‖W_u‖_F | col-fair W_c/W_u | dead W_c | dead W_u | ‖bias‖ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name in p["cells_sorted"]:
        c = a[name]
        L.append(f"| {name} | {c['residual_dim']} | {c['method']} | {c['lambda_steer']} | "
                 f"{c['fro_Wc']:.3f} | {c['fro_Wu']:.3f} | {c['col_ratio']:.3f} | "
                 f"{c['dead_Wc']:.2f} | {c['dead_Wu']:.2f} | {c['bias_norm']:.3f} |")
    # per-concept column norms for representative cells (full tables live in the json)
    reps = [n for n in ("gen_regression_steer_r128_jnt", "gen_regression_steer_r128_ind",
                        "gen_regression_steer_r128_ind_lsteer_l0p0",
                        "gen_regression_steer_r128_ind_lsteer_l4p0") if n in a]
    reps = reps or p["cells_sorted"][:1]
    topk = min(12, p["n_concepts"])
    for name in reps:
        L.append(f"\n**Per-concept ‖W_c[:,concept]‖ (col-fair) — {name} (top {topk} of "
                 f"{p['n_concepts']}):**")
        L.append("| concept | kind | ‖W‖/√k |\n|---|---|---|")
        for row in a[name]["per_concept"][:12]:
            L.append(f"| {row['concept']} | {row['kind']} | {row['norm_per_col']:.3f} |")
    L.append(f"\n**Tier 2:** {p['tier2']}.")
    L.append(f"**Collapse curve:** {p['collapse_curve']}.")
    L.append(_B_KEY)
    return "\n".join(L) + "\n"


_B_KEY = """
### Interpretation key (Diagnostic B)
- `‖W_c‖/‖W_u‖` (per-column-fair) ≪ 1 everywhere, flat across λ → confirms L_steer never engaged (consistent with its `W_c=0` trivial solution) and that elastic-net asymmetry (penalty on `W_c` only) plausibly contributed — check whether λ_reg runs show lower `‖W_c‖` than λ_reg=0 runs if both exist.
- ratio rises as residual_dim shrinks → residual capacity wasn't the constraint (matches the falsified sweep).
- per-concept columns: if `ego_stopped`/`ego_braking` columns are among the largest surviving norms, the head does use the kinematic *binaries* — more evidence the missing continuous kinematics are the payload."""


def _render_report(rj, wj, out_dir):
    """Merged REPORT.md from both diagnostics' payloads (template in the task)."""
    r = rj["results"]
    a1 = r["A1"]["metric"]["l2_avg"]
    a2 = r.get("A2", {}).get("metric", {}).get("l2_avg")
    co = rj["ref_concepts_only_l2avg"]; full = rj["ref_full_l2avg"]
    # headline W_c/W_u ratio for the reference cell if present
    ref_cell = next((n for n in ("gen_regression_steer_r128_ind_lsteer_l0p0",
                                 "gen_regression_steer_r128_ind") if n in wj["audit"]), None)
    ratio = wj["audit"][ref_cell]["col_ratio"] if ref_cell else float("nan")
    ratios = [wj["audit"][n]["col_ratio"] for n in wj["cells_sorted"]]
    rlo, rhi = (min(ratios), max(ratios)) if ratios else (float("nan"), float("nan"))

    # data-driven verdict (factual; thresholds from the interpretation keys)
    incomplete = a1 >= 0.8 * co
    verdict = (
        f"A1 (linear ceiling) = {a1:.3f} m vs trained concepts-only {co} m: "
        + ("**within ~20% of the ceiling → vocabulary incompleteness dominates**; no "
           "training-side fix (sequential fitting, L_steer, λ tuning) can materially "
           "help until the vocabulary is expanded (e.g. ego-kinematics)."
           if incomplete else
           "**A1 sits well below the trained concepts-only baseline → the joint-trained "
           "head left concept signal on the table (W_c under-used); a better training "
           "scheme has real headroom even before vocabulary expansion.**"))
    if a2 is not None:
        verdict += (f" A2 (nonlinear) = {a2:.3f} m"
                    + (" ≈ A1 → linearity is not the binding constraint."
                       if a2 >= a1 - 0.05 else
                       f" ≪ A1 → nonlinear concept→trajectory structure exists (linear-head cost ≈ {a1 - a2:.3f} m)."))

    gate = rj["results"].get("gate")
    head = [
        "## Headline",
        f"- Ridge ceiling (A1, ST-P3 Avg): **{a1:.3f} m**   [vs concepts-only trained: "
        f"{co}, full: {full}]",
        f"- Nonlinear ceiling (A2): **{a2:.3f} m**" if a2 is not None else "- Nonlinear ceiling (A2): skipped",
        f"- W_c/W_u per-column ratio, {ref_cell or 'n/a'}: **{ratio:.3f}**   "
        f"(range across cells: {rlo:.3f}–{rhi:.3f})"]
    if gate:
        head.append(
            f"- **Route-A Phase-1 gate: {'PASS ✅' if gate['pass'] else 'NO-GO ❌'}** — "
            f"A5 (concepts+named kinematics) = **{gate['a5_named']:.3f} m**, "
            f"A6 (concepts+raw history) = {gate['a6_raw_history']:.3f} m, "
            f"named recover {100*gate['frac_recovered_by_named']:.0f}% of the A1→A6 gain")
    head.append("")
    verdict_block = verdict
    if gate:
        verdict_block = (
            f"**Phase-1 gate {'PASS' if gate['pass'] else 'NO-GO'}:** adding measured ego "
            f"kinematics drops the concept-vocabulary ceiling {a1:.3f} → {gate['a5_named']:.3f} m, "
            f"and the 4 named summaries recover {100*gate['frac_recovered_by_named']:.0f}% of what "
            f"the raw 14-dim history buys — the vocabulary was the binding constraint and the "
            f"named kinematics are a sufficient statistic of the history. "
            + ("Proceed to Phase 2.\n\n" if gate['pass'] else "Do NOT proceed — refine derived quantities.\n\n")
            + verdict)
    L = [f"# Diagnostics: vocabulary ceiling + weight audit",
         f"_ridge: sklearn {rj['sklearn']}, seed {rj['seed']}, n_eval {rj['n_eval']} · "
         f"weights: {len(wj['audit'])} checkpoints under {wj['runs_dir']}_\n",
         *head,
         "## Verdict", verdict_block, "",
         "## Tables", "", rj_md(out_dir), "", wj_md(wj), "",
         "## Caveats", _caveats(rj, wj)]
    return "\n".join(L) + "\n"


def rj_md(out_dir):
    # embed Diagnostic A's rendered section from the same out_dir
    p = Path(out_dir) / "ridge_ceiling.md"
    return p.read_text() if p.exists() else "(ridge_ceiling.md not found in out_dir)"


def wj_md(wj):
    return _render_md(wj)


def _caveats(rj, wj):
    lines = [
        f"- **VERIFY resolved — concept store:** `{rj['concept_store']}` "
        "(`manifest.json` → per_type/stats/splits; `cb_vlam/data/concept_store.py`).",
        f"- **VERIFY resolved — inputs/waypoints:** `{rj['impromptu_jsons']}`; waypoints "
        "parsed by `cb_vlam.training.dataset.parse_trajectory` in **meters, ego frame, "
        "NOT normalized** → no de-normalization needed (confirmed vs `eval_gen.py`).",
        f"- **VERIFY resolved — ST-P3 metric:** reused `cb_vlam.eval.metrics.trajectory_l2` "
        "verbatim (cumulative L2 @ {1s,2s,3s}=wp idx {1,3,5}, q7 sign-flip min).",
        f"- **VERIFY resolved — checkpoints:** `{wj['runs_dir']}/*/best/` "
        "(`final_predictor.pt`+`config.json`); the task's `outputs/genreg/…` guess was "
        "actually `outputs/runs/…` (per the naut training yamls).",
        "- **VERIFY resolved — splits:** train = `store.split('train')`; ST-P3 = "
        "`val ∪ test` (ConcatDataset), matching `run_intervention`/`eval_gen`.",
        f"- **Imputation coverage:** per-slot supervision rates in ridge_ceiling.json "
        "(`supervision_rate`); default masks supervise everything, so imputation is "
        "effectively inert unless a mask toggle was enabled.",
        "- **A1-pred / Tier 2 skipped:** no predicted-concept or c/u activation dump on "
        "disk (run_intervention caches in-memory only); running the backbone was out of "
        "scope. Collapse curve skipped: only `best/` is saved.",
        "- **Units checks:** A0 (train-mean trajectory) sanity-banded to ~1–4 m; A1 asserted "
        "not to beat the full model (0.389 m).",
        f"- **W_c coefficients** in Diagnostic A are in the standardized basis; Diagnostic B "
        "norms are raw trained weights (input scale of r is unknown — see Tier 2 skip).",
    ]
    if rj.get("guards"):
        lines.append("- **Ridge guards fired:** " + "; ".join(rj["guards"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
