"""Diagnostic A — ridge ceiling: how much trajectory is in the vocabulary at all?

Answers: the maximum trajectory accuracy ANY head could extract from the current
27-concept vocabulary (29 activation slots), independent of training dynamics.
This discriminates the two live hypotheses for why steerability ≈ 0:

  * vocabulary incompleteness  → even a best-fit linear/nonlinear map from GT
    concepts lands near the trained concepts-only condition (3.775 m); no
    training-side fix can help until the vocabulary is expanded.
  * W_c collapse (fixable)      → a fresh fit lands well below 3.775 m, so the
    joint-trained head under-used the existing concepts.

CPU-only: GT concepts + GT waypoints + sklearn. No backbone forward passes.

Data matrix (reuses the exact deployed layout):
  X = GT concept-activation vector, the `interp_experiments.intervention.gt_activation`
      layout (continuous value as stored/normalized, binary ∈ {0,1}, categorical
      one-hot) → (n, 29). Masked/unsupervised entries imputed with the TRAIN-split
      marginal mean of that column (computed on train only, applied to both splits).
  y = GT flat waypoints (n, 12) in meters, ego frame, parsed by the same
      `cb_vlam.training.dataset.parse_trajectory` the training/eval path uses
      (waypoints are NOT normalized — no de-norm needed).

Metric = `cb_vlam.eval.metrics.trajectory_l2` (reused verbatim): ST-P3 cumulative
L2 @ {1s,2s,3s,Avg} + flat ADE/FDE. Fit on train; evaluate on ST-P3 (val ∪ test).

Usage (paths default to the cluster mount; override for a local copy):
  python scripts/diag_ridge_ceiling.py \
    --concept_store   /sc-rwx-vol/cbvlam/outputs/impromptu_concepts_ab/mined \
    --impromptu_train /sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_train.json \
    --impromptu_test  /sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_test.json \
    --nuscenes_root   /sc-rwx-vol/cbvlam
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.training.dataset import ConceptDataset
from cb_vlam.eval.metrics import trajectory_l2
from interp_experiments import intervention as IV

# Trained reference conditions (from the deployed gen_regression_full eval on the
# same ST-P3 split, n=5119): full model residual-on, and the trained concepts-only
# (residual = train-mean) baseline. Used only for interpretation + a units guard.
REF_FULL_L2AVG = 0.389
REF_CONCEPTS_ONLY_L2AVG = 3.775


# ── data assembly ─────────────────────────────────────────────────────────────

def _collect_split(store, split_names, jsons, nuscenes_root, horizon, max_samples):
    """Iterate ConceptDataset(s) → stacked targets + GT waypoints + validity.

    split_names is a list so ST-P3 = ["val", "test"] concatenates like the eval.
    Images are not loaded (load_image=False) — this is a pure label/target pass.
    """
    tgt_keys = ["continuous", "continuous_mask", "binary", "binary_mask",
                "categorical", "categorical_mask"]
    acc = {k: [] for k in tgt_keys}
    traj, mask, tokens = [], [], []
    for sname in split_names:
        ds = ConceptDataset(store, sname, jsons, nuscenes_root, load_image=False,
                            with_trajectory=True, horizon=horizon, max_samples=max_samples)
        for i in range(len(ds)):
            item = ds[i]
            t = item["target"]
            for k in tgt_keys:
                acc[k].append(t[k])
            traj.append(item["trajectory"])
            mask.append(item["trajectory_mask"])
            tokens.append(item["sample_token"])
    targets = {k: torch.from_numpy(np.stack(acc[k], axis=0)) for k in tgt_keys}
    y = np.stack(traj, axis=0).astype(np.float64)          # (n, 12) meters
    valid = np.stack(mask, axis=0).astype(bool)            # (n, 12)
    return targets, y, valid, tokens


def _build_X(catalog, c2col, targets, activation_dim):
    """GT activation matrix (n, A) + per-column supervised mask (n, A) bool."""
    n = targets["continuous"].shape[0]
    a_gt, sup = IV.gt_activation(catalog, targets, n, activation_dim, "cpu")
    col_sup = (sup.to(torch.float32) @ c2col) > 0          # (n, C)·(C, A) → (n, A)
    return a_gt.numpy().astype(np.float64), col_sup.numpy()


def _impute(a_gt, col_sup, col_mean):
    """Replace unsupervised entries with the train-marginal column mean."""
    X = a_gt.copy()
    miss = ~col_sup
    if miss.any():
        cols = np.where(miss.any(0))[0]
        for j in cols:
            X[miss[:, j], j] = col_mean[j]
    return X


# ── metric wiring (reuse trajectory_l2 verbatim) ─────────────────────────────

def _score(pred_flat, y_flat, valid_flat, horizon):
    """(n,12) predictions → ST-P3 L2@{1,2,3,Avg} + flat ADE/FDE via trajectory_l2."""
    n = pred_flat.shape[0]
    pred = torch.from_numpy(np.asarray(pred_flat, np.float64)).reshape(n, horizon, 2)
    gt = torch.from_numpy(np.asarray(y_flat, np.float64)).reshape(n, horizon, 2)
    valid_wp = torch.from_numpy(valid_flat.reshape(n, horizon, 2)[..., 0].astype(np.float64))
    return trajectory_l2(pred, gt, valid_wp, horizon)


def _per_dim_r2(pred, y, valid):
    """Per-output-dim R² over rows valid for that dim (12 values + mean)."""
    from sklearn.metrics import r2_score
    out = []
    for d in range(y.shape[1]):
        m = valid[:, d]
        out.append(float(r2_score(y[m, d], pred[m, d])) if m.sum() > 1 else float("nan"))
    return out, float(np.nanmean(out))


# ── models ────────────────────────────────────────────────────────────────────

def _fit_ridge(Xtr, ytr, Xev, alphas, seed):
    """Standardize (fit on train) → RidgeCV(5-fold) multi-output. Returns
    (pred_eval, coef_in_standardized_basis, chosen_alpha)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    sc = StandardScaler().fit(Xtr)
    reg = RidgeCV(alphas=alphas, cv=5).fit(sc.transform(Xtr), ytr)
    alpha = reg.alpha_
    alpha = float(np.mean(alpha)) if np.ndim(alpha) else float(alpha)
    return reg.predict(sc.transform(Xev)), np.atleast_2d(reg.coef_), alpha


def _concept_coef_norms(coef, catalog, cols_index=None):
    """Per-concept |coef| column-block norm (categorical summed over its slots).

    coef is (12, n_features) in the standardized basis. cols_index maps a full
    activation column → its position in a subsetted feature matrix (A3/A4); None
    means coef spans the full 29-column activation vector.
    """
    rows = []
    for meta in catalog:
        cols = list(range(meta["col0"], meta["col0"] + meta["k"]))
        if cols_index is not None:
            cols = [cols_index[c] for c in cols if c in cols_index]
            if not cols:
                continue
        rows.append({"concept": meta["name"], "kind": meta["kind"],
                     "coef_norm": float(np.linalg.norm(coef[:, cols]))})
    rows.sort(key=lambda d: -d["coef_norm"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_store", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/outputs/impromptu_concepts_ab/mined"))
    ap.add_argument("--impromptu_train", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_train.json"))
    ap.add_argument("--impromptu_test", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_test.json"))
    ap.add_argument("--nuscenes_root", type=Path, default=Path("/sc-rwx-vol/cbvlam"))
    ap.add_argument("--schema_hash", default=None)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/diagnostics"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_train", type=int, default=None, help="cap train rows (debug)")
    ap.add_argument("--max_eval", type=int, default=None, help="cap eval rows (debug)")
    ap.add_argument("--skip_a2", action="store_true", help="skip the HGB nonlinear ceiling")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import sklearn
    print(f"sklearn {sklearn.__version__}  seed={args.seed}")
    alphas = np.logspace(-3, 3, 13)

    store = ConceptStore(args.concept_store, schema_hash=args.schema_hash or None)
    layout = store.manifest["per_type"]
    cbl = ConceptBottleneckLayer(in_dim=1, layout=layout)   # layout-only, no weights used
    catalog = IV.build_catalog(cbl)
    A = cbl.activation_dim
    c2col = IV.concept_to_col(catalog, A, "cpu")
    jsons = [args.impromptu_train, args.impromptu_test]
    print(f"activation_dim={A}  concepts={len(catalog)}  "
          f"(cont={cbl.n_continuous} bin={cbl.n_binary} cat_slots={sum(cbl.cat_ncats)})")

    # ── assemble train / eval matrices ────────────────────────────────────────
    print("collecting train split ...")
    tgt_tr, ytr_full, vtr_full, tok_tr = _collect_split(
        store, ["train"], jsons, args.nuscenes_root, args.horizon, args.max_train)
    print("collecting ST-P3 eval split (val ∪ test) ...")
    tgt_ev, yev, vev, tok_ev = _collect_split(
        store, ["val", "test"], jsons, args.nuscenes_root, args.horizon, args.max_eval)

    a_gt_tr, sup_tr = _build_X(catalog, c2col, tgt_tr, A)
    a_gt_ev, sup_ev = _build_X(catalog, c2col, tgt_ev, A)

    # train-marginal column means (over supervised entries) → impute both splits
    col_mean = np.array([a_gt_tr[sup_tr[:, j], j].mean() if sup_tr[:, j].any() else 0.0
                         for j in range(A)])
    Xtr = _impute(a_gt_tr, sup_tr, col_mean)
    Xev = _impute(a_gt_ev, sup_ev, col_mean)
    sup_rate = sup_tr.mean(0)                               # per-column supervision coverage

    # Fit on complete-trajectory train rows (a partial GT waypoint would corrupt
    # the target); evaluate on ALL eval rows with the standard per-waypoint
    # validity gate, matching how 0.389 / 3.775 were computed.
    comp_tr = vtr_full.all(1)
    Xtr_c, ytr_c = Xtr[comp_tr], ytr_full[comp_tr]
    print(f"train rows={len(Xtr)} (complete={comp_tr.sum()})  eval rows={len(Xev)} "
          f"(complete={vev.all(1).sum()})")

    results = {}

    # A0 — train-mean trajectory (zero-information floor)
    ymean = np.array([ytr_full[vtr_full[:, j], j].mean() if vtr_full[:, j].any() else 0.0
                      for j in range(ytr_full.shape[1])])
    results["A0"] = {"name": "train-mean trajectory",
                     "metric": _score(np.tile(ymean, (len(Xev), 1)), yev, vev, args.horizon)}

    # A1 — RidgeCV linear ceiling (headline)
    p1, coef1, alpha1 = _fit_ridge(Xtr_c, ytr_c, Xev, alphas, args.seed)
    m1 = _score(p1, yev, vev, args.horizon)
    r2_1, r2m_1 = _per_dim_r2(p1, yev, vev)
    results["A1"] = {"name": "RidgeCV (linear ceiling)", "metric": m1, "alpha": alpha1,
                     "per_dim_r2": r2_1, "mean_r2": r2m_1,
                     "coef_norms_top10": _concept_coef_norms(coef1, catalog)[:10],
                     "basis": "standardized (StandardScaler on train)"}

    # A2 — HistGradientBoosting nonlinear ceiling (per output dim)
    if not args.skip_a2:
        from sklearn.ensemble import HistGradientBoostingRegressor
        p2 = np.zeros_like(p1)
        for d in range(ytr_c.shape[1]):
            reg = HistGradientBoostingRegressor(random_state=args.seed).fit(Xtr_c, ytr_c[:, d])
            p2[:, d] = reg.predict(Xev)
        m2 = _score(p2, yev, vev, args.horizon)
        r2_2, r2m_2 = _per_dim_r2(p2, yev, vev)
        results["A2"] = {"name": "HistGradientBoosting (nonlinear ceiling)",
                         "metric": m2, "per_dim_r2": r2_2, "mean_r2": r2m_2}

    # A3 — ridge on binary+categorical slots only; A4 — continuous slots only
    bc_cols = [j for meta in catalog if meta["kind"] in ("binary", "categorical")
               for j in range(meta["col0"], meta["col0"] + meta["k"])]
    ct_cols = [j for meta in catalog if meta["kind"] == "continuous"
               for j in range(meta["col0"], meta["col0"] + meta["k"])]
    for aid, name, cols in [("A3", "ridge — binary+categorical only", bc_cols),
                            ("A4", "ridge — continuous only", ct_cols)]:
        idx = {c: k for k, c in enumerate(cols)}
        p, coef, alpha = _fit_ridge(Xtr_c[:, cols], ytr_c, Xev[:, cols], alphas, args.seed)
        m = _score(p, yev, vev, args.horizon)
        _, r2m = _per_dim_r2(p, yev, vev)
        results[aid] = {"name": name, "metric": m, "alpha": alpha, "mean_r2": r2m,
                        "coef_norms_top10": _concept_coef_norms(coef, catalog, idx)[:10]}

    # A1-pred (predicted concepts) — requires an eval-activation dump on disk.
    # run_intervention caches a_pred in memory and writes only curve summaries
    # (run_intervention.py:276-285), so no dump exists → skipped by design (no
    # backbone forward is permitted here).
    results["A1_pred"] = {"skipped": "no predicted-concept activation dump on disk; "
                          "run_intervention caches in-memory only"}

    # ── guards ────────────────────────────────────────────────────────────────
    warns = []
    a0 = results["A0"]["metric"]["l2_avg"]
    a1 = results["A1"]["metric"]["l2_avg"]
    if not (0.5 <= a0 <= 8.0):
        warns.append(f"A0 l2_avg={a0:.3f} outside plausible ~1–4 m band — check units/split")
    if a1 < REF_FULL_L2AVG:
        warns.append(f"A1 l2_avg={a1:.3f} BEATS full model {REF_FULL_L2AVG} — units/split bug")
    for w in warns:
        print("  !! GUARD:", w)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sklearn": sklearn.__version__, "seed": args.seed, "activation_dim": A,
        "n_concepts": len(catalog), "horizon": args.horizon,
        "n_train": int(len(Xtr)), "n_train_complete": int(comp_tr.sum()),
        "n_eval": int(len(Xev)), "alphas": list(alphas),
        "ref_full_l2avg": REF_FULL_L2AVG, "ref_concepts_only_l2avg": REF_CONCEPTS_ONLY_L2AVG,
        "supervision_rate": {catalog[c]["name"]: float(sup_rate[catalog[c]["col0"]])
                             for c in range(len(catalog))},
        "results": results, "guards": warns,
        "impromptu_jsons": [str(j) for j in jsons],
        "concept_store": str(args.concept_store),
    }
    (args.out_dir / "ridge_ceiling.json").write_text(json.dumps(payload, indent=2))
    (args.out_dir / "ridge_ceiling.md").write_text(_render_md(payload))
    print(f"\nHEADLINE  A1 (linear) l2_avg={a1:.3f} m   "
          f"[trained concepts-only={REF_CONCEPTS_ONLY_L2AVG}, full={REF_FULL_L2AVG}]")
    if "A2" in results:
        print(f"          A2 (nonlinear) l2_avg={results['A2']['metric']['l2_avg']:.3f} m")
    print(f"wrote {args.out_dir/'ridge_ceiling.json'} and .md")


def _mrow(tag, r):
    m = r["metric"]
    def g(k): return f"{m.get(k, float('nan')):.3f}"
    note = f"mean R²={r['mean_r2']:.3f}" if "mean_r2" in r else ""
    return (f"| {tag} | {r['name']} | {g('l2_1s')} | {g('l2_2s')} | {g('l2_3s')} | "
            f"**{g('l2_avg')}** | {g('ade_m')} | {g('fde_m')} | {note} |")


def _render_md(p):
    r = p["results"]
    L = ["## Diagnostic A — ridge ceiling\n",
         f"sklearn {p['sklearn']} · seed {p['seed']} · activation_dim {p['activation_dim']} · "
         f"n_train {p['n_train']} (complete {p['n_train_complete']}) · n_eval {p['n_eval']}\n",
         "| id | model | L2@1s | L2@2s | L2@3s | L2 Avg | ADE | FDE | notes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tag in ("A0", "A1", "A2", "A3", "A4"):
        if tag in r and "metric" in r[tag]:
            L.append(_mrow(tag, r[tag]))
    L.append("")
    L.append(f"Reference: trained concepts-only (residual=train-mean) = "
             f"**{p['ref_concepts_only_l2avg']} m**; full model (residual on) = "
             f"**{p['ref_full_l2avg']} m**.")
    if "A1" in r:
        L.append(f"\nA1 chosen α = {r['A1'].get('alpha')}; coefficients reported in the "
                 f"{r['A1'].get('basis')}.")
        L.append("\n**A1 top-10 concept |coef| column norms (standardized basis):**")
        L.append("| concept | kind | ‖coef‖ |\n|---|---|---|")
        for row in r["A1"].get("coef_norms_top10", []):
            L.append(f"| {row['concept']} | {row['kind']} | {row['coef_norm']:.3f} |")
    if p["guards"]:
        L.append("\n**Guards:** " + "; ".join(p["guards"]))
    else:
        L.append("\n**Guards:** A0 in band, A1 does not beat the full model — units/split OK.")
    L.append(_A_KEY)
    return "\n".join(L) + "\n"


_A_KEY = """
### Interpretation key (Diagnostic A)
- **A1 ≈ 3.7–3.8 m** → the trained concepts-only condition (3.775) was already at the vocabulary ceiling. Incompleteness confirmed; no training-side fix (sequential fitting, L_intv, λ tuning) can materially help until the vocabulary is expanded (ego-kinematics). This is the expected outcome.
- **A1 substantially below 3.775 (e.g. ≤ 2 m)** → the joint-trained head is leaving concept signal on the table; `W_c` collapse is a real, fixable training pathology and sequential/post-hoc residual fitting has headroom even before vocabulary expansion.
- **A2 ≪ A1** → nonlinear concept→trajectory structure exists; the keep-the-head-linear decision has a measurable accuracy cost (quantify it), though it may still be worth paying for readability.
- **A3 vs A4** → tells us whether the (well-predicted) binaries or the (poorly-predicted) continuous concepts carry the trajectory signal — informs which vocabulary expansions pay off."""


if __name__ == "__main__":
    main()
