"""Test-time concept-intervention experiment for the continuous-regression model.

Protocol (Koh et al. 2020; Shin et al. 2023): replace predicted concept
activations with ground truth for a growing subset of concepts, ordered by a
selection criterion, and measure trajectory L2 as a function of #concepts
intervened. Run with the residual path ON (realistic) and OFF (concepts-only
upper bound); the ON↔OFF gap measures how much the residual "explains away" the
concepts — a direct test of the confined adversarial disentanglement.

Efficiency: the backbone + CBL + residual are run ONCE over the split and cached;
the entire (orderings × budget × residual) sweep is then just linear-head
matmuls on the cached activation matrix.

Writes intervention_curve.json. Usage: see naut/interp-intervention-gen-regression.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.backbone import CBVLAMBackbone
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.models.residual import UnsupervisedResidual
from cb_vlam.models.final_predictor import FinalPredictor
from cb_vlam.training.dataset import ConceptDataset, collate
from cb_vlam.eval.metrics import trajectory_l2

from interp_experiments import intervention as IV
from interp_experiments import selection as SEL

_METRICS = ["ade_m", "fde_m", "l2_avg", "l2_1s", "l2_2s", "l2_3s"]


def _make_dataset(store, split, jsons, nuscenes_root, horizon):
    if split == "stp3":
        parts = [ConceptDataset(store, s, jsons, nuscenes_root, with_trajectory=True,
                                horizon=horizon) for s in ("val", "test")]
        print(f"split=stp3 (val ∪ test)  n={sum(len(p) for p in parts)}")
        return ConcatDataset(parts)
    ds = ConceptDataset(store, split, jsons, nuscenes_root, with_trajectory=True,
                        horizon=horizon)
    print(f"split={split}  n={len(ds)}")
    return ds


@torch.no_grad()
def _cache_forward(backbone, cbl, residual, loader, device):
    """One pass over the split → cached (a_pred, r, targets, trajectory)."""
    a_pred, r_all, traj, tmask = [], [], [], []
    tgt = {k: [] for k in ("continuous", "continuous_mask", "binary", "binary_mask",
                           "categorical", "categorical_mask")}
    for batch in loader:
        feats = torch.stack(
            [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])], dim=0)
        cbl_out = cbl(feats)
        a_pred.append(cbl.to_activations(cbl_out).float())
        r_all.append(residual(feats).float())
        traj.append(batch["trajectory"].to(device).float())
        tmask.append(batch["trajectory_mask"].to(device))
        for k in tgt:
            tgt[k].append(batch["targets"][k])
        torch.cuda.empty_cache()
    cat = lambda xs: torch.cat(xs, dim=0)
    targets = {k: cat(v) for k, v in tgt.items()}
    return cat(a_pred), cat(r_all), targets, cat(traj), cat(tmask)


def _head_traj(a_int, r, Wc, Wr, b, residual_on):
    """Linear FinalPredictor: [a ⊕ r] → waypoints; r dropped when residual_off."""
    out = a_int @ Wc.t() + b
    if residual_on:
        out = out + r @ Wr.t()
    return out


def _curve_for_rank(rank, sup, a_pred, a_gt, c2col, Wc, Wr, b, r,
                    gt_wp, valid_wp, horizon, C):
    """Sweep budget m=0..C for one ordering; return {residual_on/off: {metric:[...]}}."""
    res = {mode: {m: [] for m in _METRICS} for mode in ("residual_on", "residual_off")}
    N = a_pred.shape[0]
    for m in range(C + 1):
        sel = (rank < m) & sup
        a_int = IV.apply(a_pred, a_gt, sel, c2col)
        for mode, on in (("residual_on", True), ("residual_off", False)):
            traj = _head_traj(a_int, r, Wc, Wr, b, on).reshape(N, horizon, 2)
            met = trajectory_l2(traj, gt_wp, valid_wp, horizon)
            for k in _METRICS:
                res[mode][k].append(met.get(k))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_train", required=True, type=Path)
    ap.add_argument("--impromptu_test", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--split", default="stp3", choices=["train", "val", "test", "stp3"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--rand_seeds", type=int, default=5, help="RAND curves to average")
    ap.add_argument("--cctp", action="store_true",
                    help="scale IMP by |activation| (Shin CCTP) instead of pure weight norm")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint_dir / "config.json").read_text())
    taps, residual_dim = cfg["feature_taps"], cfg["residual_dim"]
    horizon, lora_r, schema_hash = cfg["horizon"], cfg["lora_r"], cfg["schema_hash"]
    print(f"checkpoint: mode={cfg.get('mode')} horizon={horizon} residual_dim={residual_dim}")

    store = ConceptStore(args.concept_store, schema_hash=schema_hash)
    layout = store.manifest["per_type"]
    jsons = [args.impromptu_train, args.impromptu_test]
    ds = _make_dataset(store, args.split, jsons, args.nuscenes_root, horizon)
    if args.max_samples:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    backbone = CBVLAMBackbone(
        checkpoint_path=args.base_checkpoint, feature_taps=taps,
        processor_path=args.processor_name, dtype=args.dtype, lora_r=lora_r,
        adapter_path=str(args.checkpoint_dir / "lora_adapter"), device=args.device,
        gradient_checkpointing=False)
    cbl = ConceptBottleneckLayer(in_dim=backbone.feature_dim, layout=layout).to(args.device)
    residual = UnsupervisedResidual(in_dim=backbone.feature_dim,
                                    residual_dim=residual_dim).to(args.device)
    head = FinalPredictor(cbl.activation_dim, residual_dim,
                          output_dim=horizon * 2, mode="regression").to(args.device)
    cbl.load_state_dict(torch.load(args.checkpoint_dir / "cbl.pt", map_location=args.device))
    residual.load_state_dict(torch.load(args.checkpoint_dir / "residual.pt", map_location=args.device))
    head.load_state_dict(torch.load(args.checkpoint_dir / "final_predictor.pt", map_location=args.device))
    cbl.eval(); residual.eval(); head.eval()

    # ── Pass 1: forward + cache ──────────────────────────────────────────────
    a_pred, r, targets, traj, tmask = _cache_forward(backbone, cbl, residual, loader, args.device)
    N = a_pred.shape[0]
    A = cbl.activation_dim
    gt_wp = traj.reshape(N, horizon, 2)
    valid_wp = tmask.reshape(N, horizon, 2)[..., 0].float()

    catalog = IV.build_catalog(cbl)
    C = len(catalog)
    c2col = IV.concept_to_col(catalog, A, args.device)
    a_gt, sup = IV.gt_activation(catalog, targets, N, A, args.device)

    # Linear-head weight blocks (concept | residual | bias).
    Wc = head.concept_weight.detach().float()                  # (out, A)
    Wr = head.fc.weight[:, head.concept_dim:].detach().float()  # (out, residual_dim)
    b = head.fc.bias.detach().float()                          # (out,)

    # ── Importance table (the model's "most important concepts") ─────────────
    imp = SEL.importance_scores(Wc, catalog, N, a_pred, scale_by_activation=False)[0]
    importance_table = sorted(
        [{"concept": catalog[c]["name"], "kind": catalog[c]["kind"],
          "weight_norm": float(imp[c])} for c in range(C)],
        key=lambda d: -d["weight_norm"])
    print("\nMost important concepts (linear-weight norm):")
    for row in importance_table[:8]:
        print(f"  {row['concept']:32} {row['kind']:11} {row['weight_norm']:.3f}")

    # ── Sweep: orderings × budget × residual ─────────────────────────────────
    curves = {}

    # RAND — averaged over seeds.
    rand_acc = None
    for s in range(args.rand_seeds):
        g = torch.Generator(device=args.device).manual_seed(1234 + s)
        rk = SEL.rank_of(SEL.random_scores(N, C, args.device, g), sup)
        cur = _curve_for_rank(rk, sup, a_pred, a_gt, c2col, Wc, Wr, b, r,
                              gt_wp, valid_wp, horizon, C)
        if rand_acc is None:
            rand_acc = cur
        else:
            for mode in cur:
                for k in cur[mode]:
                    rand_acc[mode][k] = [a + b2 for a, b2 in zip(rand_acc[mode][k], cur[mode][k])]
    for mode in rand_acc:
        for k in rand_acc[mode]:
            rand_acc[mode][k] = [v / args.rand_seeds for v in rand_acc[mode][k]]
    curves["rand"] = rand_acc

    # IMP — global linear-weight importance (optionally CCTP-scaled).
    imp_scores = SEL.importance_scores(Wc, catalog, N, a_pred, scale_by_activation=args.cctp)
    curves["imp"] = _curve_for_rank(SEL.rank_of(imp_scores, sup), sup, a_pred, a_gt,
                                    c2col, Wc, Wr, b, r, gt_wp, valid_wp, horizon, C)

    # LCP — oracle GT-error ordering.
    lcp = SEL.lcp_scores(a_pred, a_gt, sup, catalog)
    curves["lcp"] = _curve_for_rank(SEL.rank_of(lcp, sup), sup, a_pred, a_gt,
                                    c2col, Wc, Wr, b, r, gt_wp, valid_wp, horizon, C)

    # ── Report + write ───────────────────────────────────────────────────────
    base = curves["imp"]["residual_on"]["l2_avg"][0]   # m=0 = the model's own prediction
    print(f"\nbaseline L2_avg (no intervention) = {base:.3f} m")
    for ordering in ("rand", "imp", "lcp"):
        on = curves[ordering]["residual_on"]["l2_avg"][-1]
        off = curves[ordering]["residual_off"]["l2_avg"][-1]
        print(f"  all-{C}-intervened [{ordering:4}]  residual_on={on:.3f}  residual_off={off:.3f}")

    out = args.out or (args.checkpoint_dir.parent / "intervention_curve.json")
    out.write_text(json.dumps({
        "meta": {"checkpoint": str(args.checkpoint_dir), "split": args.split,
                 "n": int(N), "n_concepts": C, "horizon": horizon,
                 "rand_seeds": args.rand_seeds, "imp_kind": "cctp" if args.cctp else "weight_norm",
                 "metrics": _METRICS},
        "x": list(range(C + 1)),
        "importance_table": importance_table,
        "curves": curves,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
