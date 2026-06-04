"""Held-out evaluation of a trained continuous-regression generation checkpoint.

Loads a saved ``best/`` directory (LoRA adapter + cbl + residual + final_predictor)
and runs ONE backbone pass over a chosen split (default: test), producing BOTH
reportable result families in a single JSON:

  - per-concept projection metrics  (continuous MAE/R², binary AUROC/F1,
    categorical accuracy/macro-F1 — via cb_vlam.eval.metrics)
  - trajectory metrics              (ADE / FDE in meters) + aggregate
    concept_loss / traj_loss

The model dims (taps, residual_dim, horizon, lora_r) and the expected schema_hash
are read from the checkpoint's config.json, so the eval can't silently drift from
how the checkpoint was trained.

Usage: see naut/eval-gen-regression-test.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.backbone import CBVLAMBackbone
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.models.residual import UnsupervisedResidual
from cb_vlam.models.final_predictor import FinalPredictor
from cb_vlam.training.dataset import ConceptDataset, collate
from cb_vlam.training.losses import (
    concept_loss, trajectory_loss, binary_pos_weight_from_manifest)
from cb_vlam.eval import metrics as M


@torch.no_grad()
def _eval_all(backbone, cbl, residual, head, loader, device, manifest, horizon, pos_weight):
    """One pass: collect per-concept arrays AND accumulate ADE/FDE + losses."""
    layout = manifest["per_type"]
    cont_p, cont_t, cont_m = [], [], []
    bin_p, bin_t, bin_m = [], [], []
    cat_p, cat_t, cat_m = [], [], []
    c_sum = t_sum = n = 0.0
    ade_sum = fde_sum = ade_n = fde_n = 0.0

    for batch in loader:
        feats = torch.stack(
            [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])],
            dim=0)
        cbl_out = cbl(feats)
        cvec = cbl.to_activations(cbl_out)
        r = residual(feats)
        traj = head(cvec, r)

        tgt = batch["targets"]
        tgt_dev = {k: v.to(device) for k, v in tgt.items()}
        gt = batch["trajectory"].to(device); m = batch["trajectory_mask"].to(device)
        bs = feats.shape[0]
        c_sum += float(concept_loss(cbl_out, tgt_dev, bin_pos_weight=pos_weight)["total"]) * bs
        t_sum += float(trajectory_loss(traj, gt, mask=m)) * bs
        n += bs

        if cbl_out["continuous"].shape[1]:
            cont_p.append(cbl_out["continuous"].float().cpu().numpy())
            cont_t.append(tgt["continuous"].numpy()); cont_m.append(tgt["continuous_mask"].numpy())
        if cbl_out["binary_logits"].shape[1]:
            bin_p.append(torch.sigmoid(cbl_out["binary_logits"]).float().cpu().numpy())
            bin_t.append(tgt["binary"].numpy()); bin_m.append(tgt["binary_mask"].numpy())
        if cbl_out["categorical_logits"]:
            cat_p.append(np.stack(
                [lg.argmax(-1).cpu().numpy() for lg in cbl_out["categorical_logits"]], axis=1))
            cat_t.append(tgt["categorical"].numpy()); cat_m.append(tgt["categorical_mask"].numpy())

        B = traj.shape[0]
        p = traj.reshape(B, horizon, 2); g = gt.reshape(B, horizon, 2)
        wp_valid = m.reshape(B, horizon, 2)[..., 0]
        d = ((p - g) ** 2).sum(-1).clamp(min=0).sqrt()
        ade_sum += float((d * wp_valid).sum()); ade_n += float(wp_valid.sum())
        last_valid = wp_valid[:, -1]
        fde_sum += float((d[:, -1] * last_valid).sum()); fde_n += float(last_valid.sum())
        torch.cuda.empty_cache()

    concepts = {"loss": c_sum / max(n, 1.0)}
    if cont_p:
        concepts["continuous"] = M._continuous_metrics(
            np.concatenate(cont_p), np.concatenate(cont_t), np.concatenate(cont_m),
            layout["continuous"]["names"])
    if bin_p:
        concepts["binary"] = M._binary_metrics(
            np.concatenate(bin_p), np.concatenate(bin_t), np.concatenate(bin_m),
            layout["binary"]["names"])
    if cat_p:
        concepts["categorical"] = M._categorical_metrics(
            np.concatenate(cat_p), np.concatenate(cat_t), np.concatenate(cat_m),
            layout["categorical"]["names"], layout["categorical"]["n_categories"])

    traj_metrics = {"ade_m": ade_sum / max(ade_n, 1.0), "fde_m": fde_sum / max(fde_n, 1.0),
                    "concept_loss": c_sum / max(n, 1.0), "traj_loss": t_sum / max(n, 1.0),
                    "n": int(n)}
    return concepts, traj_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=Path,
                    help="The trained 'best/' dir (cbl.pt, residual.pt, "
                         "final_predictor.pt, lora_adapter/, config.json).")
    ap.add_argument("--base_checkpoint", required=True,
                    help="Base Qwen2.5-VL weights the LoRA adapter sits on.")
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_train", required=True, type=Path)
    ap.add_argument("--impromptu_test", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint_dir / "config.json").read_text())
    taps = cfg["feature_taps"]; residual_dim = cfg["residual_dim"]
    horizon = cfg["horizon"]; lora_r = cfg["lora_r"]; schema_hash = cfg["schema_hash"]
    print(f"checkpoint: mode={cfg.get('mode')} taps={taps} residual_dim={residual_dim} "
          f"horizon={horizon} lora_r={lora_r}")

    # Data — asserts the store matches the checkpoint's schema_hash (no drift).
    store = ConceptStore(args.concept_store, schema_hash=schema_hash)
    layout = store.manifest["per_type"]
    jsons = [args.impromptu_train, args.impromptu_test]
    ds = ConceptDataset(store, args.split, jsons, args.nuscenes_root,
                        with_trajectory=True, horizon=horizon, max_samples=args.max_samples)
    print(f"split={args.split}  n={len(ds)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    # Model — fresh backbone with the SAVED adapter, then load the trained heads.
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
    pos_weight = binary_pos_weight_from_manifest(store.manifest).to(args.device)

    concepts, traj = _eval_all(backbone, cbl, residual, head, loader, args.device,
                               store.manifest, horizon, pos_weight)

    print(f"\n=== {args.split.upper()} RESULTS (n={traj['n']}) ===")
    print(f"TRAJECTORY  ADE={traj['ade_m']:.3f}m  FDE={traj['fde_m']:.3f}m  "
          f"(concept_loss={traj['concept_loss']:.3f}  traj_loss={traj['traj_loss']:.3f})")
    print("CONCEPTS    " + M.summarize(concepts))

    out = args.out or (args.checkpoint_dir.parent / f"eval_{args.split}.json")
    out.write_text(json.dumps({
        "split": args.split, "n": traj["n"], "checkpoint": str(args.checkpoint_dir),
        "trajectory": traj, "concepts": concepts}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
