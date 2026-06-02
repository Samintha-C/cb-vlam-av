"""Phase-1 joint training: LoRA backbone + CBL, supervised by concept loss only.

This trains the bottleneck to project the backbone representation into concept
space. The backbone is LoRA-adapted (base frozen), the CBL projects to the
manifest's per-type concept layout, and the only objective is the masked
per-type concept loss L_c. Downstream residual/adversarial/task terms are out of
scope here — the goal is accurate concept projection.

Batching note: the backbone consumes one (image, prompt) at a time, so a
"batch" is forwarded sample-by-sample (graphs retained) then backpropagated
once; --grad_accum accumulates several such batches before an optimizer step.

Usage (see naut/train-cbl-phase1.yaml):
    python -m cb_vlam.training.train \
        --concept_store  .../impromptu_concepts_ab/mined \
        --impromptu_train .../nuscenes_train.json \
        --impromptu_test  .../nuscenes_test.json \
        --nuscenes_root  /sc-rwx-vol/cbvlam \
        --checkpoint     .../7B_AD_finetune \
        --processor_name Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir     .../runs/phase1 \
        --feature_taps endprompt_final --batch_size 4 --grad_accum 4 --epochs 3
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.backbone import CBVLAMBackbone
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.training.dataset import ConceptDataset, collate
from cb_vlam.training.losses import concept_loss, binary_pos_weight_from_manifest
from cb_vlam.eval.metrics import evaluate, summarize


def _forward_batch(backbone, cbl, batch, device):
    """Forward each (image, prompt) in the batch, stack, project through the CBL."""
    feats = torch.stack(
        [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])],
        dim=0)
    pred = cbl(feats)
    target = {k: v.to(device) for k, v in batch["targets"].items()}
    return pred, target


def _save_checkpoint(out_dir: Path, backbone, cbl, store, args, feature_taps, tag):
    ckpt = out_dir / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    backbone.model.save_pretrained(str(ckpt / "lora_adapter"))
    torch.save(cbl.state_dict(), ckpt / "cbl.pt")
    (ckpt / "config.json").write_text(json.dumps({
        "schema_hash": store.manifest["schema_hash"],
        "schema_version": store.manifest["schema_version"],
        "feature_taps": feature_taps,
        "feature_dim": backbone.feature_dim,
        "lora_r": args.lora_r,
        "checkpoint": args.checkpoint,
    }, indent=2))
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_train", required=True, type=Path)
    ap.add_argument("--impromptu_test", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--feature_taps", default="endprompt_final")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_image_pixels", type=int, default=262144,
                    help="Cap processor max_pixels (default 262144 = Impromptu's "
                         "image_max_pixels). Lower = faster but more downscaling.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--schema_hash", default=None,
                    help="Expected manifest schema_hash — asserts no store/schema drift.")
    ap.add_argument("--eval_every", type=int, default=500, help="optimizer steps between evals")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    taps = [t.strip() for t in args.feature_taps.split(",") if t.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    store = ConceptStore(args.concept_store, schema_hash=args.schema_hash or None)
    jsons = [args.impromptu_train, args.impromptu_test]
    train_ds = ConceptDataset(store, "train", jsons, args.nuscenes_root,
                              max_samples=args.max_train_samples)
    val_ds = ConceptDataset(store, "val", jsons, args.nuscenes_root,
                            max_samples=args.max_val_samples)
    print(f"train={len(train_ds)} val={len(val_ds)} samples")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = CBVLAMBackbone(
        checkpoint_path=args.checkpoint, feature_taps=taps,
        processor_path=args.processor_name, dtype=args.dtype,
        lora_r=args.lora_r, max_image_pixels=args.max_image_pixels,
        device=args.device)
    backbone.model.print_trainable_parameters()
    cbl = ConceptBottleneckLayer(in_dim=backbone.feature_dim,
                                 layout=store.manifest["per_type"]).to(args.device)
    pos_weight = binary_pos_weight_from_manifest(store.manifest).to(args.device)

    params = list(backbone.trainable_parameters()) + list(cbl.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"optimizer steps: ~{total_steps} ({len(train_loader)} batches/epoch, "
          f"grad_accum={args.grad_accum})")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val = math.inf
    step = 0
    micro = 0
    samples_seen = 0
    # Running sums over the log window (reset each print) so the reported loss is
    # the mean over many samples, not one noisy 4-sample microbatch.
    win = {"total": 0.0, "c": 0.0, "b": 0.0, "k": 0.0, "n": 0}
    opt.zero_grad()
    backbone.model.train(); cbl.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch in train_loader:
            pred, target = _forward_batch(backbone, cbl, batch, args.device)
            losses = concept_loss(pred, target, bin_pos_weight=pos_weight)
            (losses["total"] / args.grad_accum).backward()
            micro += 1
            samples_seen += len(batch["prompts"])
            win["total"] += losses["total"].item(); win["c"] += losses["continuous"].item()
            win["b"] += losses["binary"].item(); win["k"] += losses["categorical"].item()
            win["n"] += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1

                if step % 50 == 0:
                    n = max(win["n"], 1)
                    rate = samples_seen / (time.time() - t0)
                    print(f"e{epoch} step{step}  loss={win['total']/n:.4f} "
                          f"(c={win['c']/n:.3f} b={win['b']/n:.3f} k={win['k']/n:.3f})  "
                          f"lr={sched.get_last_lr()[0]:.2e}  {rate:.2f} smp/s "
                          f"[mean/{n} microbatches]", flush=True)
                    win = {"total": 0.0, "c": 0.0, "b": 0.0, "k": 0.0, "n": 0}

                if step % args.eval_every == 0:
                    best_val = _do_eval(backbone, cbl, val_loader, store, pos_weight,
                                        args, taps, step, best_val)
                    backbone.model.train(); cbl.train()

    # Final eval + checkpoint
    _do_eval(backbone, cbl, val_loader, store, pos_weight, args, taps, step, best_val,
             final=True)
    print("Training done.")


def _do_eval(backbone, cbl, val_loader, store, pos_weight, args, taps, step,
             best_val, final=False):
    metrics = evaluate(backbone, cbl, val_loader, args.device, store.manifest,
                       bin_pos_weight=pos_weight)
    print(f"\n[eval @ step {step}{' FINAL' if final else ''}] {summarize(metrics)}\n", flush=True)
    (args.output_dir / f"metrics_step{step}.json").write_text(json.dumps(metrics, indent=2))

    val_loss = metrics.get("loss", math.inf)
    if val_loss < best_val:
        ckpt = _save_checkpoint(args.output_dir, backbone, cbl, store, args, taps, "best")
        print(f"  new best val loss {val_loss:.4f} → saved {ckpt}", flush=True)
        return val_loss
    return best_val


if __name__ == "__main__":
    main()
