"""GPU smoke test: LoRA backbone → CBL → concept loss → backward.

De-risks the joint-training forward before any full training loop is written.
On a few real samples it verifies, end to end, that:

  1. the Qwen2.5-VL checkpoint loads and LoRA-wraps,
  2. the differentiable forward produces a grad-carrying feature at the chosen
     tap (default endprompt_final),
  3. the CBL projects it into the manifest's per-type concept space,
  4. the masked concept loss computes against real ConceptStore targets, and
  5. backward populates gradients on BOTH the LoRA adapters and the CBL heads
     (i.e. concept-loss gradient actually reaches the backbone).

Run via naut/smoke-backbone-cbl.yaml. Exits non-zero on any failed assertion.

Usage:
    python scripts/smoke_backbone_cbl.py \
        --concept_store /sc-rwx-vol/cbvlam/outputs/impromptu_concepts_ab/mined \
        --impromptu_json /sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_train.json \
        --nuscenes_root /sc-rwx-vol/cbvlam \
        --checkpoint /sc-rwx-vol/cbvlam/checkpoints/7B_AD_finetune \
        --processor_name Qwen/Qwen2.5-VL-7B-Instruct \
        --feature_taps endprompt_final \
        --n_samples 4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.backbone import CBVLAMBackbone
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.training.losses import concept_loss, binary_pos_weight_from_manifest


def _collate(store: ConceptStore, tokens):
    """Stack per-sample ConceptStore targets into batched (B, n_*) tensors."""
    rows = [store.get(t) for t in tokens]
    keys = ["continuous", "continuous_mask", "binary", "binary_mask",
            "categorical", "categorical_mask"]
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_json", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--feature_taps", default="endprompt_final",
                    help="Comma-separated taps: endprompt_final,endprompt_penult,afterplan_final")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_image_pixels", type=int, default=262144,
                    help="Cap processor max_pixels (default = Impromptu's 262144).")
    args = ap.parse_args()

    taps = [t.strip() for t in args.feature_taps.split(",") if t.strip()]

    # ── 1. Concept store + a few sample tokens that have both inputs + labels ──
    print(f"Loading ConceptStore from {args.concept_store}")
    store = ConceptStore(args.concept_store)
    print(f"  {store.n_continuous} continuous, {store.n_binary} binary, "
          f"{store.n_categorical} categorical concepts")

    with open(args.impromptu_json) as f:
        records = json.load(f)

    picked = []
    for rec in records:
        tok = rec["id"]
        if tok in store:
            picked.append(rec)
        if len(picked) >= args.n_samples:
            break
    if len(picked) < args.n_samples:
        raise SystemExit(f"Only found {len(picked)} usable samples; need {args.n_samples}.")
    tokens = [r["id"] for r in picked]
    print(f"  using {len(tokens)} samples: {[t[:10] for t in tokens]}")

    # ── 2. Backbone (LoRA) + CBL ──────────────────────────────────────────────
    print(f"\nLoading LoRA backbone (taps={taps}) ...")
    backbone = CBVLAMBackbone(
        checkpoint_path=args.checkpoint,
        feature_taps=taps,
        processor_path=args.processor_name,
        dtype=args.dtype,
        lora_r=args.lora_r,
        max_image_pixels=args.max_image_pixels,
        device=args.device,
    )
    backbone.model.print_trainable_parameters()
    print(f"  feature_dim = {backbone.feature_dim}")

    cbl = ConceptBottleneckLayer(in_dim=backbone.feature_dim,
                                 layout=store.manifest["per_type"]).to(args.device)
    n_cbl = sum(p.numel() for p in cbl.parameters())
    print(f"  CBL params = {n_cbl:,}")

    pos_weight = binary_pos_weight_from_manifest(store.manifest).to(args.device)

    # ── 3. Forward (per sample) → stack → CBL → loss → backward ───────────────
    backbone.model.train(); cbl.train()
    params = list(backbone.trainable_parameters()) + list(cbl.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    opt.zero_grad()

    feats = []
    for rec in picked:
        image_path = args.nuscenes_root / rec["images"][0]
        prompt = rec["messages"][0]["content"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            feats.append(backbone(image, prompt))     # (feature_dim,) with grad
    feats = torch.stack(feats, dim=0)                 # (B, feature_dim)
    print(f"\n  stacked features: {tuple(feats.shape)}  dtype={feats.dtype}  "
          f"requires_grad={feats.requires_grad}")

    target = {k: v.to(args.device) for k, v in _collate(store, tokens).items()}
    pred = cbl(feats)
    losses = concept_loss(pred, target, bin_pos_weight=pos_weight)
    print(f"  loss: total={losses['total'].item():.4f}  "
          f"cont={losses['continuous'].item():.4f}  "
          f"bin={losses['binary'].item():.4f}  "
          f"cat={losses['categorical'].item():.4f}")

    losses["total"].backward()

    # ── 4. Assert gradients reached LoRA adapters AND the CBL ─────────────────
    lora_grad = _grad_norm(backbone.trainable_parameters())
    cbl_grad = _grad_norm(cbl.parameters())
    n_lora_with_grad = sum(1 for p in backbone.trainable_parameters()
                           if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"\n  LoRA grad norm = {lora_grad:.6f}  ({n_lora_with_grad} adapters with nonzero grad)")
    print(f"  CBL  grad norm = {cbl_grad:.6f}")

    assert torch.isfinite(losses["total"]), "loss is not finite"
    assert cbl_grad > 0, "CBL received no gradient"
    assert lora_grad > 0, "LoRA adapters received no gradient — concept loss did not reach the backbone"

    opt.step()
    print("\nSMOKE PASSED: differentiable LoRA backbone → CBL → concept loss → backward works.")


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return total ** 0.5


if __name__ == "__main__":
    main()
