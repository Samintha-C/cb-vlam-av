"""Continuous-regression generation training.

End-to-end joint training, mirroring CB-LLM (generation/train_CBLLM.py): a FRESH
LoRA backbone + FRESH CBL / residual / adversary / final-linear heads, all
trainable, with every loss live from step 1 — no concept-projection warm-start,
no frozen features, no activation cache.

Data flow (single scene tap → one trajectory per sample):

    image+prompt ─► backbone(LoRA) ─► h
                                       ├─► CBL ─────► ĉ ──(to_activations)──┐
                                       └─► Residual ─► r ──────────────────┤
                                                       │                   ▼
                                              Adversary(r)           FinalPredictor([ĉ⊕r]) ─► 6×2 waypoints

Disentanglement follows CB-LLM (train_CBLLM.py:154-175) via gradient CONFINEMENT
rather than a fused gradient-reversal term — three scoped backwards per step:

  1. MAIN      L_main = L_c + λ_t·L_t + λ_reg·R       → backbone + CBL + residual + head
               masked concept loss + smooth-L1 waypoints + elastic-net sparsity.
  2. ADVERSARY L_adv  = concept_loss(adv(r.detach())) → ONLY the adversary
               the probe learns to read concepts off r, touching nothing else.
  3. DISENTANGLE L_dis = disentanglement_loss(adv(residual(h.detach())))
               → ONLY the residual; pushes r toward the probe's uninformative
               prior (bounded entropy / marginal-mean) so concepts stay the
               steerable handle. h is detached so the backbone is never moved by
               the adversarial signal — that confinement is what makes it stable
               (the earlier fused-GRL form diverged as λ ramped up).

Best-model selection uses L_c + λ_t·L_t only (CB-LLM train_CBLLM.py:220) — the
adversarial terms are diagnostics, not part of model quality.

Usage: see naut/train-gen-regression-capped.yaml.
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
from cb_vlam.models.residual import UnsupervisedResidual
from cb_vlam.models.adversarial import AdversarialDiscriminator
from cb_vlam.models.final_predictor import FinalPredictor
from cb_vlam.training.dataset import ConceptDataset, collate
from cb_vlam.training.losses import (
    concept_loss, adversary_loss, disentanglement_loss, trajectory_loss,
    elastic_net_penalty, binary_pos_weight_from_manifest)


class GenerationModules:
    """Holds the four trainable head modules so they pass around as a unit."""
    def __init__(self, cbl, residual, adversary, head):
        self.cbl, self.residual, self.adversary, self.head = cbl, residual, adversary, head

    def modules(self):
        return [self.cbl, self.residual, self.adversary, self.head]

    def parameters(self):
        for m in self.modules():
            yield from m.parameters()

    def train(self):
        for m in self.modules():
            m.train()

    def eval(self):
        for m in self.modules():
            m.eval()


def _adv_lambda(step: int, args) -> float:
    """Optional linear 0→lambda_adv ramp over adv_warmup steps, then hold.

    CB-LLM uses no ramp (effective weight 1.0). With confinement the disentangle
    term cannot destabilize the modules, so a ramp is only a mild easing knob;
    adv_warmup=0 reproduces CB-LLM exactly.
    """
    if args.adv_warmup <= 0:
        return args.lambda_adv
    return args.lambda_adv * min(1.0, step / args.adv_warmup)


def _backbone_feats(backbone, batch):
    """Stack the per-sample backbone features (the one expensive forward)."""
    return torch.stack(
        [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])],
        dim=0)


def _predict(modules: GenerationModules, feats):
    """CBL + residual + head forward → (cbl_out, r, traj). Used by main + eval."""
    cbl_out = modules.cbl(feats)
    cvec = modules.cbl.to_activations(cbl_out)
    r = modules.residual(feats)
    traj = modules.head(cvec, r)
    return cbl_out, r, traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_train", required=True, type=Path)
    ap.add_argument("--impromptu_test", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--schema_hash", default=None)
    ap.add_argument("--feature_taps", default="endprompt_final")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_image_pixels", type=int, default=262144)
    ap.add_argument("--horizon", type=int, default=6, help="future waypoints (Impromptu=6)")
    ap.add_argument("--residual_dim", type=int, default=128)
    ap.add_argument("--lambda_traj", type=float, default=1.0)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--lambda_adv", type=float, default=1.0,
                    help="Weight on the confined disentanglement term. CB-LLM uses "
                         "1.0; safe here because the term is gradient-confined to the "
                         "residual (it cannot corrupt the backbone/concept path), so "
                         "the divergence the fused-GRL form showed cannot recur.")
    ap.add_argument("--adv_warmup", type=int, default=0,
                    help="opt steps to ramp lambda_adv 0→max (0 = CB-LLM's no-ramp).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=750)
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
                              with_trajectory=True, horizon=args.horizon,
                              max_samples=args.max_train_samples)
    val_ds = ConceptDataset(store, "val", jsons, args.nuscenes_root,
                            with_trajectory=True, horizon=args.horizon,
                            max_samples=args.max_val_samples)
    print(f"train={len(train_ds)} val={len(val_ds)} samples")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    # ── Model: fresh LoRA backbone + fresh modules (everything trainable) ────────
    backbone = CBVLAMBackbone(
        checkpoint_path=args.checkpoint, feature_taps=taps,
        processor_path=args.processor_name, dtype=args.dtype,
        lora_r=args.lora_r, max_image_pixels=args.max_image_pixels, device=args.device)
    backbone.model.print_trainable_parameters()

    layout = store.manifest["per_type"]
    cbl = ConceptBottleneckLayer(in_dim=backbone.feature_dim, layout=layout).to(args.device)
    residual = UnsupervisedResidual(in_dim=backbone.feature_dim,
                                    residual_dim=args.residual_dim).to(args.device)
    adversary = AdversarialDiscriminator(residual_dim=args.residual_dim,
                                         layout=layout).to(args.device)
    head = FinalPredictor(cbl.activation_dim, args.residual_dim,
                          output_dim=args.horizon * 2, mode="regression").to(args.device)
    modules = GenerationModules(cbl, residual, adversary, head)
    pos_weight = binary_pos_weight_from_manifest(store.manifest).to(args.device)
    print(f"feature_dim={backbone.feature_dim} activation_dim={cbl.activation_dim} "
          f"residual_dim={args.residual_dim} traj_dim={args.horizon*2}")

    params = list(backbone.trainable_parameters()) + list(modules.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"optimizer steps: ~{total_steps} ({len(train_loader)} batches/epoch, "
          f"grad_accum={args.grad_accum})")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val = math.inf
    step = micro = samples_seen = 0
    win = {"main": 0.0, "c": 0.0, "t": 0.0, "adv": 0.0, "dis": 0.0, "n": 0}
    adv_params = list(modules.adversary.parameters())
    res_params = list(modules.residual.parameters())
    opt.zero_grad()
    backbone.model.train(); modules.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch in train_loader:
            targets = {k: v.to(args.device) for k, v in batch["targets"].items()}
            traj_gt = batch["trajectory"].to(args.device)
            traj_mask = batch["trajectory_mask"].to(args.device)
            feats = _backbone_feats(backbone, batch)        # one expensive forward
            lam = _adv_lambda(step, args)

            # 1) MAIN — concepts + trajectory + sparsity → backbone, CBL, residual, head
            cbl_out, r, traj = _predict(modules, feats)
            Lc = concept_loss(cbl_out, targets, bin_pos_weight=pos_weight)["total"]
            Lt = trajectory_loss(traj, traj_gt, mask=traj_mask)
            Lreg = elastic_net_penalty(modules.head.concept_weight)
            Lmain = Lc + args.lambda_traj * Lt + args.lambda_reg * Lreg
            (Lmain / args.grad_accum).backward()

            # 2) ADVERSARY probe — read concepts off r → ONLY the adversary's params
            adv_out = modules.adversary(r.detach())
            Ladv = adversary_loss(adv_out, targets, bin_pos_weight=pos_weight)["total"]
            (Ladv / args.grad_accum).backward(inputs=adv_params)

            # 3) DISENTANGLE — push r to the probe's uninformative prior → ONLY residual.
            #    feats detached so the adversarial signal never reaches the backbone.
            dis_out = modules.adversary(modules.residual(feats.detach()))
            Ldis = disentanglement_loss(dis_out, targets)["total"]
            (lam * Ldis / args.grad_accum).backward(inputs=res_params)

            micro += 1
            samples_seen += len(batch["prompts"])
            win["main"] += Lmain.item(); win["c"] += Lc.item(); win["t"] += Lt.item()
            win["adv"] += Ladv.item(); win["dis"] += Ldis.item(); win["n"] += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1

                if step % 50 == 0:
                    n = max(win["n"], 1)
                    rate = samples_seen / (time.time() - t0)
                    print(f"e{epoch} step{step}  main={win['main']/n:.4f} "
                          f"(c={win['c']/n:.3f} t={win['t']/n:.3f})  "
                          f"adv={win['adv']/n:.3f} dis={win['dis']/n:.3f}  "
                          f"λadv={lam:.2f} lr={sched.get_last_lr()[0]:.2e}  {rate:.2f} smp/s "
                          f"[mean/{n}]", flush=True)
                    win = {"main": 0.0, "c": 0.0, "t": 0.0, "adv": 0.0, "dis": 0.0, "n": 0}

                if step % args.eval_every == 0:
                    best_val = _do_eval(backbone, modules, val_loader, store, pos_weight,
                                        args, taps, step, best_val)
                    backbone.model.train(); modules.train()

    _do_eval(backbone, modules, val_loader, store, pos_weight, args, taps, step, best_val, final=True)
    print("Training done.")


@torch.no_grad()
def _do_eval(backbone, modules: GenerationModules, val_loader, store, pos_weight, args, taps,
             step, best_val, final=False):
    backbone.model.eval(); modules.eval()
    device = args.device
    # val_quality = L_c + λ_t·L_t (CB-LLM selects on concept+task only, :220);
    # adv/dis are adversarial diagnostics, NOT part of model quality.
    tot = {"q": 0.0, "c": 0.0, "t": 0.0, "adv": 0.0, "dis": 0.0, "n": 0}
    ade_sum = fde_sum = ade_n = fde_n = 0.0
    for batch in val_loader:
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        feats = _backbone_feats(backbone, batch)
        cbl_out, r, traj = _predict(modules, feats)
        gt = batch["trajectory"].to(device); m = batch["trajectory_mask"].to(device)
        Lc = concept_loss(cbl_out, targets, bin_pos_weight=pos_weight)["total"]
        Lt = trajectory_loss(traj, gt, mask=m)
        adv_out = modules.adversary(r)
        Ladv = adversary_loss(adv_out, targets, bin_pos_weight=pos_weight)["total"]
        Ldis = disentanglement_loss(adv_out, targets)["total"]
        q = Lc + args.lambda_traj * Lt
        bs = len(batch["prompts"])
        tot["q"] += q.item() * bs; tot["c"] += Lc.item() * bs; tot["t"] += Lt.item() * bs
        tot["adv"] += Ladv.item() * bs; tot["dis"] += Ldis.item() * bs; tot["n"] += bs

        # ADE/FDE in meters over valid waypoints.
        B = traj.shape[0]
        p = traj.reshape(B, args.horizon, 2); g = gt.reshape(B, args.horizon, 2)
        wp_valid = m.reshape(B, args.horizon, 2)[..., 0]          # (B, H)
        d = ((p - g) ** 2).sum(-1).clamp(min=0).sqrt()           # (B, H) meters
        ade_sum += float((d * wp_valid).sum()); ade_n += float(wp_valid.sum())
        last_valid = wp_valid[:, -1]
        fde_sum += float((d[:, -1] * last_valid).sum()); fde_n += float(last_valid.sum())

    n = max(tot["n"], 1)
    val_q = tot["q"] / n
    ade = ade_sum / max(ade_n, 1.0); fde = fde_sum / max(fde_n, 1.0)
    print(f"\n[eval @ step {step}{' FINAL' if final else ''}] "
          f"val_quality={val_q:.4f} (c={tot['c']/n:.3f} t={tot['t']/n:.3f} | "
          f"adv={tot['adv']/n:.3f} dis={tot['dis']/n:.3f})  "
          f"ADE={ade:.3f}m FDE={fde:.3f}m  n={n}\n", flush=True)
    (args.output_dir / f"metrics_step{step}.json").write_text(json.dumps({
        "step": step, "val_quality": val_q, "concept_loss": tot["c"] / n,
        "traj_loss": tot["t"] / n, "adv_loss": tot["adv"] / n, "dis_loss": tot["dis"] / n,
        "ade_m": ade, "fde_m": fde, "n": n}, indent=2))

    if val_q < best_val:
        _save(args.output_dir, backbone, modules, store, args, taps)
        print(f"  new best val_quality {val_q:.4f} → saved", flush=True)
        return val_q
    return best_val


def _save(out_dir: Path, backbone, modules: GenerationModules, store, args, taps):
    ckpt = out_dir / "best"; ckpt.mkdir(parents=True, exist_ok=True)
    backbone.model.save_pretrained(str(ckpt / "lora_adapter"))
    torch.save(modules.cbl.state_dict(), ckpt / "cbl.pt")
    torch.save(modules.residual.state_dict(), ckpt / "residual.pt")
    torch.save(modules.adversary.state_dict(), ckpt / "adversary.pt")
    torch.save(modules.head.state_dict(), ckpt / "final_predictor.pt")
    (ckpt / "config.json").write_text(json.dumps({
        "mode": "continuous_regression",
        "disentangle": "cbllm_confined",
        "schema_hash": store.manifest["schema_hash"],
        "schema_version": store.manifest["schema_version"],
        "feature_taps": taps, "feature_dim": backbone.feature_dim,
        "activation_dim": modules.cbl.activation_dim, "residual_dim": args.residual_dim,
        "horizon": args.horizon, "lora_r": args.lora_r, "checkpoint": args.checkpoint,
        "lambda_traj": args.lambda_traj, "lambda_reg": args.lambda_reg,
        "lambda_adv": args.lambda_adv,
    }, indent=2))


if __name__ == "__main__":
    main()
