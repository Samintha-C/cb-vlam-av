"""Visualize a trained generation checkpoint's PREDICTIONS next to the frame.

Like scripts/visualize_concepts.py, but instead of the mined ground-truth it shows
what the model produces, one page per sample:

    [ front camera ] [ BEV trajectory: pred vs GT ] [ concepts: pred vs GT ]

  - BEV panel: predicted 6 waypoints vs the GT waypoints (ego at origin), with the
    per-sample ADE in the title.
  - Concept panel: predicted activation vs GT per concept (continuous value,
    binary probability, categorical class), each row colored by correctness
    (green = match / red = miss; grey = not supervised for this sample).

Model dims + schema_hash are read from the checkpoint's config.json (same as
cb_vlam.eval.eval_gen), so the viz can't drift from how the model was trained.

Usage: see naut/viz-gen-regression.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.utils.data import DataLoader

from cb_vlam.data.concept_store import ConceptStore
from cb_vlam.models.backbone import CBVLAMBackbone
from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.models.residual import UnsupervisedResidual
from cb_vlam.models.final_predictor import FinalPredictor
from cb_vlam.training.dataset import ConceptDataset, collate
from cb_vlam.concept_mining.schema import get_all_concepts

_CAT_VALUES = {c["name"]: c.get("values", [])
               for c in get_all_concepts() if c["type"] == "categorical"}

_BG, _PANEL, _GRID = "#1a1a2e", "#16213e", "#444466"


def _cat_label(name, idx):
    vals = _CAT_VALUES.get(name, [])
    idx = int(round(idx))
    return vals[idx] if 0 <= idx < len(vals) else f"?{idx}"


def _bev_panel(ax, pred_wp, gt_wp, wp_valid):
    """Top-down ego frame: x forward (up), y lateral. Ego at origin."""
    ax.set_facecolor(_PANEL)
    g = gt_wp[wp_valid]
    p = pred_wp[wp_valid]
    # plot lateral (y) on horizontal axis, forward (x) on vertical axis
    if len(g):
        ax.plot(g[:, 1], g[:, 0], "-o", color="mediumseagreen", ms=5, lw=2, label="GT")
    if len(p):
        ax.plot(p[:, 1], p[:, 0], "-o", color="tomato", ms=5, lw=2, label="pred")
    ax.plot(0, 0, "^", color="white", ms=11, label="ego")
    ax.axhline(0, color="#666688", lw=0.6); ax.axvline(0, color="#666688", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()  # left-positive y to the left, matching driver's view
    ax.set_xlabel("lateral y (m)", color="white", fontsize=9)
    ax.set_ylabel("forward x (m)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)
    if len(g) and len(p):
        ade = float(np.linalg.norm(p - g, axis=1).mean())
        fde = float(np.linalg.norm(p[-1] - g[-1]))
        ax.set_title(f"trajectory  ADE={ade:.2f}m  FDE={fde:.2f}m",
                     color="white", fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=8, facecolor=_BG, edgecolor="none",
              labelcolor="white")


def _rows(layout, cbl_out, targets):
    """Build (name, kind, pred_disp, gt_disp, bar_val, gt_bar, correct, supervised) rows."""
    rows = []
    cont = cbl_out["continuous"][0].float().cpu().numpy()
    binp = torch.sigmoid(cbl_out["binary_logits"][0]).float().cpu().numpy()
    cat_pred = [int(lg[0].argmax()) for lg in cbl_out["categorical_logits"]]

    ct, cm = targets["continuous"][0].numpy(), targets["continuous_mask"][0].numpy()
    bt, bm = targets["binary"][0].numpy(), targets["binary_mask"][0].numpy()
    gt_cat, catm = targets["categorical"][0].numpy(), targets["categorical_mask"][0].numpy()

    for j, n in enumerate(layout["continuous"]["names"]):
        sup = bool(cm[j])
        rows.append((n, "cont", f"{cont[j]:.2f}", f"{ct[j]:.2f}" if sup else "—",
                     float(cont[j]), float(ct[j]) if sup else 0.0,
                     abs(cont[j] - ct[j]) < 0.15 if sup else None, sup))
    for j, n in enumerate(layout["binary"]["names"]):
        sup = bool(bm[j])
        pr = float(binp[j]); gt = float(bt[j])
        rows.append((n, "bin", f"{pr:.2f}", f"{int(gt)}" if sup else "—",
                     pr, gt, ((pr >= 0.5) == (gt >= 0.5)) if sup else None, sup))
    for j, n in enumerate(layout["categorical"]["names"]):
        sup = bool(catm[j])
        pr, gt = cat_pred[j], int(gt_cat[j])
        rows.append((n, "cat", _cat_label(n, pr), _cat_label(n, gt) if sup else "—",
                     float(pr), float(gt) if sup else 0.0,
                     (pr == gt) if sup else None, sup))
    return rows


def _concept_panel(ax, rows):
    ax.set_facecolor(_PANEL)
    y = np.arange(len(rows))
    for i, (n, kind, pd, gd, bar, gtbar, correct, sup) in enumerate(rows):
        if not sup:
            color = "#555566"
        elif correct:
            color = "mediumseagreen"
        else:
            color = "tomato"
        ax.barh(i, bar, color=color, edgecolor="none", height=0.7)
        if sup:
            ax.plot(gtbar, i, "|", color="white", ms=12, mew=2)   # GT tick
        ax.text(max(bar, gtbar) + 0.05, i, f"p={pd} g={gd}",
                va="center", ha="left", color="white", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5, color="white")
    ax.set_xlim(-1.2, 3.4)
    ax.axvline(0, color="#666688", lw=0.8)
    ax.set_xlabel("predicted activation (bar) vs GT (tick)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)
    legend = [mpatches.Patch(color="mediumseagreen", label="correct"),
              mpatches.Patch(color="tomato", label="wrong"),
              mpatches.Patch(color="#555566", label="not supervised")]
    ax.legend(handles=legend, loc="lower right", fontsize=7,
              facecolor=_BG, edgecolor="none", labelcolor="white")
    ax.set_title("concepts: predicted vs GT", color="white", fontsize=10, pad=6)


def _render_page(image, token, pred_wp, gt_wp, wp_valid, rows):
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(22, max(7, n * 0.33)),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.2]})
    fig.patch.set_facecolor(_BG)
    ax_img, ax_bev, ax_con = axes

    if image is not None:
        ax_img.imshow(np.asarray(image))
    else:
        ax_img.text(0.5, 0.5, "no image", ha="center", va="center",
                    color="white", transform=ax_img.transAxes)
        ax_img.set_facecolor(_PANEL)
    ax_img.axis("off")
    ax_img.set_title(f"sample {token[:16]}…", color="white", fontsize=9, pad=6)

    _bev_panel(ax_bev, pred_wp, gt_wp, wp_valid)
    _concept_panel(ax_con, rows)
    plt.tight_layout(pad=1.5)
    return fig


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--concept_store", required=True, type=Path)
    ap.add_argument("--impromptu_train", required=True, type=Path)
    ap.add_argument("--impromptu_test", required=True, type=Path)
    ap.add_argument("--nuscenes_root", required=True, type=Path)
    ap.add_argument("--processor_name", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n_samples", type=int, default=30, help="pages in the PDF")
    ap.add_argument("--out", type=Path, default=Path("./outputs/predictions.pdf"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint_dir / "config.json").read_text())
    taps, residual_dim = cfg["feature_taps"], cfg["residual_dim"]
    horizon, lora_r, schema_hash = cfg["horizon"], cfg["lora_r"], cfg["schema_hash"]

    store = ConceptStore(args.concept_store, schema_hash=schema_hash)
    layout = store.manifest["per_type"]
    ds = ConceptDataset(store, args.split, [args.impromptu_train, args.impromptu_test],
                        args.nuscenes_root, load_image=True, with_trajectory=True,
                        horizon=horizon, max_samples=args.n_samples)
    print(f"split={args.split}  rendering {len(ds)} pages")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)

    backbone = CBVLAMBackbone(
        checkpoint_path=args.base_checkpoint, feature_taps=taps,
        processor_path=args.processor_name, dtype=args.dtype, lora_r=lora_r,
        adapter_path=str(args.checkpoint_dir / "lora_adapter"), device=args.device,
        gradient_checkpointing=False)
    cbl = ConceptBottleneckLayer(in_dim=backbone.feature_dim, layout=layout).to(args.device)
    residual = UnsupervisedResidual(in_dim=backbone.feature_dim, residual_dim=residual_dim).to(args.device)
    head = FinalPredictor(cbl.activation_dim, residual_dim, output_dim=horizon * 2,
                          mode="regression").to(args.device)
    cbl.load_state_dict(torch.load(args.checkpoint_dir / "cbl.pt", map_location=args.device))
    residual.load_state_dict(torch.load(args.checkpoint_dir / "residual.pt", map_location=args.device))
    head.load_state_dict(torch.load(args.checkpoint_dir / "final_predictor.pt", map_location=args.device))
    cbl.eval(); residual.eval(); head.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        for k, batch in enumerate(loader):
            feats = torch.stack(
                [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])], dim=0)
            cbl_out = cbl(feats)
            traj = head(cbl.to_activations(cbl_out), residual(feats))[0].float().cpu().numpy()

            pred_wp = traj.reshape(horizon, 2)
            gt_wp = batch["trajectory"][0].numpy().reshape(horizon, 2)
            wp_valid = batch["trajectory_mask"][0].numpy().reshape(horizon, 2)[:, 0]
            rows = _rows(layout, cbl_out, batch["targets"])

            fig = _render_page(batch["images"][0], batch["sample_tokens"][0],
                               pred_wp, gt_wp, wp_valid, rows)
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            if (k + 1) % 5 == 0:
                print(f"  {k + 1}/{len(ds)} pages")

    print(f"\nSaved {len(ds)}-page PDF → {args.out}")


if __name__ == "__main__":
    main()
