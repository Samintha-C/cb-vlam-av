"""Visualize mined concepts next to the front-camera image for a given sample.

Usage:
    python scripts/visualize_concepts.py \
        --dataset_path ./outputs/pass_a_trainval_test \
        --data_root /path/to/nuscenes \
        --index 0 \
        --out ./outputs/concept_vis.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from datasets import load_from_disk


# Concepts that are binary (0.0 or 1.0) — rendered differently from continuous
_BINARY_CONCEPTS = {
    "ego_stopped", "ego_braking", "ego_turning",
    "lead_vehicle_present", "lead_vehicle_decelerating",
    "pedestrian_in_crosswalk_ahead", "cyclist_present",
    "left_lane_blocked", "right_lane_blocked",
    "approaching_intersection", "lane_available_left",
    "lane_available_right", "in_intersection",
}


def visualize(record: dict, image_path: Path, out_path: Path) -> None:
    concepts: dict = record["concepts"]
    names = list(concepts.keys())
    values = [concepts[k] for k in names]

    # Color: binary concepts in steel blue, continuous in coral
    colors = ["steelblue" if n in _BINARY_CONCEPTS else "tomato" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: front camera image ──────────────────────────────────────────────
    ax_img = axes[0]
    if image_path.exists():
        img = np.asarray(Image.open(image_path).convert("RGB"))
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "image not available",
                    ha="center", va="center", color="white", transform=ax_img.transAxes)
    ax_img.axis("off")
    ax_img.set_title(
        f"scene: {record['scene_token'][:8]}…\n"
        f"t={record['timestamp']}  frame={record['frame_index']}",
        color="white", fontsize=9, pad=6,
    )

    # ── Right: concept bar chart ──────────────────────────────────────────────
    ax_bar = axes[1]
    ax_bar.set_facecolor("#16213e")
    y = np.arange(len(names))
    bars = ax_bar.barh(y, values, color=colors, edgecolor="none", height=0.7)

    # Value labels at bar end
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", color="white", fontsize=7,
        )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=8, color="white")
    ax_bar.set_xlim(-1.05, 1.15)
    ax_bar.set_xlabel("Concept value", color="white", fontsize=9)
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444466")
    ax_bar.axvline(0, color="#666688", linewidth=0.8)
    ax_bar.set_title("Mined concepts (Pass A)", color="white", fontsize=10, pad=6)

    legend = [
        mpatches.Patch(color="tomato",    label="continuous"),
        mpatches.Patch(color="steelblue", label="binary"),
    ]
    ax_bar.legend(handles=legend, loc="lower right", fontsize=8,
                  facecolor="#1a1a2e", edgecolor="none", labelcolor="white")

    plt.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to HuggingFace dataset saved by mine_concepts.py")
    parser.add_argument("--data_root", required=True,
                        help="nuScenes data root (for loading front camera images)")
    parser.add_argument("--index", type=int, default=0,
                        help="Record index in the dataset to visualize")
    parser.add_argument("--sample_token", type=str, default=None,
                        help="nuScenes sample_token to visualize (overrides --index)")
    parser.add_argument("--out", type=str, default="./outputs/concept_vis.png")
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)

    if args.sample_token:
        matches = [i for i, r in enumerate(ds) if r["sample_token"] == args.sample_token]
        if not matches:
            raise ValueError(f"sample_token {args.sample_token!r} not found in dataset")
        idx = matches[0]
    else:
        idx = args.index

    record = ds[idx]
    print(f"Visualizing index={idx}  sample_token={record['sample_token']}")
    print(f"Concepts: {record['concepts']}")

    # Resolve front camera image path from nuScenes sample_data
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version="v1.0-trainval",
            dataroot=args.data_root,
            verbose=False,
        )
        sample = nusc.get("sample", record["sample_token"])
        cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        image_path = Path(args.data_root) / cam_sd["filename"]
    except Exception as e:
        print(f"Warning: could not load image path from nuScenes ({e}). Rendering without image.")
        image_path = Path("/nonexistent")

    visualize(record, image_path, Path(args.out))


if __name__ == "__main__":
    main()
