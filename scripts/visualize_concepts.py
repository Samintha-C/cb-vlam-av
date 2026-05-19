"""Visualize mined concepts next to the front-camera image.

Produces a multi-page PDF — one page per sample — saved to --out.

Usage:
    python scripts/visualize_concepts.py \
        --dataset_path ./outputs/pass_a_trainval_test \
        --data_root /sc-rwx-vol/cbvlam/nuscenes \
        --n_samples 20 \
        --out ./outputs/concepts.pdf
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of how this script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
from datasets import load_from_disk


from cb_vlam.concept_mining.schema import get_all_concepts

_CONCEPT_INFO = {c["name"]: c for c in get_all_concepts()}
_BINARY_CONCEPTS = {n for n, c in _CONCEPT_INFO.items() if c["type"] == "binary"}
_CATEGORICAL_CONCEPTS = {n: c["values"] for n, c in _CONCEPT_INFO.items() if c["type"] == "categorical"}


def _categorical_label(name: str, val: float) -> str:
    """Lookup the categorical string for a (concept, integer-index) pair."""
    values = _CATEGORICAL_CONCEPTS.get(name, [])
    idx = int(round(val))
    if 0 <= idx < len(values):
        return values[idx]
    return f"?{idx}"


def _render_page(record: dict, image_path: Path) -> plt.Figure:
    concepts: dict = record["concepts"]
    names = list(concepts.keys())
    values = [concepts[k] for k in names]

    def color_for(n):
        if n in _CATEGORICAL_CONCEPTS:
            return "mediumseagreen"
        if n in _BINARY_CONCEPTS:
            return "steelblue"
        return "tomato"
    colors = [color_for(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: front camera image
    ax_img = axes[0]
    if image_path.exists():
        ax_img.imshow(np.asarray(Image.open(image_path).convert("RGB")))
    else:
        ax_img.text(0.5, 0.5, "image not extracted yet",
                    ha="center", va="center", color="white", fontsize=10,
                    transform=ax_img.transAxes)
        ax_img.set_facecolor("#16213e")
    ax_img.axis("off")
    ax_img.set_title(
        f"scene: {record['scene_token'][:12]}…  frame {record['frame_index']}\n"
        f"sample: {record['sample_token'][:12]}…",
        color="white", fontsize=8, pad=6,
    )

    # Right: horizontal bar chart
    ax_bar = axes[1]
    ax_bar.set_facecolor("#16213e")
    y = np.arange(len(names))
    bars = ax_bar.barh(y, values, color=colors, edgecolor="none", height=0.7)

    for bar, name, val in zip(bars, names, values):
        if name in _CATEGORICAL_CONCEPTS:
            txt = _categorical_label(name, val)
        else:
            txt = f"{val:.3f}"
        ax_bar.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            txt, va="center", ha="left", color="white", fontsize=7,
        )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=7, color="white")
    # x-range covers categorical indices too (up to 5 for speed_limit_sign)
    max_cat = max((len(v) for v in _CATEGORICAL_CONCEPTS.values()), default=1)
    ax_bar.set_xlim(-1.1, max(1.25, max_cat + 0.5))
    ax_bar.set_xlabel("Concept value", color="white", fontsize=9)
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444466")
    ax_bar.axvline(0, color="#666688", linewidth=0.8)
    ax_bar.set_title("Mined concepts (A: kinematic, B: agents, C: VLM)",
                     color="white", fontsize=10, pad=6)

    legend = [
        mpatches.Patch(color="tomato",         label="continuous"),
        mpatches.Patch(color="steelblue",      label="binary"),
        mpatches.Patch(color="mediumseagreen", label="categorical"),
    ]
    ax_bar.legend(handles=legend, loc="lower right", fontsize=8,
                  facecolor="#1a1a2e", edgecolor="none", labelcolor="white")

    plt.tight_layout(pad=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--version",      default="v1.0-trainval")
    parser.add_argument("--n_samples",    type=int, default=20,
                        help="Number of samples to include in the PDF")
    parser.add_argument("--out",          default="./outputs/concepts.pdf")
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)
    n = min(args.n_samples, len(ds))
    print(f"Dataset: {len(ds)} records. Rendering {n} pages.")

    # Load nuScenes once for image path resolution
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
        def get_image_path(sample_token: str) -> Path:
            sample  = nusc.get("sample", sample_token)
            cam_sd  = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            return Path(args.data_root) / cam_sd["filename"]
    except Exception as e:
        print(f"Warning: nuScenes load failed ({e}). Rendering without images.")
        def get_image_path(_: str) -> Path:
            return Path("/nonexistent")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for i in range(n):
            record = ds[i]
            image_path = get_image_path(record["sample_token"])
            fig = _render_page(record, image_path)
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{n} pages written")

    print(f"\nSaved {n}-page PDF → {out_path}")


if __name__ == "__main__":
    main()
