"""Training dataset: joins Impromptu model inputs with ConceptStore targets.

Each item is one nuScenes sample: the Impromptu front-camera image + driving
prompt (the backbone input) paired with the mined concept targets + masks (the
supervision), keyed by ``sample_token``. Items are restricted to one manifest
split (train/val/test), so there is no temporal leakage across splits.

The backbone consumes one (image, prompt) at a time (Qwen2.5-VL image grids vary
per sample), so the collate keeps images/prompts as Python lists and only stacks
the fixed-shape concept targets into batched tensors.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cb_vlam.data.concept_store import ConceptStore

_TARGET_KEYS = ["continuous", "continuous_mask", "binary", "binary_mask",
                "categorical", "categorical_mask"]

# Matches a numeric [x, y] pair only — the literal "[x, y]" template in the
# Impromptu PLANNING preamble has non-numeric content and is skipped, so this
# extracts exactly the predicted waypoints (verified: all train records → 6).
_WAYPOINT_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_trajectory(assistant_content: str, horizon: int = 6
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the ground-truth waypoints from an Impromptu assistant message.

    Returns (flat (horizon*2,) float32 waypoints in meters, (horizon*2,) bool
    mask). Pairs beyond `horizon` are dropped; if fewer are found, the missing
    tail is zero-padded and masked out (mask False). A total parse failure
    yields an all-zero target with an all-False mask, so the sample contributes
    nothing to L_t rather than corrupting it.
    """
    pairs = _WAYPOINT_RE.findall(assistant_content)[:horizon]
    traj = np.zeros(horizon * 2, dtype=np.float32)
    mask = np.zeros(horizon * 2, dtype=bool)
    for i, (x, y) in enumerate(pairs):
        traj[2 * i] = float(x)
        traj[2 * i + 1] = float(y)
        mask[2 * i] = mask[2 * i + 1] = True
    return traj, mask


class ConceptDataset(Dataset):
    def __init__(self,
                 store: ConceptStore,
                 split: str,
                 impromptu_jsons: Sequence[Path],
                 nuscenes_root: Path,
                 load_image: bool = True,
                 with_trajectory: bool = False,
                 horizon: int = 6,
                 max_samples: int = None):
        """
        Args:
            store: loaded ConceptStore (provides split tokens + targets).
            split: "train" | "val" | "test" (manifest split).
            impromptu_jsons: Impromptu JSON files supplying image+prompt per
                sample_token. Pass both nuscenes_train.json and nuscenes_test.json
                — train tokens live in the former, val/test tokens in the latter.
            nuscenes_root: root that image paths in the records resolve against.
            load_image: decode the PIL image in __getitem__ (True for training).
            with_trajectory: also parse the GT waypoints from the assistant
                message (generation task loss L_t). Off by default so
                concept-projection training is unaffected.
            horizon: number of future waypoints (Impromptu = 6, i.e. 3s @ 0.5s).
            max_samples: optional cap (for quick runs).
        """
        self.store = store
        self.nuscenes_root = Path(nuscenes_root)
        self.load_image = load_image
        self.with_trajectory = with_trajectory
        self.horizon = horizon

        # token -> Impromptu record (first occurrence wins; the JSONs repeat a
        # sample under different prompt windows — we dedupe to one input).
        rec_by_token: Dict[str, Dict[str, Any]] = {}
        for jp in impromptu_jsons:
            with open(jp) as f:
                for rec in json.load(f):
                    rec_by_token.setdefault(rec["id"], rec)
        self.rec_by_token = rec_by_token

        # Split tokens that have both an input record and a store row.
        self.tokens: List[str] = [
            t for t in store.split(split) if t in rec_by_token and t in store
        ]
        if max_samples is not None:
            self.tokens = self.tokens[:max_samples]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        tok = self.tokens[i]
        rec = self.rec_by_token[tok]
        item: Dict[str, Any] = {
            "sample_token": tok,
            "prompt": rec["messages"][0]["content"],
            "image_path": str(self.nuscenes_root / rec["images"][0]),
            "target": self.store.get_numpy(tok),  # numpy arrays
        }
        if self.with_trajectory:
            traj, mask = parse_trajectory(rec["messages"][1]["content"], self.horizon)
            item["trajectory"] = traj
            item["trajectory_mask"] = mask
        if self.load_image:
            with Image.open(item["image_path"]) as im:
                item["image"] = im.convert("RGB")
        return item


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """List images/prompts/tokens; stack the per-type concept targets."""
    out: Dict[str, Any] = {
        "images": [b.get("image") for b in batch],
        "prompts": [b["prompt"] for b in batch],
        "sample_tokens": [b["sample_token"] for b in batch],
    }
    targets = {
        k: torch.from_numpy(np.stack([b["target"][k] for b in batch], axis=0))
        for k in _TARGET_KEYS
    }
    out["targets"] = targets
    if "trajectory" in batch[0]:
        out["trajectory"] = torch.from_numpy(
            np.stack([b["trajectory"] for b in batch], axis=0))
        out["trajectory_mask"] = torch.from_numpy(
            np.stack([b["trajectory_mask"] for b in batch], axis=0))
    return out
