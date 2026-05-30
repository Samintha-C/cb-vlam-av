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
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cb_vlam.data.concept_store import ConceptStore

_TARGET_KEYS = ["continuous", "continuous_mask", "binary", "binary_mask",
                "categorical", "categorical_mask"]


class ConceptDataset(Dataset):
    def __init__(self,
                 store: ConceptStore,
                 split: str,
                 impromptu_jsons: Sequence[Path],
                 nuscenes_root: Path,
                 load_image: bool = True,
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
            max_samples: optional cap (for quick runs).
        """
        self.store = store
        self.nuscenes_root = Path(nuscenes_root)
        self.load_image = load_image

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
    return out
