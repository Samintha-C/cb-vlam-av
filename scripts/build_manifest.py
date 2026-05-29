"""Build the supervision manifest for a mined concept dataset.

Consumes the HF Dataset produced by mine_concepts.py plus the Impromptu split
membership (from build_sample_token_filter.py), and emits a single manifest.json
that the training pipeline reads to interpret the dataset without re-reading
schema.py at runtime. The manifest captures:

  - schema fingerprint (detects train/data mismatch)
  - canonical concept order + per-type (continuous/binary/categorical) layout
  - train/val/test splits, partitioned at the SCENE level (no scene straddles
    two splits → no temporal leakage). train = Impromptu train scenes;
    val/test = a deterministic seeded carve of Impromptu test scenes.
  - per-concept stats computed on the TRAIN split only (for loss weighting /
    normalization): continuous mean/std, binary pos_frac, categorical counts.
  - mask_rules: which "saturated = absent" concepts may be masked out of the
    concept loss, each gated by a training-time toggle flag (default: supervise).

Usage:
    python scripts/build_manifest.py \
        --dataset_path   outputs/pass_ab_full \
        --membership     outputs/impromptu_split_membership.json \
        --passes         AB \
        --val_frac       0.25 \
        --seed           1234
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk

from cb_vlam.concept_mining.schema import get_all_concepts


# ── Mask rules ────────────────────────────────────────────────────────────────
# "Saturated = absent" concepts: the extractor encodes "no such object" with a
# saturated sentinel (e.g. lead_vehicle_distance = 1.0 when there is no lead).
# By default we SUPERVISE these (the saturated value is itself a learnable
# signal — the user's call), but each can be masked out of L_c via a toggle so
# the experiment can be run both ways without re-mining.
#
# `when` references a companion concept the trainer can read from the same
# concept vector. `equals`/`gte` give the masking predicate.
#
# ego_lateral_offset_in_lane is deliberately ABSENT: its sentinel (0.0, no lane
# found) is indistinguishable from a legitimately lane-centered ego, so it
# cannot be masked reliably and is always supervised.
MASK_RULES: Dict[str, Dict[str, Any]] = {
    "lead_vehicle_distance":          {"when": {"concept": "lead_vehicle_present", "equals": 0.0}, "toggle": "mask_lead_when_absent",      "default": "supervise"},
    "lead_vehicle_relative_velocity": {"when": {"concept": "lead_vehicle_present", "equals": 0.0}, "toggle": "mask_lead_when_absent",      "default": "supervise"},
    "time_to_collision_lead":         {"when": {"concept": "lead_vehicle_present", "equals": 0.0}, "toggle": "mask_lead_when_absent",      "default": "supervise"},
    "following_distance_seconds":     {"when": {"concept": "lead_vehicle_present", "equals": 0.0}, "toggle": "mask_lead_when_absent",      "default": "supervise"},
    "nearest_pedestrian_distance":    {"when": {"concept": "pedestrian_ahead",     "equals": 0.0}, "toggle": "mask_ped_when_absent",       "default": "supervise"},
    "nearest_crosswalk_distance":     {"when": {"concept": "nearest_crosswalk_distance", "gte": 1.0}, "toggle": "mask_crosswalk_when_absent", "default": "supervise"},
}


def _schema_hash(concepts: List[Dict]) -> str:
    """Stable sha256 over the concept definitions (order-sensitive)."""
    blob = json.dumps(concepts, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _per_type_layout(concepts: List[Dict]) -> Dict[str, Any]:
    """Group concept indices by type, in canonical order."""
    cont_idx, cont_names = [], []
    bin_idx, bin_names = [], []
    cat_idx, cat_names, cat_ncats = [], [], []
    for i, c in enumerate(concepts):
        t = c["type"]
        if t == "float":
            cont_idx.append(i); cont_names.append(c["name"])
        elif t == "binary":
            bin_idx.append(i); bin_names.append(c["name"])
        elif t == "categorical":
            cat_idx.append(i); cat_names.append(c["name"]); cat_ncats.append(len(c["values"]))
        else:
            raise ValueError(f"Unknown concept type {t!r} for {c['name']!r}")
    return {
        "continuous":  {"indices": cont_idx, "names": cont_names, "n": len(cont_idx)},
        "binary":      {"indices": bin_idx, "names": bin_names, "n": len(bin_idx)},
        "categorical": {"indices": cat_idx, "names": cat_names,
                        "n_categories": cat_ncats, "n": len(cat_idx)},
    }


def _assign_splits(ds, membership: Dict[str, List[str]],
                   val_frac: float, seed: int) -> Dict[str, Any]:
    """Partition the mined samples into train/val/test at the scene level.

    train       = scenes whose samples are in Impromptu train
    val + test  = scenes whose samples are in Impromptu test, carved val_frac/
                  (1-val_frac) by a seeded shuffle of the scene list.

    Only sample_tokens present in the mined dataset are emitted. Fails loudly
    if any scene straddles Impromptu train and test (would leak).
    """
    train_set = set(membership["train"])
    test_set = set(membership["test"])

    # Map each mined scene -> set of Impromptu split labels its samples carry.
    scene_labels: Dict[str, set] = defaultdict(set)
    scene_samples: Dict[str, List[str]] = defaultdict(list)
    unknown = 0
    for row in ds:
        st, samp = row["scene_token"], row["sample_token"]
        scene_samples[st].append(samp)
        if samp in train_set:
            scene_labels[st].add("train")
        elif samp in test_set:
            scene_labels[st].add("test")
        else:
            unknown += 1

    straddle = [s for s, labs in scene_labels.items() if labs == {"train", "test"}]
    if straddle:
        raise SystemExit(
            f"FATAL: {len(straddle)} scene(s) have samples in BOTH Impromptu "
            f"train and test. Scene-level split would leak. Example: {straddle[:3]}"
        )

    train_scenes = sorted(s for s, labs in scene_labels.items() if labs == {"train"})
    test_pool_scenes = sorted(s for s, labs in scene_labels.items() if labs == {"test"})

    # Deterministic carve of the test pool into val / test.
    rng = random.Random(seed)
    shuffled = test_pool_scenes[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    val_scenes = sorted(shuffled[:n_val])
    test_scenes = sorted(shuffled[n_val:])

    def tokens_for(scenes: List[str]) -> List[str]:
        out = []
        for s in scenes:
            out.extend(scene_samples[s])
        return out

    splits = {
        "by": "scene_token",
        "source": "impromptu_train_as_train; impromptu_test seeded-carved into val/test",
        "seed": seed,
        "val_frac": val_frac,
        "scene_counts": {"train": len(train_scenes), "val": len(val_scenes), "test": len(test_scenes)},
        "train": tokens_for(train_scenes),
        "val":   tokens_for(val_scenes),
        "test":  tokens_for(test_scenes),
    }
    if unknown:
        print(f"  note: {unknown} mined samples not in either Impromptu split "
              f"(excluded from all splits)")
    return splits


def _compute_stats(ds, train_tokens: List[str], concepts: List[Dict],
                   layout: Dict[str, Any]) -> Dict[str, Any]:
    """Per-concept stats over the train split only."""
    train_set = set(train_tokens)
    cont_names = layout["continuous"]["names"]
    bin_names = layout["binary"]["names"]
    cat_names = layout["categorical"]["names"]
    cat_ncats = dict(zip(cat_names, layout["categorical"]["n_categories"]))

    cont_vals = {n: [] for n in cont_names}
    bin_pos = {n: 0 for n in bin_names}
    bin_tot = {n: 0 for n in bin_names}
    cat_counts = {n: [0] * cat_ncats[n] for n in cat_names}

    for row in ds:
        if row["sample_token"] not in train_set:
            continue
        cd = row["concepts"]
        for n in cont_names:
            cont_vals[n].append(float(cd[n]))
        for n in bin_names:
            bin_pos[n] += 1 if float(cd[n]) >= 0.5 else 0
            bin_tot[n] += 1
        for n in cat_names:
            idx = int(round(float(cd[n])))
            if 0 <= idx < cat_ncats[n]:
                cat_counts[n][idx] += 1

    import statistics
    stats: Dict[str, Any] = {}
    for n in cont_names:
        vals = cont_vals[n]
        mean = statistics.fmean(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        stats[n] = {"mean": mean, "std": std}
    for n in bin_names:
        frac = (bin_pos[n] / bin_tot[n]) if bin_tot[n] else 0.0
        stats[n] = {"pos_frac": frac}
    for n in cat_names:
        stats[n] = {"class_counts": cat_counts[n]}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", required=True, type=Path,
                        help="Mining output dir containing data/ (HF Dataset). "
                             "manifest.json is written alongside it.")
    parser.add_argument("--membership", required=True, type=Path,
                        help="impromptu_split_membership.json from build_sample_token_filter.py")
    parser.add_argument("--passes", default="AB",
                        help="Which passes this dataset was mined with, e.g. 'AB'.")
    parser.add_argument("--val_frac", type=float, default=0.25,
                        help="Fraction of Impromptu-test scenes carved into val.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_subdir", default="data",
                        help="Subdirectory under dataset_path holding the HF Dataset.")
    args = parser.parse_args()

    passes = tuple(p for p in args.passes.upper() if p in ("A", "B", "C"))
    concepts = get_all_concepts(passes=passes)

    data_dir = args.dataset_path / args.data_subdir
    print(f"Loading mined dataset from {data_dir}")
    ds = load_from_disk(str(data_dir))
    print(f"  {len(ds)} mined records")

    with open(args.membership) as f:
        membership = json.load(f)

    layout = _per_type_layout(concepts)
    print(f"  layout: {layout['continuous']['n']} continuous, "
          f"{layout['binary']['n']} binary, {layout['categorical']['n']} categorical")

    print("Assigning scene-level splits…")
    splits = _assign_splits(ds, membership, args.val_frac, args.seed)
    print(f"  scenes: {splits['scene_counts']}")
    print(f"  samples: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")

    print("Computing train-split stats…")
    stats = _compute_stats(ds, splits["train"], concepts, layout)

    # Mask rules restricted to concepts actually in this pass selection.
    concept_names = {c["name"] for c in concepts}
    mask_rules = {k: v for k, v in MASK_RULES.items() if k in concept_names}

    manifest = {
        "schema_version": f"v1.{''.join(passes).lower()}",
        "schema_hash": _schema_hash(concepts),
        "passes": list(passes),
        "concept_order": [c["name"] for c in concepts],
        "per_type": layout,
        "stats": stats,
        "mask_rules": mask_rules,
        "splits": splits,
    }

    out_path = args.dataset_path / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
