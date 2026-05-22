"""CLI: extract backbone features for nuScenes samples.

Reads Impromptu-VLA's nuscenes_{train,test}.json, runs each record through
the loaded backbone, and saves a HuggingFace Dataset of feature vectors
keyed by sample_token.

Usage:
    python -m cb_vlam.backbones.extract \
        --checkpoint /sc-rwx-vol/cbvlam/checkpoints/7B_AD_finetune \
        --nuscenes_root /sc-rwx-vol/cbvlam \
        --impromptu_repo /sc-rwx-vol/cbvlam/Impromptu-VLA \
        --output_dir /sc-rwx-vol/cbvlam/outputs/impromptu_features
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from PIL import Image
from tqdm import tqdm

from cb_vlam.backbones.impromptu_vla import ImpromptuVLABackbone


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path) as f:
        return json.load(f)


def process_split(
    backbone: ImpromptuVLABackbone,
    records: List[Dict[str, Any]],
    nuscenes_root: Path,
    split_name: str,
    output_dir: Path,
    flush_every: int = 1000,
) -> None:
    """Run the backbone over every record of one split, save as HF Dataset.

    Checkpoints by writing a partial HF dataset every `flush_every` records
    so that a job restart can resume cleanly (sentinel files mark each
    completed shard).
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    done_sentinel = split_dir / ".all_done"
    if done_sentinel.exists():
        print(f"=== {split_name}: complete sentinel present, skipping ===")
        return

    n = len(records)
    rows: List[Dict[str, Any]] = []
    shard_idx = 0

    # Skip shards that are already on disk
    while (split_dir / f"shard_{shard_idx:04d}").exists():
        shard_idx += 1
    start_record = shard_idx * flush_every
    if start_record >= n:
        print(f"=== {split_name}: all shards already written ===")
        done_sentinel.touch()
        return
    if start_record > 0:
        print(f"=== {split_name}: resuming from record {start_record} (shard {shard_idx}) ===")

    failures = 0
    pbar = tqdm(range(start_record, n), desc=f"extract[{split_name}]")
    for i in pbar:
        rec = records[i]
        sample_token = rec["id"]
        image_rel = rec["images"][0]
        image_path = nuscenes_root / image_rel
        user_prompt = rec["messages"][0]["content"]

        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                features = backbone.extract(image, user_prompt)
        except Exception as e:
            failures += 1
            tqdm.write(f"  [fail] {sample_token}: {type(e).__name__}: {e}")
            continue

        row = {
            "sample_token": sample_token,
            "split": split_name,
            "image_path": image_rel,
            **{k: v.tolist() for k, v in features.items()},
        }
        rows.append(row)

        # Flush a shard every flush_every records
        if len(rows) >= flush_every:
            shard_path = split_dir / f"shard_{shard_idx:04d}"
            Dataset.from_list(rows).save_to_disk(str(shard_path))
            tqdm.write(f"  saved {shard_path.name} ({len(rows)} rows)")
            rows = []
            shard_idx += 1

    # Flush leftover rows as the final shard
    if rows:
        shard_path = split_dir / f"shard_{shard_idx:04d}"
        Dataset.from_list(rows).save_to_disk(str(shard_path))
        tqdm.write(f"  saved {shard_path.name} ({len(rows)} rows)")

    done_sentinel.touch()
    print(f"=== {split_name}: done. failures: {failures}/{n} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Impromptu-VLA checkpoint directory")
    parser.add_argument("--nuscenes_root", required=True,
                        help="Root where the 'nuscenes/' directory lives (images resolve relative to here)")
    parser.add_argument("--impromptu_repo", required=True,
                        help="Path to cloned Impromptu-VLA repo (for nuscenes_train.json / nuscenes_test.json)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--flush_every", type=int, default=1000,
                        help="Save a HF Dataset shard every N records")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                        help="For dev/testing: cap the records per split")
    parser.add_argument("--splits", default="train,test",
                        help="Comma-separated list of splits to process")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading backbone from {args.checkpoint} ...")
    backbone = ImpromptuVLABackbone(args.checkpoint, dtype=args.dtype)
    backbone.load(device=args.device)
    print("Backbone loaded.")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    json_for = {"train": "nuscenes_train.json", "test": "nuscenes_test.json"}

    for split_name in splits:
        json_path = Path(args.impromptu_repo) / json_for[split_name]
        records = load_records(json_path)
        if args.max_samples_per_split is not None:
            records = records[: args.max_samples_per_split]
        print(f"\n=== {split_name}: {len(records)} records ===")
        process_split(
            backbone=backbone,
            records=records,
            nuscenes_root=Path(args.nuscenes_root),
            split_name=split_name,
            output_dir=output_dir,
            flush_every=args.flush_every,
        )

    print("\nAll splits done.")


if __name__ == "__main__":
    main()
