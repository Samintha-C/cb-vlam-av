"""Build the sample-token allow-list for mining, from Impromptu VLA splits.

Impromptu VLA ships its nuScenes training data as two JSON files
(nuscenes_train.json, nuscenes_test.json), each a list of records keyed by
`id` — a 32-char nuScenes sample_token. We mine concepts only for the samples
Impromptu actually uses, so this script extracts the union of unique tokens
and writes:

  1. A bare token list (--out_tokens) consumed by
     mine_concepts.py --sample_tokens_file. This is the mining allow-list.

  2. A split-membership sidecar (--out_membership) recording which tokens came
     from Impromptu's train vs test file. build_manifest.py consumes this to
     assign the final train / val / test splits (val is carved from test).

The Impromptu JSONs contain duplicate ids (the same sample under different
ego-history prompt windows); we dedupe to unique sample_tokens here.

Usage:
    python scripts/build_sample_token_filter.py \
        --impromptu_train /path/to/nuscenes_train.json \
        --impromptu_test  /path/to/nuscenes_test.json \
        --out_tokens      outputs/impromptu_sample_tokens.json \
        --out_membership  outputs/impromptu_split_membership.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Set


def _load_unique_ids(path: Path) -> List[str]:
    """Return the de-duplicated, order-preserving list of `id` fields."""
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of records")
    seen: Set[str] = set()
    ordered: List[str] = []
    for rec in records:
        tok = rec["id"]
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impromptu_train", required=True, type=Path)
    parser.add_argument("--impromptu_test", required=True, type=Path)
    parser.add_argument("--out_tokens", required=True, type=Path,
                        help="Bare JSON list of unique sample_tokens (mining allow-list).")
    parser.add_argument("--out_membership", required=True, type=Path,
                        help="JSON {train: [...], test: [...]} of Impromptu split membership.")
    args = parser.parse_args()

    train_ids = _load_unique_ids(args.impromptu_train)
    test_ids = _load_unique_ids(args.impromptu_test)

    train_set, test_set = set(train_ids), set(test_ids)
    overlap = train_set & test_set
    if overlap:
        # Should never happen with a clean nuScenes split; surface loudly.
        raise SystemExit(
            f"FATAL: {len(overlap)} sample_tokens appear in BOTH Impromptu "
            f"train and test. Example: {sorted(overlap)[:3]}. "
            f"Splitting on these would leak."
        )

    # Union, train-first, preserving order. No dupes across files (checked above).
    union = train_ids + test_ids

    args.out_tokens.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tokens, "w") as f:
        json.dump(union, f)

    args.out_membership.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_membership, "w") as f:
        json.dump({"train": train_ids, "test": test_ids}, f)

    print(f"Impromptu train: {len(train_ids)} unique sample_tokens")
    print(f"Impromptu test:  {len(test_ids)} unique sample_tokens")
    print(f"Union (mining allow-list): {len(union)} tokens")
    print(f"Wrote {args.out_tokens}")
    print(f"Wrote {args.out_membership}")


if __name__ == "__main__":
    main()
