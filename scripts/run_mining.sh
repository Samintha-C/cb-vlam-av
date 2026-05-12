#!/usr/bin/env bash
# Run concept mining on the nuScenes mini split.
# Adjust paths as needed.

set -e

DATA_ROOT="${NUSCENES_DATA_ROOT:-$HOME/data/nuscenes}"
OUTPUT_PATH="./outputs/mini_concepts"

python -m cb_vlam.concept_mining.mine_concepts \
    --data_root "$DATA_ROOT" \
    --version v1.0-mini \
    --output_path "$OUTPUT_PATH" \
    --passes ab \
    --max_scenes 2
