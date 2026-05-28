"""CLI entry point for the CB-VLAM-AV concept mining pipeline.

Iterates a nuScenes split, runs the four extractors per frame, and saves
the resulting per-frame concept records as a HuggingFace Dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import numpy as np
from tqdm import tqdm
from datasets import Dataset

from cb_vlam.concept_mining.extractors import (
    KinematicExtractor,
    AgentExtractor,
    InfrastructureExtractor,
    SceneContextExtractor,
)
from cb_vlam.concept_mining.schema import default_concept_dict, CONCEPT_KEYS
from cb_vlam.data.nuscenes_loader import NuScenesLoader


def _load_sample_token_filter(path: Path) -> Set[str]:
    """Load a sample-token allow-list from JSON.

    Accepts two formats:
      - ["token1", "token2", ...]  (bare list of sample_tokens)
      - [{"id": "token1", ...}, ...]  (Impromptu-VLA records, keyed by 'id')
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    if isinstance(data[0], str):
        return set(data)
    if isinstance(data[0], dict) and "id" in data[0]:
        return {rec["id"] for rec in data}
    raise ValueError(
        f"{path}: list elements must be strings or dicts with 'id' field"
    )


def mine_scene(loader: NuScenesLoader,
               scene_info: Dict[str, Any],
               kinematic: KinematicExtractor,
               agent: AgentExtractor,
               infra: InfrastructureExtractor,
               scene_ctx: SceneContextExtractor,
               passes: List[str],
               sample_token_filter: Optional[Set[str]] = None,
               vlm_workers: int = 1) -> List[Dict[str, Any]]:
    """Run concept extraction over all samples in one scene.

    Args:
        loader: NuScenesLoader instance.
        scene_info: Scene metadata from loader.iter_scenes().
        kinematic, agent, infra, scene_ctx: Extractor instances.
        passes: Which passes to run, e.g., ["a", "b"] to skip Pass C.

    Returns:
        List of per-frame concept records.
    """
    # Reset all extractors at scene start
    kinematic.reset()
    agent.reset()
    infra.reset()
    scene_ctx.reset()

    # Buffer all samples so Pass C can fire VLM calls concurrently before the
    # main per-frame loop (which must stay sequential for kinematic/agent deltas).
    all_samples = list(enumerate(loader.iter_samples(scene_info["scene_token"])))

    if "c" in passes:
        scene_ctx.set_scene_context(scene_info.get("location", ""))
        if vlm_workers > 1:
            indexed_images = [(idx, s["front_image"]) for idx, s in all_samples]
            scene_ctx.precompute_scene(indexed_images, max_workers=vlm_workers)

    records: List[Dict[str, Any]] = []
    prev_sample: Optional[Dict[str, Any]] = None

    for frame_index, sample in all_samples:
        # Note: prev_sample is still advanced even for skipped frames so that
        # kinematic/agent deltas remain correct when we DO emit a record.
        if (sample_token_filter is not None
                and sample["sample_token"] not in sample_token_filter):
            prev_sample = sample
            continue

        concepts = default_concept_dict()

        if "a" in passes:
            try:
                a = kinematic.extract(
                    ego_pose=sample["ego_pose"],
                    prev_ego_pose=prev_sample["ego_pose"] if prev_sample else None,
                )
                concepts.update(a)
            except NotImplementedError:
                pass  # extractor not implemented yet

        if "b" in passes:
            try:
                b_agent = agent.extract(
                    ego_pose=sample["ego_pose"],
                    annotations=sample["annotations"],
                    prev_annotations=prev_sample["annotations"] if prev_sample else None,
                    nusc_map=scene_info["nusc_map"],
                )
                concepts.update(b_agent)
            except NotImplementedError:
                pass

            try:
                b_infra = infra.extract(
                    ego_pose=sample["ego_pose"],
                    nusc_map=scene_info["nusc_map"],
                )
                concepts.update(b_infra)
            except NotImplementedError:
                pass

        if "c" in passes:
            try:
                c = scene_ctx.extract(
                    image=sample["front_image"],
                    frame_index=frame_index,
                )
                concepts.update(c)
            except NotImplementedError:
                pass

        records.append({
            "scene_token": scene_info["scene_token"],
            "sample_token": sample["sample_token"],
            "timestamp": int(sample["timestamp"]),
            "frame_index": int(frame_index),
            "concepts": concepts,
        })

        prev_sample = sample

    return records


def mine(data_root: Path,
         version: str,
         output_path: Path,
         passes: List[str],
         max_scenes: Optional[int] = None,
         vlm_backend: str = "openrouter",
         vlm_model: str = "google/gemini-2.5-flash",
         keyframe_stride: int = 1,
         sample_tokens_file: Optional[Path] = None,
         vlm_workers: int = 1,
         vlm_image_dim: int = 1024,
         vlm_verbose: bool = False,
         vlm_timeout: int = 180) -> None:
    """Main mining loop."""
    sample_token_filter: Optional[Set[str]] = None
    if sample_tokens_file is not None:
        sample_token_filter = _load_sample_token_filter(sample_tokens_file)
        print(f"Loaded sample-token filter: {len(sample_token_filter)} tokens "
              f"from {sample_tokens_file}")

    loader = NuScenesLoader(
        data_root=data_root,
        version=version,
        load_images="c" in passes,
        load_maps="b" in passes,
    )

    kinematic = KinematicExtractor()
    agent = AgentExtractor()
    infra = InfrastructureExtractor()
    scene_ctx = SceneContextExtractor(backend=vlm_backend, model_name=vlm_model,
                                       keyframe_stride=keyframe_stride,
                                       max_image_dim=vlm_image_dim,
                                       verbose=vlm_verbose,
                                       request_timeout=vlm_timeout)

    all_records: List[Dict[str, Any]] = []
    vlm_calls_total = 0

    scenes = list(loader.iter_scenes(max_scenes=max_scenes))
    print(f"Mining {len(scenes)} scenes  |  passes={passes}  |  "
          f"vlm={vlm_model if 'c' in passes else 'n/a'}  |  "
          f"keyframe_stride={keyframe_stride if 'c' in passes else 'n/a'}")

    for scene_idx, scene_info in enumerate(scenes):
        print(f"\n[{scene_idx+1}/{len(scenes)}] {scene_info['scene_name']}  "
              f"({scene_info['location']})", flush=True)

        scene_ctx._vlm_call_count = 0
        records = mine_scene(
            loader, scene_info, kinematic, agent, infra, scene_ctx, passes,
            sample_token_filter=sample_token_filter,
            vlm_workers=vlm_workers,
        )
        all_records.extend(records)

        vlm_calls = getattr(scene_ctx, "_vlm_call_count", 0)
        vlm_calls_total += vlm_calls
        print(f"  → {len(records)} frames  |  VLM calls: {vlm_calls}  "
              f"|  total so far: {vlm_calls_total}", flush=True)

    print(f"\nTotal records: {len(all_records)}  |  Total VLM calls: {vlm_calls_total}")

    if not all_records:
        print("No records produced. Exiting.")
        return

    ds = Dataset.from_list(all_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mine concepts from nuScenes")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version (v1.0-mini for dev, v1.0-trainval for full)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the concept dataset")
    parser.add_argument("--passes", type=str, default="ab",
                        help="Which passes to run, e.g., 'a', 'ab', 'abc'")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit number of scenes (for development)")
    parser.add_argument("--vlm_backend", type=str, default="openrouter",
                        choices=["openrouter", "nrp"])
    parser.add_argument("--vlm_model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--keyframe_stride", type=int, default=1,
                        help="Run VLM every N frames; labels carried forward between calls")
    parser.add_argument("--vlm_workers", type=int, default=1,
                        help="Concurrent VLM calls per scene. Use 8 for qwen3-small, "
                             "16 for qwen3 (NRP fair-use limits).")
    parser.add_argument("--vlm_image_dim", type=int, default=1024,
                        help="Max dimension (longer side) for images sent to the VLM. "
                             "Smaller = fewer image tokens = faster inference. "
                             "Try 512 if VLM is the bottleneck.")
    parser.add_argument("--sample_tokens_file", type=str, default=None,
                        help="JSON file restricting mining to specific sample_tokens. "
                             "Accepts either a bare list of tokens or Impromptu-VLA "
                             "records (list of dicts with 'id' field).")
    parser.add_argument("--vlm_verbose", action="store_true",
                        help="Print raw VLM response and CoT trace (reasoning_content) "
                             "for every call. Useful for diagnosing concept misfires.")
    parser.add_argument("--vlm_timeout", type=int, default=180,
                        help="Per-call HTTP timeout in seconds. Bump to 300-600 for "
                             "qwen3 (full) which is slower than qwen3-small.")
    args = parser.parse_args()

    passes = list(args.passes.lower())
    valid = {"a", "b", "c"}
    if not set(passes).issubset(valid):
        raise ValueError(f"Invalid passes: {args.passes}. Must be subset of 'abc'.")

    mine(
        data_root=Path(args.data_root),
        version=args.version,
        output_path=Path(args.output_path),
        passes=passes,
        max_scenes=args.max_scenes,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        keyframe_stride=args.keyframe_stride,
        sample_tokens_file=Path(args.sample_tokens_file) if args.sample_tokens_file else None,
        vlm_workers=args.vlm_workers,
        vlm_image_dim=args.vlm_image_dim,
        vlm_verbose=args.vlm_verbose,
        vlm_timeout=args.vlm_timeout,
    )


if __name__ == "__main__":
    main()
