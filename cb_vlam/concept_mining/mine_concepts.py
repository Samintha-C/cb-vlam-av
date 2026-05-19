"""CLI entry point for the CB-VLAM-AV concept mining pipeline.

Iterates a nuScenes split, runs the four extractors per frame, and saves
the resulting per-frame concept records as a HuggingFace Dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

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


def mine_scene(loader: NuScenesLoader,
               scene_info: Dict[str, Any],
               kinematic: KinematicExtractor,
               agent: AgentExtractor,
               infra: InfrastructureExtractor,
               scene_ctx: SceneContextExtractor,
               passes: List[str]) -> List[Dict[str, Any]]:
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

    records: List[Dict[str, Any]] = []
    prev_sample: Optional[Dict[str, Any]] = None

    for frame_index, sample in enumerate(loader.iter_samples(scene_info["scene_token"])):
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
         keyframe_stride: int = 1) -> None:
    """Main mining loop.

    Args:
        data_root: Path to nuScenes data root.
        version: nuScenes version (e.g., "v1.0-mini").
        output_path: Where to save the resulting HuggingFace Dataset.
        passes: Which passes to run.
        max_scenes: Optional scene limit for development.
        vlm_backend: "anthropic" or "local" (only used if Pass C is enabled).
        vlm_model: VLM model name.
    """
    loader = NuScenesLoader(
        data_root=data_root,
        version=version,
        load_images="c" in passes,
        load_maps=False,  # enable once InfrastructureExtractor is implemented
    )

    kinematic = KinematicExtractor()
    agent = AgentExtractor()
    infra = InfrastructureExtractor()
    scene_ctx = SceneContextExtractor(backend=vlm_backend, model_name=vlm_model,
                                       keyframe_stride=keyframe_stride)

    all_records: List[Dict[str, Any]] = []

    for scene_info in tqdm(loader.iter_scenes(max_scenes=max_scenes), desc="Scenes"):
        records = mine_scene(
            loader, scene_info, kinematic, agent, infra, scene_ctx, passes
        )
        all_records.extend(records)

    print(f"Total records: {len(all_records)}")

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
                        choices=["openrouter", "anthropic", "local"])
    parser.add_argument("--vlm_model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--keyframe_stride", type=int, default=1,
                        help="Run VLM every N frames; labels carried forward between calls")
    args = parser.parse_args()

    passes = list(args.passes.lower())
    valid = {"a", "b", "c"}
    if not set(passes).issubset(valid):
        raise ValueError(f"Invalid passes: {args.passes}. Must be subset of 'abc'.")

    mine(=== Mining passes A + B + C (50 scenes, stride=4) ===
usage: mine_concepts.py [-h] --data_root DATA_ROOT [--version VERSION]
                        --output_path OUTPUT_PATH [--passes PASSES]
                        [--max_scenes MAX_SCENES]
                        [--vlm_backend {openrouter,anthropic,local}]
                        [--vlm_model VLM_MODEL]
mine_concepts.py: error: unrecognized arguments: --keyframe_stride 4
        data_root=Path(args.data_root),
        version=args.version,
        output_path=Path(args.output_path),
        passes=passes,
        max_scenes=args.max_scenes,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        keyframe_stride=args.keyframe_stride,
    )


if __name__ == "__main__":
    main()
