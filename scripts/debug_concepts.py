"""Diagnostic trace for deterministic Pass-B concepts.

Loads all sample tokens from a mined HuggingFace dataset (--dataset_path) and
prints the raw annotations that drive each concept, along with their ego-frame
geometry — so you can tell at a glance whether a "construction zone" trigger is
a traffic cone 5 m ahead in the front cam or a parked construction truck 25 m
behind on a cross street.

Usage (all samples in a dataset):
    python scripts/debug_concepts.py \
        --data_root /sc-rwx-vol/cbvlam/nuscenes \
        --version v1.0-trainval \
        --dataset_path /sc-rwx-vol/cbvlam/outputs/pass_ab_boston_test

Usage (specific tokens only, no dataset required):
    python scripts/debug_concepts.py \
        --data_root /sc-rwx-vol/cbvlam/nuscenes \
        --version v1.0-trainval \
        --tokens 25496f19ffd1,03ee880dd4e3,b2ee26cb848
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pyquaternion import Quaternion

CONSTRUCTION_PREFIXES = (
    "movable_object.trafficcone",
    "human.pedestrian.construction_worker",
    "vehicle.construction",
)

# nuScenes CAM_FRONT has a ~70° horizontal FOV (≈35° each side of centerline).
FRONT_CAM_HALF_FOV_DEG = 35.0


def _to_ego(global_xyz, ego_t, ego_R_inv):
    return ego_R_inv @ (np.array(global_xyz[:3]) - ego_t)


def _angle_from_forward_deg(x_fwd: float, y_lat: float) -> float:
    return float(np.degrees(np.arctan2(y_lat, x_fwd)))


def _fov_tag(x_fwd: float, angle: float) -> str:
    if x_fwd <= 0:
        return "BEHIND"
    if abs(angle) < FRONT_CAM_HALF_FOV_DEG:
        return "IN-FOV"
    return "OUT-FOV"


def _resolve_prefix(nusc, prefix: str):
    """Return (kind, full_token) for a 12-char prefix. kind in {'scene','sample'}."""
    for s in nusc.scene:
        if s["token"].startswith(prefix):
            return "scene", s["token"]
    for s in nusc.sample:
        if s["token"].startswith(prefix):
            return "sample", s["token"]
    raise ValueError(f"No scene or sample matches prefix {prefix!r}")


def _middle_sample_of_scene(nusc, scene_token: str) -> str:
    """Walk the linked list of samples and return the middle one."""
    scene = nusc.get("scene", scene_token)
    tokens = []
    t = scene["first_sample_token"]
    while t:
        tokens.append(t)
        t = nusc.get("sample", t)["next"]
    return tokens[len(tokens) // 2]


def debug_sample(nusc, sample_token: str) -> None:
    sample = nusc.get("sample", sample_token)
    scene  = nusc.get("scene", sample["scene_token"])
    log    = nusc.get("log",   scene["log_token"])

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose",    lidar_sd["ego_pose_token"])
    ego_t    = np.array(ego_pose["translation"][:3], dtype=np.float64)
    ego_R_inv = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix

    print(f"\n{'='*78}")
    print(f"sample {sample_token}")
    print(f"scene  {scene['token']}  ({scene['name']})  [{log['location']}]")
    print(f"{'='*78}")

    construction = []
    pedestrians  = []
    vehicles_moving = []

    for t in sample["anns"]:
        ann = nusc.get("sample_annotation", t)
        attrs = [nusc.get("attribute", at)["name"]
                 for at in ann.get("attribute_tokens", [])]
        pos = _to_ego(ann["translation"], ego_t, ego_R_inv)
        x_fwd, y_lat = float(pos[0]), float(pos[1])
        dist = float(np.linalg.norm(pos[:2]))
        angle = _angle_from_forward_deg(x_fwd, y_lat)
        cat = ann["category_name"]
        rec = dict(cat=cat, attrs=attrs, dist=dist,
                   x_fwd=x_fwd, y_lat=y_lat, angle=angle)

        if cat.startswith(CONSTRUCTION_PREFIXES) and dist < 30.0:
            construction.append(rec)

        if (cat.startswith("human.pedestrian")
                and not cat.startswith("human.pedestrian.construction_worker")
                and x_fwd > 0 and dist < 30.0):
            pedestrians.append(rec)

        if (cat.startswith("vehicle.")
                and not cat.startswith("vehicle.bicycle")
                and not cat.startswith("vehicle.motorcycle")
                and dist < 30.0
                and "vehicle.parked" not in attrs):
            vehicles_moving.append(rec)

    def _row(r):
        return (f"  {r['cat']:48s}  d={r['dist']:5.1f}m  "
                f"x_fwd={r['x_fwd']:+6.1f}  y_lat={r['y_lat']:+6.1f}  "
                f"ang={r['angle']:+6.1f}°  [{_fov_tag(r['x_fwd'], r['angle'])}]")

    print(f"\n[construction_zone_det] would fire = {bool(construction)}  "
          f"({len(construction)} annotation(s) in 30 m 360° band)")
    fov_hits = sum(1 for r in construction if r["x_fwd"] > 0
                   and abs(r["angle"]) < FRONT_CAM_HALF_FOV_DEG)
    if construction:
        print(f"  → in front-cam FOV: {fov_hits}/{len(construction)}")
    for r in sorted(construction, key=lambda r: r["dist"]):
        print(_row(r))
        if r["attrs"]:
            print(f"    attrs={r['attrs']}")

    print(f"\n[pedestrian_ahead / pedestrian_density / nearest_pedestrian_distance]")
    print(f"  forward pedestrians within 30 m: {len(pedestrians)}")
    for r in sorted(pedestrians, key=lambda r: r["dist"]):
        print(_row(r))
        if r["attrs"]:
            print(f"    attrs={r['attrs']}")

    print(f"\n[traffic_density_det]  moving (non-parked) vehicles within 30 m: "
          f"{len(vehicles_moving)}  (360°)")
    in_front  = sum(1 for r in vehicles_moving if r["x_fwd"] > 0)
    in_fov    = sum(1 for r in vehicles_moving if r["x_fwd"] > 0
                    and abs(r["angle"]) < FRONT_CAM_HALF_FOV_DEG)
    print(f"  → ahead of ego (x_fwd>0): {in_front}   in front-cam FOV: {in_fov}")
    for r in sorted(vehicles_moving, key=lambda r: r["dist"]):
        print(_row(r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_path",
                       help="Path to a HuggingFace dataset saved by mine_concepts.py. "
                            "All sample_tokens in the dataset will be traced.")
    group.add_argument("--tokens",
                       help="Comma-separated 12-char prefixes (scene or sample).")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    if args.dataset_path:
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_path)
        sample_tokens = ds["sample_token"]
        print(f"Loaded {len(sample_tokens)} records from {args.dataset_path}")
        for sample_token in sample_tokens:
            debug_sample(nusc, sample_token)
    else:
        for prefix in (t.strip() for t in args.tokens.split(",") if t.strip()):
            try:
                kind, full = _resolve_prefix(nusc, prefix)
            except ValueError as e:
                print(f"\n# SKIP {prefix!r}: {e}")
                continue
            if kind == "scene":
                sample_token = _middle_sample_of_scene(nusc, full)
                print(f"\n# prefix {prefix!r} → scene {full}, middle sample")
            else:
                sample_token = full
                print(f"\n# prefix {prefix!r} → sample {full}")
            debug_sample(nusc, sample_token)


if __name__ == "__main__":
    main()
