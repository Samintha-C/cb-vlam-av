"""Concept extractors for CB-VLAM-AV.

Each extractor reads a different slice of the nuScenes data and produces
concept values for one frame at a time. Extractors are stateful — they
track previous-frame info for computing deltas — and should have a
reset() method called between scenes.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from cb_vlam.concept_mining.schema import (
    PASS_A_CONCEPTS,
    PASS_B_AGENT_CONCEPTS,
    PASS_B_INFRA_CONCEPTS,
    PASS_C_SCENE_CONCEPTS,
)


class BaseExtractor:
    """Base class for all extractors. Subclasses must implement extract()."""

    def __init__(self):
        self.prev_state: Optional[Any] = None

    def reset(self) -> None:
        """Clear per-scene state. Call between scenes."""
        self.prev_state = None

    def extract(self, *args, **kwargs) -> Dict[str, float]:
        """Extract concepts for one frame.

        Returns:
            Dict mapping concept names (matching schema.py) to values.
        """
        raise NotImplementedError


def _yaw_from_quaternion_wxyz(q) -> float:
    """Extract yaw (rotation about z) from a nuScenes wxyz quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class KinematicExtractor(BaseExtractor):
    """Pass A: extract kinematic concepts from ego pose telemetry.

    Reads ego state from nuScenes ego_pose annotations (translation, rotation,
    timestamp) and computes speed, acceleration, heading, yaw rate, and
    frame-to-frame deltas.

    No neural network, no inference. Pure numpy.
    """

    def __init__(self,
                 max_speed_mps: float = 30.0,
                 max_accel_mps2: float = 5.0,
                 max_yaw_rate_radps: float = np.pi / 4,
                 stopped_speed_mps: float = 0.5,
                 braking_accel_mps2: float = -1.0,
                 turning_yaw_rate_radps: float = 0.1,
                 lookback_frames: int = 3):
        super().__init__()
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_yaw_rate_radps = max_yaw_rate_radps
        self.stopped_speed_mps = stopped_speed_mps
        self.braking_accel_mps2 = braking_accel_mps2
        self.turning_yaw_rate_radps = turning_yaw_rate_radps
        self.lookback_frames = lookback_frames

    def extract(self,
                ego_pose: Dict[str, Any],
                prev_ego_pose: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract kinematic concepts from one frame's ego pose.

        Speed and yaw rate are planar finite differences between consecutive
        ego poses; acceleration and speed delta need the previous frame's
        derived speed, kept in self.prev_state.

        Args:
            ego_pose: nuScenes ego_pose record. Has 'translation' (xyz),
                'rotation' (wxyz quaternion), 'timestamp' (microseconds).
            prev_ego_pose: Previous frame's ego_pose, or None for first frame.

        Returns:
            Dict with keys matching PASS_A_CONCEPTS schema.
        """
        # First frame of a scene: nothing computable from a single pose.
        # Seed prev_state with the current yaw so the next frame's yaw rate
        # is well-defined, and return zeros.
        if prev_ego_pose is None:
            self.prev_state = {
                "speed_mps": 0.0,
                "yaw": _yaw_from_quaternion_wxyz(ego_pose["rotation"]),
                "speed_history": [0.0],
            }
            return {
                "ego_speed": 0.0,
                "ego_acceleration": 0.0,
                "ego_yaw_rate": 0.0,
                "ego_speed_delta": 0.0,
                "ego_stopped": 1.0,
                "ego_braking": 0.0,
                "ego_turning": 0.0,
                "lateral_acceleration": 0.0,
            }

        dt = (ego_pose["timestamp"] - prev_ego_pose["timestamp"]) / 1e6
        # Guard against pathological dt (would only happen on corrupt data).
        if dt <= 0:
            dt = 0.5

        cur_xy = np.asarray(ego_pose["translation"][:2], dtype=np.float64)
        prev_xy = np.asarray(prev_ego_pose["translation"][:2], dtype=np.float64)
        speed_mps = float(np.linalg.norm(cur_xy - prev_xy) / dt)

        cur_yaw = _yaw_from_quaternion_wxyz(ego_pose["rotation"])
        prev_yaw = _yaw_from_quaternion_wxyz(prev_ego_pose["rotation"])
        yaw_delta = _wrap_pi(cur_yaw - prev_yaw)
        yaw_rate_radps = float(yaw_delta / dt)

        prev_speed_mps = self.prev_state["speed_mps"] if self.prev_state else 0.0
        accel_mps2 = float((speed_mps - prev_speed_mps) / dt)

        # ego_speed_delta: speed change over the last lookback_frames frames
        # (~1.5 s at 2 Hz). Captures trend/sustained deceleration, distinct
        # from the instantaneous ego_acceleration signal.
        speed_history = self.prev_state.get("speed_history", [prev_speed_mps])
        oldest_speed = speed_history[0]
        speed_delta_k = float(
            _clip((speed_mps - oldest_speed) / self.max_speed_mps, -1.0, 1.0)
        )
        new_history = (speed_history + [speed_mps])[-self.lookback_frames:]

        # Normalize per schema bounds.
        speed_norm = _clip(speed_mps / self.max_speed_mps, 0.0, 1.0)

        lat_accel_mps2 = yaw_rate_radps * speed_mps
        out = {
            "ego_speed": speed_norm,
            "ego_acceleration": _clip(accel_mps2 / self.max_accel_mps2, -1.0, 1.0),
            "ego_yaw_rate": _clip(yaw_rate_radps / self.max_yaw_rate_radps, -1.0, 1.0),
            "ego_speed_delta": speed_delta_k,
            "ego_stopped": 1.0 if speed_mps < self.stopped_speed_mps else 0.0,
            "ego_braking": 1.0 if accel_mps2 < self.braking_accel_mps2 else 0.0,
            "ego_turning": 1.0 if abs(yaw_rate_radps) > self.turning_yaw_rate_radps else 0.0,
            "lateral_acceleration": _clip(lat_accel_mps2 / self.max_accel_mps2, -1.0, 1.0),
        }

        self.prev_state = {
            "speed_mps": speed_mps,
            "yaw": cur_yaw,
            "speed_history": new_history,
        }
        return out


class AgentExtractor(BaseExtractor):
    """Pass B (agents): extract concepts about surrounding traffic agents.

    Reads 3D bounding box annotations from nuScenes and computes:
    - Lead vehicle distance and relative velocity
    - Pedestrian and cyclist presence
    - Nearby vehicle counts
    - Lateral clearances

    All distances are computed in the ego vehicle frame.
    """

    def __init__(self,
                 lead_vehicle_max_distance_m: float = 50.0,
                 nearby_vehicle_range_m: float = 30.0,
                 pedestrian_check_range_m: float = 20.0,
                 lane_width_m: float = 3.5,
                 class_presence_radius_m: float = 30.0,
                 ped_intent_radius_m: float = 15.0,
                 ped_intent_crosswalk_radius_m: float = 5.0,
                 ttc_max_seconds: float = 10.0,
                 following_max_seconds: float = 5.0,
                 ped_density_max_count: float = 5.0,
                 lane_count_radius_m: float = 10.0):
        super().__init__()
        self.lead_vehicle_max_distance_m = lead_vehicle_max_distance_m
        self.nearby_vehicle_range_m = nearby_vehicle_range_m
        self.pedestrian_check_range_m = pedestrian_check_range_m
        self.lane_width_m = lane_width_m
        self.class_presence_radius_m = class_presence_radius_m
        self.ped_intent_radius_m = ped_intent_radius_m
        self.ped_intent_crosswalk_radius_m = ped_intent_crosswalk_radius_m
        self.ttc_max_seconds = ttc_max_seconds
        self.following_max_seconds = following_max_seconds
        self.ped_density_max_count = ped_density_max_count
        self.lane_count_radius_m = lane_count_radius_m

    def extract(self,
                ego_pose: Dict[str, Any],
                annotations: List[Dict[str, Any]],
                prev_annotations: Optional[List[Dict[str, Any]]] = None,
                nusc_map: Optional[Any] = None) -> Dict[str, float]:
        """Extract agent concepts for one frame.

        Args:
            ego_pose: Ego pose record.
            annotations: List of nuScenes sample_annotation records visible at this timestamp.
                Each has 'translation', 'size', 'rotation', 'category_name', 'instance_token',
                'attribute_names' (resolved by the loader from attribute_tokens).
            prev_annotations: Annotations from previous frame, for velocity computation.
            nusc_map: NuScenesMap handle. Required for pedestrian_intent_crossing_det
                (defaults to 0 if not provided).

        Returns:
            Dict with keys matching PASS_B_AGENT_CONCEPTS schema.
        """
        from pyquaternion import Quaternion

        ego_t   = np.array(ego_pose["translation"][:3], dtype=np.float64)
        ego_R_inv = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        cur_yaw = _yaw_from_quaternion_wxyz(ego_pose["rotation"])
        prev_yaw = (self.prev_state.get("ego_yaw", cur_yaw)
                    if self.prev_state else cur_yaw)
        # Mid-interval ego yaw for lead-vehicle projection (reduces turn error).
        mid_yaw = prev_yaw + _wrap_pi(cur_yaw - prev_yaw) * 0.5
        cos_m, sin_m = float(np.cos(mid_yaw)), float(np.sin(mid_yaw))

        def to_ego_frame(global_xyz):
            """(x_forward, y_left, z_up) in ego vehicle frame."""
            return ego_R_inv @ (np.array(global_xyz[:3]) - ego_t)

        # dt and ego speed from stored previous state
        dt = 0.5
        ego_speed_mps = 0.0
        if self.prev_state is not None:
            dt = (ego_pose["timestamp"] - self.prev_state["timestamp"]) / 1e6
            if dt <= 0:
                dt = 0.5
            ego_speed_mps = float(
                np.linalg.norm(ego_t[:2] - self.prev_state["ego_t"][:2]) / dt
            )

        # Build instance-keyed lookup for prev frame velocity computation
        prev_by_inst: Dict[str, Any] = {}
        if prev_annotations is not None:
            prev_by_inst = {a["instance_token"]: a for a in prev_annotations}

        # Classify each annotation into a bucket
        vehicles:    List[Dict] = []
        pedestrians: List[Dict] = []
        cyclists:    List[Dict] = []

        for ann in annotations:
            cat = ann["category_name"]
            pos = to_ego_frame(ann["translation"])
            dist = float(np.linalg.norm(pos[:2]))
            attrs = ann.get("attribute_names", [])
            entry = {"ann": ann, "pos": pos, "dist": dist, "attrs": attrs}

            if cat.startswith("vehicle.bicycle") or cat.startswith("vehicle.motorcycle"):
                # Only count as a cyclist if someone is actually riding it.
                # cycle.without_rider is a parked bike/moto on a stand — not a traffic hazard.
                if "cycle.with_rider" in attrs:
                    cyclists.append(entry)
            elif cat.startswith("vehicle."):
                vehicles.append(entry)
            elif cat.startswith("human.pedestrian"):
                # Construction workers are excluded from the pedestrian bucket:
                # they're already captured by construction_zone_det and including
                # them here would make pedestrian_ahead fire when passing a work site.
                if not cat.startswith("human.pedestrian.construction_worker"):
                    pedestrians.append(entry)

        # ── Lead vehicle ──────────────────────────────────────────────────────
        half_lane = self.lane_width_m / 2.0
        lead_candidates = [
            v for v in vehicles
            if 0 < v["pos"][0] < self.lead_vehicle_max_distance_m
            and abs(v["pos"][1]) < half_lane
        ]

        lead_vel_fwd = 0.0   # forward velocity of lead in ego frame (m/s)

        if lead_candidates:
            lead = min(lead_candidates, key=lambda v: v["pos"][0])
            inst  = lead["ann"]["instance_token"]

            # Lead longitudinal velocity from instance tracking.
            # Project through mid-interval ego heading to reduce the O(ω·dt)
            # rotation error that accumulates when projecting through the
            # current-frame rotation alone on turns.
            if inst in prev_by_inst:
                global_delta_2d = (
                    np.array(lead["ann"]["translation"][:2])
                    - np.array(prev_by_inst[inst]["translation"][:2])
                )
                lead_vel_fwd = float(
                    (cos_m * global_delta_2d[0] + sin_m * global_delta_2d[1]) / dt
                )

            # Closing rate = ego advancing faster than lead (positive → approaching)
            closing_rate = ego_speed_mps - lead_vel_fwd

            # Lead decelerating: compare lead vel to previous frame's stored lead vel
            prev_lead_vel = self.prev_state.get("lead_vel_fwd", 0.0) if self.prev_state else 0.0
            lead_accel = (lead_vel_fwd - prev_lead_vel) / dt

            lead_dist_m = float(lead["pos"][0])
            # Time-to-collision: only meaningful if actively closing.
            if closing_rate > 0.1:
                ttc_norm = float(np.clip(lead_dist_m / closing_rate / self.ttc_max_seconds, 0.0, 1.0))
            else:
                ttc_norm = 1.0
            # Following distance in seconds: only meaningful if ego moving forward.
            if ego_speed_mps > 0.5:
                follow_norm = float(np.clip(lead_dist_m / ego_speed_mps / self.following_max_seconds, 0.0, 1.0))
            else:
                follow_norm = 1.0

            out_lead = {
                "lead_vehicle_present":           1.0,
                "lead_vehicle_distance":          float(np.clip(lead_dist_m / self.lead_vehicle_max_distance_m, 0.0, 1.0)),
                "lead_vehicle_relative_velocity": float(np.clip(closing_rate / 10.0, -1.0, 1.0)),
                "lead_vehicle_decelerating":      1.0 if lead_accel < -1.0 else 0.0,
                "time_to_collision_lead":         ttc_norm,
                "following_distance_seconds":     follow_norm,
            }
        else:
            out_lead = {
                "lead_vehicle_present":           0.0,
                "lead_vehicle_distance":          1.0,
                "lead_vehicle_relative_velocity": 0.0,
                "lead_vehicle_decelerating":      0.0,
                "time_to_collision_lead":         1.0,
                "following_distance_seconds":     1.0,
            }

        # ── Pedestrians ───────────────────────────────────────────────────────
        # pedestrian_ahead: any pedestrian within 10 m ahead in the forward corridor (±4 m lateral).
        peds_crosswalk = [
            p for p in pedestrians
            if 0 < p["pos"][0] < 10.0 and abs(p["pos"][1]) < 4.0
        ]
        # Forward-only filter for "presence" semantics — matches what the front
        # camera sees. nuScenes annotations are 360° around ego; we exclude
        # anything behind so the concept stays consistent with a forward
        # visual check.
        peds_fwd = [p for p in pedestrians if p["pos"][0] > 0]
        nearest_ped_dist = min((p["dist"] for p in peds_fwd), default=None)

        out_ped = {
            "pedestrian_ahead": 1.0 if peds_crosswalk else 0.0,
            "nearest_pedestrian_distance":   float(np.clip(nearest_ped_dist / 30.0, 0.0, 1.0))
                                             if nearest_ped_dist is not None else 1.0,
        }

        # ── Nearby vehicles & cyclists ────────────────────────────────────────
        # Exclude parked vehicles: a permanently-parked car on the curb isn't
        # "traffic". (vehicle.stopped — e.g., a car at a red light — still counts.)
        nearby_count = sum(
            1 for v in vehicles
            if v["dist"] < self.nearby_vehicle_range_m
            and "vehicle.parked" not in v["attrs"]
        )
        cyc_close    = any(
            c["pos"][0] > 0 and c["dist"] < self.pedestrian_check_range_m
            for c in cyclists
        )

        # Per-vehicle planar speed via instance tracking — needed for moving/parked checks
        def _vehicle_speed(v) -> float:
            inst = v["ann"]["instance_token"]
            if inst not in prev_by_inst:
                return 0.0
            delta = (np.array(v["ann"]["translation"][:2])
                     - np.array(prev_by_inst[inst]["translation"][:2]))
            return float(np.linalg.norm(delta) / dt)

        # Adjacent lane occupancy: moving vehicle in the lane band within ±20m.
        # ±20m matches the schema and gives enough look-ahead/behind for lane
        # change decisions (a vehicle 15m behind is still relevant).
        L = self.lane_width_m
        left_blocked  = any(
            _vehicle_speed(v) > 0.5
            for v in vehicles
            if L * 0.5 < v["pos"][1] < L * 2.5 and abs(v["pos"][0]) < 20.0
        )
        right_blocked = any(
            _vehicle_speed(v) > 0.5
            for v in vehicles
            if -L * 2.5 < v["pos"][1] < -L * 0.5 and abs(v["pos"][0]) < 20.0
        )

        # Parked cars: vehicles with the explicit vehicle.parked attribute in
        # side bands within ±15m. Using the annotation attribute is more
        # reliable than speed-derived heuristics: speed==0 on frame 0 (no prev)
        # and vehicle.stopped (e.g., at a red light) would otherwise misfire.
        parked = any(
            "vehicle.parked" in v["attrs"]
            for v in vehicles
            if abs(v["pos"][1]) > L * 0.5 and abs(v["pos"][0]) < 15.0
        )

        # ── Deterministic class-presence concepts ────────────────────────────
        # All use a 30m radius (class_presence_radius_m). Iterate annotations
        # once per class rather than reusing the vehicle/ped/cyclist buckets,
        # which filtered out movable_object.* and animal.* categories.
        # Returns (forward_x, planar_distance) so callers can apply a
        # front-half-plane filter where the semantic calls for it.
        def _ann_ego_xd(a):
            pos = to_ego_frame(a["translation"])
            return float(pos[0]), float(np.linalg.norm(pos[:2]))

        R = self.class_presence_radius_m
        construction_prefixes = (
            "movable_object.barrier",
            "movable_object.trafficcone",
            "human.pedestrian.construction_worker",
            "vehicle.construction",
        )
        # Emergency vehicles: 360° — sirens/lights matter from any direction.
        emergency_present = any(
            a["category_name"].startswith("vehicle.emergency.") and _ann_ego_xd(a)[1] < R
            for a in annotations
        )
        # Construction zone: 360° — area-level signal; barriers behind also indicate a zone.
        construction_present = any(
            a["category_name"].startswith(construction_prefixes) and _ann_ego_xd(a)[1] < R
            for a in annotations
        )
        # Animal or debris on road: forward-only — hazards on the ego's path.
        animal_or_debris = False
        for a in annotations:
            if not (a["category_name"].startswith("animal")
                    or a["category_name"].startswith("movable_object.debris")):
                continue
            x_fwd, d = _ann_ego_xd(a)
            if x_fwd > 0 and d < R:
                animal_or_debris = True
                break

        # ── Pedestrian intent crossing ───────────────────────────────────────
        # A pedestrian with pedestrian.moving attribute, within 15m AHEAD of
        # ego AND within 5m of a ped_crossing polygon, signals likely
        # crossing intent into the ego's path.
        ped_intent = 0.0
        if nusc_map is not None:
            for ped in pedestrians:
                if ped["pos"][0] <= 0:  # behind ego
                    continue
                if ped["dist"] > self.ped_intent_radius_m:
                    continue
                if "pedestrian.moving" not in ped["ann"].get("attribute_names", []):
                    continue
                gx = float(ped["ann"]["translation"][0])
                gy = float(ped["ann"]["translation"][1])
                try:
                    crossings = nusc_map.get_records_in_radius(
                        gx, gy, self.ped_intent_crosswalk_radius_m, ["ped_crossing"]
                    )["ped_crossing"]
                except Exception:
                    crossings = []
                if crossings:
                    ped_intent = 1.0
                    break

        # ── Pedestrian density ───────────────────────────────────────────────
        # Forward-only: density of peds the ego is approaching, not all
        # pedestrians in a 360° ring (which over-counts crowds behind ego).
        ped_count_nearby = sum(
            1 for p in pedestrians
            if p["pos"][0] > 0 and p["dist"] < self.class_presence_radius_m
        )
        ped_density_norm = float(np.clip(
            ped_count_nearby / self.ped_density_max_count, 0.0, 1.0
        ))

        # ── Traffic density (per-lane, categorical encoded as float index) ────
        # Normalize moving-vehicle count by the number of lanes near ego so
        # the signal is comparable across a one-lane residential street and a
        # multi-lane arterial. lane_count uses the HD-map `lane` layer; a
        # single drivable lane intersects only itself within a small radius,
        # while a wide road intersects all parallel lanes.
        lane_count = 1
        if nusc_map is not None:
            try:
                from nuscenes.map_expansion import arcline_path_utils as _apu
                nearby_lanes = nusc_map.get_records_in_radius(
                    float(ego_t[0]), float(ego_t[1]),
                    self.lane_count_radius_m, ["lane"]
                )["lane"]
                # Keep only lanes whose heading aligns with ego forward.
                # On a divided road the 10 m radius catches both directions;
                # without this filter the count doubles and halves per-lane density.
                ego_yaw = _yaw_from_quaternion_wxyz(ego_pose["rotation"])
                ego_fwd = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
                same_dir = 0
                for lt in nearby_lanes:
                    arcline = nusc_map.arcline_path_3.get(lt)
                    if arcline is None:
                        same_dir += 1  # can't tell; include
                        continue
                    try:
                        poses = _apu.discretize_lane(arcline, resolution_meters=2.0)
                        if not poses:
                            same_dir += 1
                            continue
                        lane_yaw = poses[0][2]
                        lane_fwd = np.array([np.cos(lane_yaw), np.sin(lane_yaw)])
                        if float(lane_fwd @ ego_fwd) > 0:
                            same_dir += 1
                    except Exception:
                        same_dir += 1
                lane_count = max(1, same_dir)
            except Exception:
                lane_count = 1

        vehicles_per_lane = nearby_count / lane_count
        # 0=light (<1.0 v/lane), 1=moderate (<2.0), 2=heavy (>=2.0)
        if vehicles_per_lane >= 2.0:
            traffic_dens_idx = 2.0
        elif vehicles_per_lane >= 1.0:
            traffic_dens_idx = 1.0
        else:
            traffic_dens_idx = 0.0

        out_misc = {
            "vehicle_count_nearby":  float(np.clip(nearby_count / 10.0, 0.0, 1.0)),
            "cyclist_present":       1.0 if cyc_close else 0.0,
            "left_lane_blocked":     1.0 if left_blocked  else 0.0,
            "right_lane_blocked":    1.0 if right_blocked else 0.0,
            "parked_cars_present":   1.0 if parked        else 0.0,
            "emergency_vehicle_present_det":  1.0 if emergency_present else 0.0,
            "construction_zone_det":          1.0 if construction_present else 0.0,
            "animal_or_debris_on_road_det":   1.0 if animal_or_debris else 0.0,
            "pedestrian_intent_crossing_det": ped_intent,
            "pedestrian_density":             ped_density_norm,
            "traffic_density_det":            traffic_dens_idx,
        }

        # ── Update state ──────────────────────────────────────────────────────
        self.prev_state = {
            "timestamp":    ego_pose["timestamp"],
            "ego_t":        ego_t.copy(),
            "lead_vel_fwd": lead_vel_fwd,
            "ego_yaw":      cur_yaw,
        }

        return {**out_lead, **out_ped, **out_misc}


class InfrastructureExtractor(BaseExtractor):
    """Pass B (infrastructure): extract concepts from the HD map.

    Uses NuScenesMap to query:
    - Whether ego is inside an intersection polygon
    - Distance to the next intersection along the forward direction
    - Whether adjacent left/right lanes exist on the current road
    - Curvature of the lane centerline `lookahead_m` ahead

    `speed_limit_normalized` is always 0.0: nuScenes HD maps do not encode
    speed limits. The Pass C VLM produces `speed_limit_sign` separately and
    can supply a noisy version of this signal.
    """

    def __init__(self,
                 lookahead_m: float = 30.0,
                 max_intersection_distance_m: float = 100.0,
                 lane_search_radius_m: float = 2.0,
                 lane_width_m: float = 3.5,
                 max_curvature_per_m: float = 0.05,
                 max_crosswalk_distance_m: float = 30.0,
                 max_traffic_light_distance_m: float = 50.0,
                 max_lateral_offset_m: float = 2.0):
        super().__init__()
        self.lookahead_m = lookahead_m
        self.max_intersection_distance_m = max_intersection_distance_m
        self.lane_search_radius_m = lane_search_radius_m
        self.lane_width_m = lane_width_m
        self.max_curvature_per_m = max_curvature_per_m
        self.max_crosswalk_distance_m = max_crosswalk_distance_m
        self.max_traffic_light_distance_m = max_traffic_light_distance_m
        self.max_lateral_offset_m = max_lateral_offset_m

    @staticmethod
    def _menger_curvature_signed(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """Signed Menger curvature of 3 points. Positive = left turn."""
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if a * b * c < 1e-6:
            return 0.0
        # Signed twice-area via 2D cross product
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        return float(2.0 * cross / (a * b * c))

    def extract(self,
                ego_pose: Dict[str, Any],
                nusc_map: Any) -> Dict[str, float]:
        """Extract infrastructure concepts for one frame."""
        from nuscenes.map_expansion import arcline_path_utils

        x, y = float(ego_pose["translation"][0]), float(ego_pose["translation"][1])
        yaw = _yaw_from_quaternion_wxyz(ego_pose["rotation"])
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

        # ── in_intersection + point-in-polygon concepts ──────────────────────
        # layers_on_point returns a dict {layer_name: token or ""} for every
        # polygonal layer that the (x, y) point falls inside.
        in_intersection = 0.0
        layers = nusc_map.layers_on_point(x, y)
        rs_token = layers.get("road_segment", "")
        if rs_token:
            rs = nusc_map.get("road_segment", rs_token)
            if rs.get("is_intersection", False):
                in_intersection = 1.0
        over_stop_line = 1.0 if layers.get("stop_line") else 0.0
        on_walkway     = 1.0 if layers.get("walkway") else 0.0
        in_carpark     = 1.0 if layers.get("carpark_area") else 0.0

        # ── distance_to_intersection ──────────────────────────────────────────
        # Distance to the nearest boundary vertex of each forward intersection.
        # Using the nearest forward boundary vertex rather than the centroid
        # gives a better "distance to entering the intersection" estimate for
        # elongated or T-junction polygons where the centroid can be well past
        # the entrance line.
        nearby = nusc_map.get_records_in_radius(
            x, y, self.max_intersection_distance_m, ["road_segment"]
        )["road_segment"]
        ego_arr = np.array([x, y])
        min_fwd_dist = float("inf")
        for tok in nearby:
            rs = nusc_map.get("road_segment", tok)
            if not rs.get("is_intersection", False):
                continue
            try:
                poly = nusc_map.extract_polygon(rs["polygon_token"])
            except Exception:
                continue
            fwd_verts = [
                np.array([bx, by])
                for bx, by in poly.exterior.coords
                if float((np.array([bx, by]) - ego_arr) @ forward) > 0
            ]
            if not fwd_verts:
                continue
            d = float(min(np.linalg.norm(pt - ego_arr) for pt in fwd_verts))
            if d < min_fwd_dist:
                min_fwd_dist = d

        if in_intersection:
            dist_norm = 0.0
            approaching = 1.0
        elif min_fwd_dist == float("inf"):
            dist_norm = 1.0
            approaching = 0.0
        else:
            dist_norm = float(min(min_fwd_dist / self.max_intersection_distance_m, 1.0))
            approaching = 1.0 if min_fwd_dist < 30.0 else 0.0

        # ── nearest crosswalk distance ───────────────────────────────────────
        # ped_crossing polygons. Distance is to polygon centroid; 1.0 if none
        # within max_crosswalk_distance_m.
        nearby_xwalks = nusc_map.get_records_in_radius(
            x, y, self.max_crosswalk_distance_m, ["ped_crossing"]
        )["ped_crossing"]
        min_xwalk_dist = float("inf")
        for tok in nearby_xwalks:
            rec = nusc_map.get("ped_crossing", tok)
            try:
                poly = nusc_map.extract_polygon(rec["polygon_token"])
                cx, cy = poly.centroid.x, poly.centroid.y
            except Exception:
                continue
            d = float(np.linalg.norm([cx - x, cy - y]))
            if d < min_xwalk_dist:
                min_xwalk_dist = d
        if min_xwalk_dist == float("inf"):
            crosswalk_dist_norm = 1.0
        else:
            crosswalk_dist_norm = float(
                min(min_xwalk_dist / self.max_crosswalk_distance_m, 1.0)
            )

        # ── traffic_light_location_ahead ─────────────────────────────────────
        # Binary: any traffic_light whose polygon centroid is within
        # max_traffic_light_distance_m AND in the forward half-plane.
        # nuScenes traffic_light layer encodes location only, no signal state.
        try:
            nearby_tls = nusc_map.get_records_in_radius(
                x, y, self.max_traffic_light_distance_m, ["traffic_light"]
            )["traffic_light"]
            tl_location_ahead = 0.0
            for tl_tok in nearby_tls:
                tl_rec = nusc_map.get("traffic_light", tl_tok)
                try:
                    poly = nusc_map.extract_polygon(tl_rec["polygon_token"])
                    cx, cy = poly.centroid.x, poly.centroid.y
                except Exception:
                    continue
                rel = np.array([cx - x, cy - y])
                if float(rel @ forward) > 0:
                    tl_location_ahead = 1.0
                    break
        except Exception:
            # Some map locations may not include the traffic_light layer.
            tl_location_ahead = 0.0

        # ── lane availability + curvature ─────────────────────────────────────
        lane_left = 0.0
        lane_right = 0.0
        curvature_norm = 0.0
        lateral_offset_norm = 0.0

        current_lane_token = nusc_map.get_closest_lane(x, y, radius=self.lane_search_radius_m)
        if current_lane_token:
            arcline = nusc_map.arcline_path_3.get(current_lane_token)
            if arcline:
                poses = arcline_path_utils.discretize_lane(arcline, resolution_meters=1.0)
                if poses:
                    lane_xy = np.array([(p[0], p[1]) for p in poses])
                    d_to_lane = np.linalg.norm(lane_xy - np.array([x, y]), axis=1)
                    nearest_idx = int(np.argmin(d_to_lane))
                    target_idx = min(nearest_idx + int(self.lookahead_m),
                                     len(poses) - 1)
                    if target_idx - nearest_idx >= 2:
                        mid_idx = (nearest_idx + target_idx) // 2
                        curv = self._menger_curvature_signed(
                            lane_xy[nearest_idx],
                            lane_xy[mid_idx],
                            lane_xy[target_idx],
                        )
                        curvature_norm = float(np.clip(
                            curv / self.max_curvature_per_m, -1.0, 1.0))

                    lane_yaw = float(poses[nearest_idx][2])
                    # Perpendicular: left is +90° from heading
                    perp_left = np.array([-np.sin(lane_yaw), np.cos(lane_yaw)])
                    # Signed lateral offset (+ = ego left of centerline)
                    ego_minus_lane = np.array([x, y]) - lane_xy[nearest_idx]
                    lateral_offset_m = float(np.dot(ego_minus_lane, perp_left))
                    lateral_offset_norm = float(np.clip(
                        lateral_offset_m / self.max_lateral_offset_m, -1.0, 1.0
                    ))
                    probe_left = np.array([x, y]) + perp_left * self.lane_width_m
                    probe_right = np.array([x, y]) - perp_left * self.lane_width_m

                    left_tok = nusc_map.get_closest_lane(
                        float(probe_left[0]), float(probe_left[1]),
                        radius=self.lane_search_radius_m,
                    )
                    right_tok = nusc_map.get_closest_lane(
                        float(probe_right[0]), float(probe_right[1]),
                        radius=self.lane_search_radius_m,
                    )
                    if left_tok and left_tok != current_lane_token:
                        lane_left = 1.0
                    if right_tok and right_tok != current_lane_token:
                        lane_right = 1.0

        return {
            "approaching_intersection": float(approaching),
            "distance_to_intersection": dist_norm,
            "lane_available_left":      lane_left,
            "lane_available_right":     lane_right,
            # nuScenes maps have no speed-limit field; left at 0.0 (Pass C
            # `speed_limit_sign` covers this from the front-camera image).
            "speed_limit_normalized":   0.0,
            "road_curvature_ahead":     curvature_norm,
            "in_intersection":          in_intersection,
            "over_stop_line":               over_stop_line,
            "nearest_crosswalk_distance":   crosswalk_dist_norm,
            "on_walkway":                   on_walkway,
            "in_carpark":                   in_carpark,
            "traffic_light_location_ahead": tl_location_ahead,
            "ego_lateral_offset_in_lane":   lateral_offset_norm,
        }


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_NRP_URL = "https://ellm.nrp-nautilus.io/v1/chat/completions"

# NRP fair-use per-model concurrency caps (per user). Source:
# https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/
# Exceeding these is a fair-use violation even if not auto-enforced today.
_NRP_CONCURRENCY_CAPS = {
    "kimi": 2,
    "glm-5": 4,
    "minimax-m2": 8,
    "qwen3-small": 8,
    "gemma": 8,
    "gemma-small": 8,
    "qwen3": 16,
    "gpt-oss": 16,
    "qwen3-embedding": 16,
}

# Both backends are OpenAI-compatible — they differ only in endpoint URL,
# the env var that holds the bearer token, and (for OpenRouter) two attribution
# headers. Keeping the table here makes adding a third backend trivial.
_BACKEND_CONFIG = {
    "openrouter": {
        "url": _OPENROUTER_URL,
        "env_var": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/Samintha-C/cb-vlam-av",
            "X-Title": "cb-vlam-av",
        },
    },
    "nrp": {
        "url": _NRP_URL,
        "env_var": "NRP_API_KEY",
        "extra_headers": {},
    },
}


_MAJORITY_VOTE_CONCEPTS = frozenset({
    "emergency_vehicle_present",
    "accident_or_disabled_vehicle",
    "animal_or_debris_on_road",
    "construction_zone",
})


def _build_pass_c_prompt(location: str = "") -> str:
    """Build the structured Pass C VLM prompt with optional location context."""
    from cb_vlam.concept_mining.schema import PASS_C_SCENE_CONCEPTS

    categorical_lines = []
    binary_lines = []
    for c in PASS_C_SCENE_CONCEPTS:
        if c["type"] == "categorical":
            vals = "/".join(c["values"])
            categorical_lines.append(f'- "{c["name"]}" (one of: {vals}) — {c["desc"]}')
        else:
            binary_lines.append(f'- "{c["name"]}" (true/false) — {c["desc"]}')

    loc = location.lower()
    if "singapore" in loc:
        loc_line = (
            "Scene location: Singapore. Traffic flows on the LEFT. "
            "Speed limit signs show km/h — match the numeral on the sign to the closest "
            "schema value without unit conversion (e.g., a sign reading '40' → '40mph')."
        )
    elif "boston" in loc:
        loc_line = (
            "Scene location: Boston, USA. Traffic flows on the RIGHT. "
            "Speed limit signs show mph."
        )
    else:
        loc_line = ""

    parts = []
    if loc_line:
        parts.append(f"[{loc_line}]\n\n")
    parts.append(
        "You are analyzing a front-camera image from an autonomous vehicle. "
        "Extract the following concepts and return ONLY a valid JSON object — "
        "no markdown fences, no commentary.\n\n"
        "CATEGORICAL CONCEPTS (return the string value):\n"
    )
    parts.append("\n".join(categorical_lines))
    parts.append("\n\nBINARY CONCEPTS (return true or false):\n")
    parts.append("\n".join(binary_lines))
    parts.append(
        "\n\nGuidelines:\n"
        "- 'lead vehicle' = the vehicle directly ahead of ego in the same lane.\n"
        "- For categoricals with a 'none' option, return 'none' if not visible/applicable.\n"
        "- Be conservative: if uncertain, return the default (false / 'none' / 'clear').\n"
        "- Return every key from the lists above. Return ONLY the JSON object.\n\n"
        "PRECISION RULES — only set true if the criteria below are STRICTLY met:\n"
        '- "construction_zone": Permanent road dividers, lane barriers, concrete medians, '
        "and bollards are NOT construction zones. Only mark true for ACTIVE work: cones "
        "delimiting an excavation or work area, workers in hi-vis vests, construction "
        "machinery, or temporary work-zone signage.\n"
        '- "tunnel_or_bridge_ahead": Building overhangs, covered walkways, and pedestrian '
        "bridges over the sidewalk are NOT tunnels or bridges over the roadway. Only mark "
        "true if the roadway itself is about to enter a tunnel portal or cross a bridge span."
    )
    return "".join(parts)


class SceneContextExtractor(BaseExtractor):
    """Pass C: extract scene context with a VLM.

    Currently supports the OpenRouter backend, which proxies many vendor APIs
    (Gemini, Claude, GPT, etc.) behind a single OpenAI-compatible endpoint.

    Categorical concepts are returned as integer indices into the schema's
    `values` list. Binaries are returned as 0.0 or 1.0.
    """

    def __init__(self,
                 backend: str = "openrouter",
                 model_name: str = "google/gemini-2.5-flash",
                 keyframe_stride: int = 1,
                 max_image_dim: int = 1024,
                 jpeg_quality: int = 85,
                 request_timeout: int = 180,
                 verbose: bool = False):
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self.keyframe_stride = keyframe_stride
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality
        self.request_timeout = request_timeout
        self.verbose = verbose
        self._cached_labels: Optional[Dict[str, float]] = None
        self._prompt = _build_pass_c_prompt()

        # Build categorical lookup: name -> {value_string: index}
        from cb_vlam.concept_mining.schema import PASS_C_SCENE_CONCEPTS
        self._categorical_index: Dict[str, Dict[str, int]] = {}
        self._binary_names: set = set()
        for c in PASS_C_SCENE_CONCEPTS:
            if c["type"] == "categorical":
                self._categorical_index[c["name"]] = {v: i for i, v in enumerate(c["values"])}
            elif c["type"] == "binary":
                self._binary_names.add(c["name"])

    def reset(self) -> None:
        super().reset()
        self._cached_labels = None
        self._precomputed: Dict[int, Dict[str, float]] = {}

    def set_scene_context(self, location: str) -> None:
        """Rebuild the VLM prompt with location-specific context. Call once per scene."""
        self._prompt = _build_pass_c_prompt(location)

    def precompute_scene(
        self,
        indexed_images: List[tuple],
        max_workers: int = 1,
    ) -> None:
        """Fire VLM calls for all stride-sampled frames concurrently.

        Args:
            indexed_images: List of (frame_index, image_ndarray) for every
                frame in the scene, in scene order.
            max_workers: Number of concurrent HTTP requests.  Should not
                exceed the per-model concurrency limit on the gateway
                (8 for qwen3-small, 16 for qwen3).

        After this returns, `extract()` reads from `self._precomputed`
        instead of making synchronous calls.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Compliance: cap workers at the NRP fair-use limit for this model.
        if self.backend == "nrp":
            cap = _NRP_CONCURRENCY_CAPS.get(self.model_name)
            if cap is not None and max_workers > cap:
                print(f"  [pass-c] capping vlm_workers {max_workers} -> {cap} "
                      f"(NRP fair-use limit for {self.model_name})", flush=True)
                max_workers = cap

        self._precomputed = {}
        vlm_frames = [
            (idx, img) for idx, img in indexed_images
            if img is not None and idx % self.keyframe_stride == 0
        ]
        if not vlm_frames:
            return

        import time
        call_counter = [0]
        counter_lock = threading.Lock()
        results: Dict[int, Optional[Dict[str, float]]] = {}
        durations: List[float] = []
        batch_start = time.monotonic()

        def _call_one(frame_idx, image):
            t0 = time.monotonic()
            try:
                concepts = self._query_vlm_voted(image, label=f"f{frame_idx}")
                return frame_idx, concepts, time.monotonic() - t0
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  [pass-c] frame {frame_idx:3d}  ERROR: {err}", flush=True)
                return frame_idx, None, time.monotonic() - t0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(_call_one, idx, img): idx
                             for idx, img in vlm_frames}
            for future in as_completed(future_to_idx):
                frame_idx, concepts, dt = future.result()
                results[frame_idx] = concepts
                durations.append(dt)
                with counter_lock:
                    call_counter[0] += 1
                    n = call_counter[0]
                status = "ok" if concepts is not None else "FAILED"
                if concepts:
                    notable = {k: v for k, v in concepts.items()
                               if k in self._binary_names and v == 1.0}
                    notable.update({k: v for k, v in concepts.items()
                                    if k in self._categorical_index and v != 0.0})
                    suffix = (f"  [{', '.join(f'{k}={v:.0f}' for k, v in notable.items())}]"
                              if notable else "")
                else:
                    suffix = ""
                print(f"  [pass-c] frame {frame_idx:3d}  call #{n}/{len(vlm_frames)} "
                      f"({dt:.1f}s) ... {status}{suffix}", flush=True)

        wall = time.monotonic() - batch_start
        if durations:
            d = sorted(durations)
            med = d[len(d) // 2]
            print(f"  [pass-c] batch wall={wall:.1f}s  "
                  f"per-call min={d[0]:.1f}s median={med:.1f}s max={d[-1]:.1f}s  "
                  f"effective_concurrency={sum(durations)/wall:.2f}x",
                  flush=True)

        # Apply carry-forward in frame order to fill self._precomputed
        cached = None
        for idx, _ in sorted(indexed_images, key=lambda x: x[0]):
            if idx in results:
                if results[idx] is not None:
                    cached = results[idx]
                self._precomputed[idx] = cached.copy() if cached else self._defaults()
            else:
                self._precomputed[idx] = cached.copy() if cached else self._defaults()

        self._vlm_call_count = getattr(self, "_vlm_call_count", 0) + len(vlm_frames)

    def _defaults(self) -> Dict[str, float]:
        """All-zero / first-category default values for every Pass C concept."""
        from cb_vlam.concept_mining.schema import PASS_C_SCENE_CONCEPTS
        return {c["name"]: 0.0 for c in PASS_C_SCENE_CONCEPTS}

    def extract(self,
                image: np.ndarray,
                frame_index: int) -> Dict[str, float]:
        """Extract scene context concepts for one frame.

        Args:
            image: Front camera RGB image (H, W, 3) uint8, or None.
            frame_index: Frame index within the scene.

        Returns:
            Dict with keys matching PASS_C_SCENE_CONCEPTS schema.
        """
        # Fast path: precompute_scene already ran for this scene.
        if frame_index in self._precomputed:
            return self._precomputed[frame_index].copy()

        if image is None:
            return self._cached_labels.copy() if self._cached_labels else self._defaults()

        # Carry-forward: only query VLM on keyframes (every `keyframe_stride`).
        if frame_index % self.keyframe_stride != 0 and self._cached_labels is not None:
            return self._cached_labels.copy()

        try:
            self._vlm_call_count = getattr(self, "_vlm_call_count", 0) + 1
            print(f"  [pass-c] frame {frame_index:3d}  call #{self._vlm_call_count} ... ",
                  end="", flush=True)
            labels = self._query_vlm_voted(image, label=f"f{frame_index}")
            self._cached_labels = labels
            # Print a compact summary of notable non-default concepts
            notable = {k: v for k, v in labels.items()
                       if k in self._binary_names and v == 1.0}
            notable.update({k: v for k, v in labels.items()
                            if k in self._categorical_index and v != 0.0})
            print("ok" + (f"  [{', '.join(f'{k}={v:.0f}' for k, v in notable.items())}]"
                          if notable else ""), flush=True)
            return labels.copy()
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            return self._cached_labels.copy() if self._cached_labels else self._defaults()

    def _raw_to_concepts(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Convert the parsed JSON response into a dict of float concept values."""
        out = self._defaults()
        for name, val in raw.items():
            if name in self._categorical_index:
                idx_map = self._categorical_index[name]
                if isinstance(val, str) and val.lower() in idx_map:
                    out[name] = float(idx_map[val.lower()])
            elif name in self._binary_names:
                if isinstance(val, bool):
                    out[name] = 1.0 if val else 0.0
                elif isinstance(val, (int, float)):
                    out[name] = 1.0 if val else 0.0
                elif isinstance(val, str):
                    out[name] = 1.0 if val.lower() in ("true", "yes", "1") else 0.0
        return out

    def _query_vlm_voted(self, image: np.ndarray, n_votes: int = 3,
                         label: str = "") -> Dict[str, float]:
        """Query VLM with lazy majority voting for rare binary concepts.

        Fires one VLM call. If any concept in _MAJORITY_VOTE_CONCEPTS fires true,
        fires n_votes-1 additional calls and requires a strict majority (> n/2 votes)
        to keep 1.0. Frames with no rare-concept hits pay no extra API cost.

        `label` is propagated to every verbose log line so concurrent frames'
        outputs can be told apart.
        """
        base_label = f"{label} v1" if label else ""
        base = self._raw_to_concepts(self._query_vlm(image, label=base_label))
        if not any(base.get(name, 0.0) >= 0.5 for name in _MAJORITY_VOTE_CONCEPTS):
            return base
        raws: List[Dict[str, float]] = [base]
        for i in range(n_votes - 1):
            vote_label = f"{label} v{i+2}" if label else f"v{i+2}"
            try:
                raws.append(self._raw_to_concepts(self._query_vlm(image, label=vote_label)))
            except Exception:
                pass
        for name in _MAJORITY_VOTE_CONCEPTS:
            if name not in base:
                continue
            votes_true = sum(1 for r in raws if r.get(name, 0.0) >= 0.5)
            new_val = 1.0 if votes_true > len(raws) / 2 else 0.0
            if self.verbose and base[name] >= 0.5:
                decision = "kept" if new_val >= 0.5 else "rejected"
                tag = f" {label}" if label else ""
                print(f"  [vlm-vote{tag}] {name}: {votes_true}/{len(raws)} → {decision}",
                      flush=True)
            base[name] = new_val
        return base

    def _query_vlm(self, image: np.ndarray, label: str = "") -> Dict[str, Any]:
        """Send image to VLM via the configured OpenAI-compatible backend.

        `label` is appended to verbose log tags so concurrent calls can be
        distinguished (e.g., ``[vlm-raw f12 v2]``).
        """
        import base64
        import json
        import os
        import time
        from io import BytesIO

        import requests
        from PIL import Image

        if self.backend not in _BACKEND_CONFIG:
            raise NotImplementedError(
                f"Backend {self.backend!r} not implemented. "
                f"Available: {sorted(_BACKEND_CONFIG)}"
            )
        cfg = _BACKEND_CONFIG[self.backend]

        api_key = os.environ.get(cfg["env_var"])
        if not api_key:
            raise RuntimeError(f"{cfg['env_var']} env var not set")

        # Resize + JPEG-encode to keep image tokens manageable
        pil = Image.fromarray(image)
        pil.thumbnail((self.max_image_dim, self.max_image_dim))
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            **cfg["extra_headers"],
        }

        # Up to 3 attempts with exponential backoff for rate-limit / transient errors.
        # 4xx errors (other than 429) are fatal and surface immediately — they
        # indicate a malformed payload that retrying won't fix.
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(cfg["url"], headers=headers, json=payload,
                                  timeout=self.request_timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(2 ** attempt)
                    continue
                if 400 <= r.status_code < 500:
                    raise RuntimeError(
                        f"VLM rejected request (HTTP {r.status_code}): {r.text[:500]}"
                    )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                reasoning = msg.get("reasoning_content", "")
                content = msg["content"]
                if self.verbose:
                    tag = f" {label}" if label else ""
                    if reasoning:
                        print(f"\n  [vlm-cot{tag}]\n{reasoning}\n  [/vlm-cot{tag}]",
                              flush=True)
                    print(f"  [vlm-raw{tag}] {content!r}", flush=True)
                # Strip stray markdown fences if present
                content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                return json.loads(content)
            except RuntimeError:
                raise
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM query failed after 3 attempts: {last_err}")
