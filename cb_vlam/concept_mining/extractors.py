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
                 turning_yaw_rate_radps: float = 0.1):
        super().__init__()
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_yaw_rate_radps = max_yaw_rate_radps
        self.stopped_speed_mps = stopped_speed_mps
        self.braking_accel_mps2 = braking_accel_mps2
        self.turning_yaw_rate_radps = turning_yaw_rate_radps

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
            }
            return {
                "ego_speed": 0.0,
                "ego_acceleration": 0.0,
                "ego_yaw_rate": 0.0,
                "ego_speed_delta": 0.0,
                "ego_stopped": 1.0,
                "ego_braking": 0.0,
                "ego_turning": 0.0,
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

        # Normalize per schema bounds.
        speed_norm = _clip(speed_mps / self.max_speed_mps, 0.0, 1.0)
        prev_speed_norm = _clip(prev_speed_mps / self.max_speed_mps, 0.0, 1.0)

        out = {
            "ego_speed": speed_norm,
            "ego_acceleration": _clip(accel_mps2 / self.max_accel_mps2, -1.0, 1.0),
            "ego_yaw_rate": _clip(yaw_rate_radps / self.max_yaw_rate_radps, -1.0, 1.0),
            "ego_speed_delta": speed_norm - prev_speed_norm,
            "ego_stopped": 1.0 if speed_mps < self.stopped_speed_mps else 0.0,
            "ego_braking": 1.0 if accel_mps2 < self.braking_accel_mps2 else 0.0,
            "ego_turning": 1.0 if abs(yaw_rate_radps) > self.turning_yaw_rate_radps else 0.0,
        }

        self.prev_state = {"speed_mps": speed_mps, "yaw": cur_yaw}
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
                 lane_width_m: float = 3.5):
        super().__init__()
        self.lead_vehicle_max_distance_m = lead_vehicle_max_distance_m
        self.nearby_vehicle_range_m = nearby_vehicle_range_m
        self.pedestrian_check_range_m = pedestrian_check_range_m
        self.lane_width_m = lane_width_m

    def extract(self,
                ego_pose: Dict[str, Any],
                annotations: List[Dict[str, Any]],
                prev_annotations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Extract agent concepts for one frame.

        Args:
            ego_pose: Ego pose record.
            annotations: List of nuScenes sample_annotation records visible at this timestamp.
                Each has 'translation', 'size', 'rotation', 'category_name', 'instance_token'.
            prev_annotations: Annotations from previous frame, for velocity computation.

        Returns:
            Dict with keys matching PASS_B_AGENT_CONCEPTS schema.
        """
        from pyquaternion import Quaternion

        ego_t   = np.array(ego_pose["translation"][:3], dtype=np.float64)
        ego_R_inv = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix

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
            entry = {"ann": ann, "pos": pos, "dist": dist}

            if cat.startswith("vehicle.bicycle") or cat.startswith("vehicle.motorcycle"):
                cyclists.append(entry)
            elif cat.startswith("vehicle."):
                vehicles.append(entry)
            elif cat.startswith("human.pedestrian"):
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

            # Lead longitudinal velocity from instance tracking
            if inst in prev_by_inst:
                global_delta = (
                    np.array(lead["ann"]["translation"][:3])
                    - np.array(prev_by_inst[inst]["translation"][:3])
                )
                lead_vel_fwd = float((ego_R_inv @ global_delta)[0] / dt)

            # Closing rate = ego advancing faster than lead (positive → approaching)
            closing_rate = ego_speed_mps - lead_vel_fwd

            # Lead decelerating: compare lead vel to previous frame's stored lead vel
            prev_lead_vel = self.prev_state.get("lead_vel_fwd", 0.0) if self.prev_state else 0.0
            lead_accel = (lead_vel_fwd - prev_lead_vel) / dt

            out_lead = {
                "lead_vehicle_present":           1.0,
                "lead_vehicle_distance":          float(np.clip(lead["pos"][0] / self.lead_vehicle_max_distance_m, 0.0, 1.0)),
                "lead_vehicle_relative_velocity": float(np.clip(closing_rate / 10.0, -1.0, 1.0)),
                "lead_vehicle_decelerating":      1.0 if lead_accel < -1.0 else 0.0,
            }
        else:
            out_lead = {
                "lead_vehicle_present":           0.0,
                "lead_vehicle_distance":          1.0,
                "lead_vehicle_relative_velocity": 0.0,
                "lead_vehicle_decelerating":      0.0,
            }

        # ── Pedestrians ───────────────────────────────────────────────────────
        # pedestrian_in_crosswalk_ahead: without HD map we approximate as any
        # pedestrian within 10 m ahead in the forward corridor (±4 m lateral).
        peds_crosswalk = [
            p for p in pedestrians
            if 0 < p["pos"][0] < 10.0 and abs(p["pos"][1]) < 4.0
        ]
        nearest_ped_dist = min((p["dist"] for p in pedestrians), default=None)

        out_ped = {
            "pedestrian_in_crosswalk_ahead": 1.0 if peds_crosswalk else 0.0,
            "nearest_pedestrian_distance":   float(np.clip(nearest_ped_dist / 30.0, 0.0, 1.0))
                                             if nearest_ped_dist is not None else 1.0,
        }

        # ── Nearby vehicles & cyclists ────────────────────────────────────────
        nearby_count = sum(1 for v in vehicles if v["dist"] < self.nearby_vehicle_range_m)
        cyc_close    = any(c["dist"] < self.pedestrian_check_range_m for c in cyclists)

        # Adjacent lane occupancy: check vehicles in the lateral band of the next lane
        L = self.lane_width_m
        left_blocked  = any(
            v for v in vehicles
            if L * 0.5 < v["pos"][1] < L * 2.5 and abs(v["pos"][0]) < 10.0
        )
        right_blocked = any(
            v for v in vehicles
            if -L * 2.5 < v["pos"][1] < -L * 0.5 and abs(v["pos"][0]) < 10.0
        )

        out_misc = {
            "vehicle_count_nearby":  float(np.clip(nearby_count / 10.0, 0.0, 1.0)),
            "cyclist_present":       1.0 if cyc_close else 0.0,
            "left_lane_blocked":     1.0 if left_blocked  else 0.0,
            "right_lane_blocked":    1.0 if right_blocked else 0.0,
        }

        # ── Update state ──────────────────────────────────────────────────────
        self.prev_state = {
            "timestamp":    ego_pose["timestamp"],
            "ego_t":        ego_t.copy(),
            "lead_vel_fwd": lead_vel_fwd,
        }

        return {**out_lead, **out_ped, **out_misc}


class InfrastructureExtractor(BaseExtractor):
    """Pass B (infrastructure): extract concepts from HD map.

    Uses nuScenes' NuScenesMap API to query:
    - Current lane and lane geometry
    - Distance to next intersection
    - Lane availability (left/right lanes exist)
    - Road curvature ahead

    Speed limit is not directly available in nuScenes maps; either omit it
    for now or infer it from road type (highway vs urban).
    """

    def __init__(self,
                 lookahead_m: float = 30.0,
                 max_intersection_distance_m: float = 100.0):
        super().__init__()
        self.lookahead_m = lookahead_m
        self.max_intersection_distance_m = max_intersection_distance_m

    def extract(self,
                ego_pose: Dict[str, Any],
                nusc_map: Any) -> Dict[str, float]:
        """Extract infrastructure concepts for one frame.

        Args:
            ego_pose: Ego pose record.
            nusc_map: NuScenesMap instance for the current scene's location.

        Returns:
            Dict with keys matching PASS_B_INFRA_CONCEPTS schema.
        """
        raise NotImplementedError


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_pass_c_prompt() -> str:
    """Build the structured prompt listing every Pass C concept and its options."""
    from cb_vlam.concept_mining.schema import PASS_C_SCENE_CONCEPTS

    categorical_lines = []
    binary_lines = []
    for c in PASS_C_SCENE_CONCEPTS:
        if c["type"] == "categorical":
            vals = "/".join(c["values"])
            categorical_lines.append(f'- "{c["name"]}" (one of: {vals}) — {c["desc"]}')
        else:
            binary_lines.append(f'- "{c["name"]}" (true/false) — {c["desc"]}')

    return (
        "You are analyzing a front-camera image from an autonomous vehicle. "
        "Extract the following concepts and return ONLY a valid JSON object — "
        "no markdown fences, no commentary.\n\n"
        "CATEGORICAL CONCEPTS (return the string value):\n"
        + "\n".join(categorical_lines) + "\n\n"
        "BINARY CONCEPTS (return true or false):\n"
        + "\n".join(binary_lines) + "\n\n"
        "Guidelines:\n"
        "- 'lead vehicle' = the vehicle directly ahead of ego in the same lane.\n"
        "- For categoricals with a 'none' option, return 'none' if not visible/applicable.\n"
        "- Be conservative: if uncertain, return the default (false / 'none' / 'clear').\n"
        "- Return every key from the lists above. Return ONLY the JSON object."
    )


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
                 request_timeout: int = 60):
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self.keyframe_stride = keyframe_stride
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality
        self.request_timeout = request_timeout
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
        if image is None:
            return self._cached_labels.copy() if self._cached_labels else self._defaults()

        # Carry-forward: only query VLM on keyframes (every `keyframe_stride`).
        if frame_index % self.keyframe_stride != 0 and self._cached_labels is not None:
            return self._cached_labels.copy()

        try:
            raw = self._query_vlm(image)
            labels = self._raw_to_concepts(raw)
            self._cached_labels = labels
            return labels.copy()
        except Exception as e:
            print(f"[pass-c] VLM call failed at frame {frame_index}: {e}")
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

    def _query_vlm(self, image: np.ndarray) -> Dict[str, Any]:
        """Send image to VLM via OpenRouter; return the parsed JSON dict."""
        import base64
        import json
        import os
        import time
        from io import BytesIO

        import requests
        from PIL import Image

        if self.backend != "openrouter":
            raise NotImplementedError(f"Backend {self.backend!r} not implemented")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY env var not set")

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
            "HTTP-Referer":  "https://github.com/Samintha-C/cb-vlam-av",
            "X-Title":       "cb-vlam-av",
        }

        # Up to 3 attempts with exponential backoff for rate-limit / transient errors
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(_OPENROUTER_URL, headers=headers, json=payload,
                                  timeout=self.request_timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                # Strip stray markdown fences if present
                content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                return json.loads(content)
            except Exception as e:
                last_err = str(e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM query failed after 3 attempts: {last_err}")
