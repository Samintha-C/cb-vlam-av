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
        raise NotImplementedError


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


class SceneContextExtractor(BaseExtractor):
    """Pass C: extract scene context (weather, lighting, etc.) using a VLM.

    Runs on keyframes only (1Hz, not 2Hz) and carries labels forward for
    non-keyframe timestamps. The VLM is queried with a structured prompt
    that asks for a JSON response covering all categorical concepts.

    Supports two backends:
    - "anthropic": uses the Anthropic API
    - "local": uses a HuggingFace VLM (e.g., Qwen2-VL) via transformers
    """

    def __init__(self,
                 backend: str = "anthropic",
                 model_name: str = "claude-haiku-4-5",
                 keyframe_stride: int = 2):
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self.keyframe_stride = keyframe_stride
        self._cached_labels: Optional[Dict[str, int]] = None

    def reset(self) -> None:
        super().reset()
        self._cached_labels = None

    def extract(self,
                image: np.ndarray,
                frame_index: int) -> Dict[str, float]:
        """Extract scene context concepts for one frame.

        Args:
            image: Front camera RGB image (H, W, 3) uint8.
            frame_index: Frame index within the scene (used for keyframe detection).

        Returns:
            Dict with keys matching PASS_C_SCENE_CONCEPTS schema.
            Categorical concepts are returned as integer indices into the
            values list defined in the schema.
        """
        raise NotImplementedError

    def _query_vlm(self, image: np.ndarray) -> Dict[str, int]:
        """Send image to VLM with structured prompt, parse JSON response.

        Returns:
            Dict mapping concept name to integer category index.
        """
        raise NotImplementedError
