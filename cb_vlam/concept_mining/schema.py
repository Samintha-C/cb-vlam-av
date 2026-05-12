"""Concept schema for CB-VLAM-AV.

This is the single source of truth for which concepts the pipeline produces.
Extractors should write only these keys; the model expects only these keys.
"""

from typing import Dict, List

# Each concept is described by (name, type, description).
# Types: "float" (continuous), "binary" (0.0 or 1.0), "categorical" (integer index)

PASS_A_CONCEPTS: List[Dict] = [
    {"name": "ego_speed", "type": "float", "desc": "Ego vehicle speed (m/s), normalized to [0, 1] over [0, 30]"},
    {"name": "ego_acceleration", "type": "float", "desc": "Ego longitudinal acceleration (m/s^2), normalized to [-1, 1] over [-5, 5]"},
    {"name": "ego_yaw_rate", "type": "float", "desc": "Ego yaw rate (rad/s), normalized to [-1, 1] over [-pi/4, pi/4]"},
    {"name": "ego_speed_delta", "type": "float", "desc": "Frame-to-frame change in normalized ego speed"},
    {"name": "ego_stopped", "type": "binary", "desc": "1.0 if ego speed < 0.5 m/s"},
    {"name": "ego_braking", "type": "binary", "desc": "1.0 if ego acceleration < -1 m/s^2"},
    {"name": "ego_turning", "type": "binary", "desc": "1.0 if abs(ego yaw rate) > 0.1 rad/s"},
]

PASS_B_AGENT_CONCEPTS: List[Dict] = [
    {"name": "lead_vehicle_present", "type": "binary", "desc": "1.0 if there is a vehicle ahead of ego in the same lane within 50m"},
    {"name": "lead_vehicle_distance", "type": "float", "desc": "Distance to lead vehicle (m), normalized [0, 1] over [0, 50]. 1.0 if no lead vehicle."},
    {"name": "lead_vehicle_relative_velocity", "type": "float", "desc": "Closing rate (m/s, positive = approaching), normalized [-1, 1] over [-10, 10]"},
    {"name": "lead_vehicle_decelerating", "type": "binary", "desc": "1.0 if lead vehicle's acceleration < -1 m/s^2"},
    {"name": "pedestrian_in_crosswalk_ahead", "type": "binary", "desc": "1.0 if any pedestrian is in a crosswalk within 20m ahead of ego"},
    {"name": "nearest_pedestrian_distance", "type": "float", "desc": "Distance to nearest pedestrian (m), normalized [0, 1] over [0, 30]. 1.0 if none."},
    {"name": "vehicle_count_nearby", "type": "float", "desc": "Number of vehicles within 30m, normalized [0, 1] over [0, 10]"},
    {"name": "cyclist_present", "type": "binary", "desc": "1.0 if any cyclist is within 20m"},
    {"name": "left_lane_blocked", "type": "binary", "desc": "1.0 if a vehicle occupies the left adjacent lane within 10m"},
    {"name": "right_lane_blocked", "type": "binary", "desc": "1.0 if a vehicle occupies the right adjacent lane within 10m"},
]

PASS_B_INFRA_CONCEPTS: List[Dict] = [
    {"name": "approaching_intersection", "type": "binary", "desc": "1.0 if distance to next intersection < 30m"},
    {"name": "distance_to_intersection", "type": "float", "desc": "Distance to next intersection (m), normalized [0, 1] over [0, 100]"},
    {"name": "lane_available_left", "type": "binary", "desc": "1.0 if a left lane exists on the current road"},
    {"name": "lane_available_right", "type": "binary", "desc": "1.0 if a right lane exists on the current road"},
    {"name": "speed_limit_normalized", "type": "float", "desc": "Speed limit on current segment (m/s), normalized [0, 1] over [0, 35]"},
    {"name": "road_curvature_ahead", "type": "float", "desc": "Curvature of the lane centerline 30m ahead (1/m), normalized [-1, 1] over [-0.05, 0.05]"},
    {"name": "in_intersection", "type": "binary", "desc": "1.0 if ego is currently inside an intersection polygon"},
]

PASS_C_SCENE_CONCEPTS: List[Dict] = [
    {"name": "weather", "type": "categorical", "values": ["clear", "rain", "snow", "fog"], "desc": "Weather condition from VLM"},
    {"name": "lighting", "type": "categorical", "values": ["day", "dusk", "night"], "desc": "Lighting condition from VLM"},
    {"name": "traffic_density", "type": "categorical", "values": ["light", "moderate", "heavy"], "desc": "Traffic density from VLM"},
    {"name": "road_type", "type": "categorical", "values": ["highway", "urban", "residential", "parking"], "desc": "Road type from VLM"},
]


def get_all_concepts() -> List[Dict]:
    """Return all concept definitions in canonical order."""
    return (
        PASS_A_CONCEPTS
        + PASS_B_AGENT_CONCEPTS
        + PASS_B_INFRA_CONCEPTS
        + PASS_C_SCENE_CONCEPTS
    )


def get_concept_keys() -> List[str]:
    """Return list of concept names in canonical order."""
    return [c["name"] for c in get_all_concepts()]


CONCEPT_KEYS: List[str] = get_concept_keys()


def default_concept_dict() -> Dict[str, float]:
    """Return a dict with all concept keys set to 0.0."""
    return {k: 0.0 for k in CONCEPT_KEYS}
