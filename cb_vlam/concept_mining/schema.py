"""Concept schema for CB-VLAM-AV.

This is the single source of truth for which concepts the pipeline produces.
Extractors should write only these keys; the model expects only these keys.
"""

from typing import Dict, Iterable, List

# Each concept is described by (name, type, description).
# Types: "float" (continuous), "binary" (0.0 or 1.0), "categorical" (integer index)

PASS_A_CONCEPTS: List[Dict] = [
    {"name": "ego_speed", "type": "float", "desc": "Ego vehicle speed (m/s), normalized to [0, 1] over [0, 30]"},
    {"name": "ego_acceleration", "type": "float", "desc": "Ego longitudinal acceleration (m/s^2), normalized to [-1, 1] over [-5, 5]"},
    {"name": "ego_yaw_rate", "type": "float", "desc": "Ego yaw rate (rad/s), normalized to [-1, 1] over [-pi/4, pi/4]"},
    {"name": "ego_speed_delta", "type": "float", "desc": "Speed change over the last ~1.5 s (lookback_frames=3 at 2 Hz), normalized [-1, 1] over [-30, 30] m/s. Captures sustained deceleration trends; distinct from the instantaneous ego_acceleration signal."},
    {"name": "ego_stopped", "type": "binary", "desc": "1.0 if ego speed < 0.5 m/s"},
    {"name": "ego_braking", "type": "binary", "desc": "1.0 if ego acceleration < -1 m/s^2"},
    {"name": "ego_turning", "type": "binary", "desc": "1.0 if abs(ego yaw rate) > 0.1 rad/s"},
    {"name": "lateral_acceleration", "type": "float", "desc": "Lateral accel = yaw_rate * forward_speed (m/s^2), normalized to [-1, 1] over [-5, 5]"},
]

PASS_B_AGENT_CONCEPTS: List[Dict] = [
    {"name": "lead_vehicle_present", "type": "binary", "desc": "1.0 if there is a vehicle ahead of ego in the same lane within 50m"},
    {"name": "lead_vehicle_distance", "type": "float", "desc": "Distance to lead vehicle (m), normalized [0, 1] over [0, 50]. 1.0 if no lead vehicle."},
    {"name": "lead_vehicle_relative_velocity", "type": "float", "desc": "Closing rate (m/s, positive = approaching), normalized [-1, 1] over [-10, 10]"},
    {"name": "lead_vehicle_decelerating", "type": "binary", "desc": "1.0 if lead vehicle's acceleration < -1 m/s^2"},
    {"name": "pedestrian_ahead", "type": "binary", "desc": "1.0 if any pedestrian is within 10m ahead of ego in the forward corridor (±4m lateral)"},
    {"name": "nearest_pedestrian_distance", "type": "float", "desc": "Distance to nearest pedestrian ahead of ego (m), normalized [0, 1] over [0, 30]. 1.0 if none ahead."},
    {"name": "vehicle_count_nearby", "type": "float", "desc": "Number of moving (non-parked) vehicles within 30m (any direction), normalized [0, 1] over [0, 10]"},
    {"name": "cyclist_present", "type": "binary", "desc": "1.0 if any cyclist is within 20m AHEAD of ego"},
    {"name": "left_lane_blocked", "type": "binary", "desc": "1.0 if a moving vehicle (|v|>0.5 m/s) occupies the immediately adjacent left lane within ±20m"},
    {"name": "right_lane_blocked", "type": "binary", "desc": "1.0 if a moving vehicle (|v|>0.5 m/s) occupies the immediately adjacent right lane within ±20m"},
    {"name": "parked_cars_present", "type": "binary", "desc": "1.0 if any stationary vehicle (|v|<0.5 m/s) is in the side bands within ±15m"},
    # ── Deterministic annotation-based concepts (suffix _det disambiguates
    # from VLM versions of the same concept in Pass C).
    {"name": "emergency_vehicle_present_det", "type": "binary", "desc": "1.0 if any vehicle.emergency.* annotation within 30m (ambulance/police/fire)"},
    {"name": "construction_zone_det", "type": "binary", "desc": "1.0 if any construction-class object within 30m (movable_object.barrier ∪ movable_object.trafficcone ∪ human.pedestrian.construction_worker ∪ vehicle.construction)"},
    {"name": "animal_or_debris_on_road_det", "type": "binary", "desc": "1.0 if any animal or movable_object.debris annotation within 30m AHEAD of ego"},
    {"name": "pedestrian_intent_crossing_det", "type": "binary", "desc": "1.0 if any pedestrian AHEAD of ego with pedestrian.moving attribute is within 15m and within 5m of a ped_crossing polygon"},
    {"name": "pedestrian_density", "type": "float", "desc": "Count of pedestrians within 30m AHEAD of ego, normalized [0, 1] over [0, 5]"},
    {"name": "traffic_density_det", "type": "categorical", "values": ["light", "moderate", "heavy"], "desc": "Per-lane traffic density: moving vehicles within 30m divided by lane count near ego (HD-map lane layer in 10m radius). light=<1.0 v/lane, moderate=<2.0, heavy=>=2.0"},
    {"name": "time_to_collision_lead", "type": "float", "desc": "Lead distance / closing rate (s), clipped to [0, 1] over [0, 10]. 1.0 if no lead or not closing."},
    {"name": "following_distance_seconds", "type": "float", "desc": "Lead distance / ego speed (s), clipped to [0, 1] over [0, 5]. 1.0 if no lead or ego stopped."},
]

PASS_B_INFRA_CONCEPTS: List[Dict] = [
    {"name": "approaching_intersection", "type": "binary", "desc": "1.0 if distance to next intersection < 30m"},
    {"name": "distance_to_intersection", "type": "float", "desc": "Distance to next intersection (m), normalized [0, 1] over [0, 100]"},
    {"name": "lane_available_left", "type": "binary", "desc": "1.0 if a left lane exists on the current road"},
    {"name": "lane_available_right", "type": "binary", "desc": "1.0 if a right lane exists on the current road"},
    {"name": "speed_limit_normalized", "type": "float", "desc": "Speed limit on current segment (m/s), normalized [0, 1] over [0, 35]"},
    {"name": "road_curvature_ahead", "type": "float", "desc": "Curvature of the lane centerline 30m ahead (1/m), normalized [-1, 1] over [-0.05, 0.05]"},
    {"name": "in_intersection", "type": "binary", "desc": "1.0 if ego is currently inside an intersection polygon"},
    # ── Deterministic map-layer concepts.
    {"name": "over_stop_line", "type": "binary", "desc": "1.0 if ego position is on a stop_line polygon"},
    {"name": "nearest_crosswalk_distance", "type": "float", "desc": "Distance to nearest ped_crossing polygon (m), normalized [0, 1] over [0, 30]. 1.0 if none within 30m."},
    {"name": "on_walkway", "type": "binary", "desc": "1.0 if ego position is on a walkway polygon (anomaly: ego should not be on sidewalks)"},
    {"name": "in_carpark", "type": "binary", "desc": "1.0 if ego position is inside a carpark_area polygon"},
    {"name": "traffic_light_location_ahead", "type": "binary", "desc": "1.0 if any traffic_light polygon is within 50m ahead of ego (location only; state is VLM-only)"},
    {"name": "ego_lateral_offset_in_lane", "type": "float", "desc": "Perpendicular distance from ego to current lane centerline (m), normalized [-1, 1] over [-2, 2]. 0.0 if no lane found."},
]

PASS_C_SCENE_CONCEPTS: List[Dict] = [
    #scene context
    {"name": "weather",           "type": "categorical", "values": ["clear", "rain", "snow", "fog"],         "desc": "Weather condition"},
    {"name": "lighting",          "type": "categorical", "values": ["day", "dusk", "night"],                 "desc": "Lighting condition"},
    {"name": "traffic_density",   "type": "categorical", "values": ["light", "moderate", "heavy"],           "desc": "Density of nearby traffic"},
    {"name": "road_type",         "type": "categorical", "values": ["urban", "residential", "highway", "parking"], "desc": "Road type"},

    #traffic stuff/infrastructure
    {"name": "traffic_light_state", "type": "categorical", "values": ["none", "red", "yellow", "green"],     "desc": "Signal facing ego"},
    {"name": "stop_sign_visible",   "type": "binary",                                                         "desc": "Stop sign visible facing ego"},
    {"name": "speed_limit_sign",    "type": "categorical", "values": ["none", "20mph", "30mph", "40mph", "50mph", "65plus"], "desc": "Visible speed limit"},
    {"name": "construction_zone",   "type": "binary",                                                         "desc": "Cones/workers/orange signs present"},
    {"name": "school_zone",         "type": "binary",                                                         "desc": "School zone sign or school visible"},

    #surface conditions
    {"name": "surface_wet",           "type": "binary", "desc": "Road surface wet or recently wet"},
    {"name": "surface_obscured",      "type": "binary", "desc": "Road obscured by snow/leaves/debris/glare"},
    {"name": "lane_markings_visible", "type": "binary", "desc": "Lane markings clearly visible"},

    #visibility
    {"name": "visibility_degraded",  "type": "binary", "desc": "Reduced visibility (fog/heavy rain/glare)"},
    {"name": "headlights_required",  "type": "binary", "desc": "Headlights should be on for current conditions"},

    #behavior cues from other entities
    {"name": "lead_vehicle_brake_lights",  "type": "binary",                                                    "desc": "Brake lights illuminated on lead vehicle"},
    {"name": "lead_vehicle_turn_signal",   "type": "categorical", "values": ["none", "left", "right", "hazards"], "desc": "Turn signal state on lead vehicle"},
    {"name": "pedestrian_intent_crossing", "type": "binary",                                                    "desc": "Pedestrian intending to cross roadway"},
    {"name": "emergency_vehicle_present",  "type": "binary",                                                    "desc": "Police/ambulance/fire/tow with lights visible"},

    #nature of roads
    {"name": "road_narrows_ahead",      "type": "binary", "desc": "Road visibly narrows ahead"},
    {"name": "tunnel_or_bridge_ahead",  "type": "binary", "desc": "Tunnel or bridge ahead"},
    {"name": "hill_crest_ahead",        "type": "binary", "desc": "Hill crest hiding view ahead"},

    #obstructions/misc
    {"name": "accident_or_disabled_vehicle", "type": "binary", "desc": "Accident or disabled vehicle visible"},
    {"name": "animal_or_debris_on_road",     "type": "binary", "desc": "Animal or debris on the road"},
]


# Canonical pass identifiers, used by get_all_concepts(passes=...).
_PASS_BLOCKS = {
    "A": PASS_A_CONCEPTS,
    "B": PASS_B_AGENT_CONCEPTS + PASS_B_INFRA_CONCEPTS,
    "C": PASS_C_SCENE_CONCEPTS,
}

DEFAULT_PASSES = ("A", "B", "C")


def get_all_concepts(passes: Iterable[str] = DEFAULT_PASSES) -> List[Dict]:
    """Return concept definitions for the requested passes in canonical order.

    Args:
        passes: Iterable of pass identifiers from {"A", "B", "C"}. Order of the
            returned list always follows A → B → C regardless of the order
            given. Defaults to all three passes.

    The canonical concatenation order (A, then B-agent + B-infra, then C) is
    preserved so concept vector indices stay stable as long as the same set of
    passes is requested.
    """
    sel = {p.upper() for p in passes}
    invalid = sel - set(_PASS_BLOCKS)
    if invalid:
        raise ValueError(f"Unknown pass(es): {sorted(invalid)}. Valid: A, B, C.")
    out: List[Dict] = []
    for p in ("A", "B", "C"):
        if p in sel:
            out.extend(_PASS_BLOCKS[p])
    return out


def get_concept_keys(passes: Iterable[str] = DEFAULT_PASSES) -> List[str]:
    """Return list of concept names in canonical order for the given passes."""
    return [c["name"] for c in get_all_concepts(passes)]


CONCEPT_KEYS: List[str] = get_concept_keys()


def default_concept_dict() -> Dict[str, float]:
    """Return a dict with all concept keys set to 0.0."""
    return {k: 0.0 for k in CONCEPT_KEYS}
