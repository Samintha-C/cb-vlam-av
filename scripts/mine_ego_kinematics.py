"""Route A / Phase 1 — mine measured ego-kinematic concepts from ego history.

Derives named low-dimensional kinematic concepts DETERMINISTICALLY from the
ImpromptuVLA ego-history block (7 past BEV [x, y] positions, 3 s @ 0.5 s, current
frame re-centred to [0,0]) via finite differences:

    ego_speed            |v| of the most recent step            (m/s)
    ego_accel_long       d/dt of longitudinal (x) velocity      (m/s^2)
    ego_yaw_rate         d/dt of heading (dir. of motion)       (rad/s)
    ego_path_curvature   yaw_rate / speed                       (1/m)

These are a new concept KIND = "measured": not predicted by the CBL (no L_c, no
predictor noise), injected as exact activations, always supervised. This adds no
information — it moves the dominant trajectory signal (ego kinematics) out of the
unnamed residual into named, inspectable, intervenable slots.

Source (VERIFY-resolved): the released Impromptu JSON carries the ego history only
as the formatted `(t-Xs) [x, y]` block in messages[0].content — there is no
structured sidecar (data/dataset_info.json has no history feature; the structured
`traj` arrays exist only in the upstream data_traj_generate generators, not the
shipped file). This miner extracts the bracketed COORDINATE arrays deterministically
(a fixed template, not NL interpretation) and writes a structured artifact, so all
downstream code (ridge harness, Phase-2 dataset) reads structured data, never the
prompt prose. nuScenes ego_pose is the canonical alternative for Phase-2 store
mining but is deferred: the Phase-1 gate needs A5 to be the finite-difference
summary of the SAME 7-point history that A6 holds raw.

Normalization matches the existing continuous concepts / KinematicExtractor bounds:
speed/30→[0,1], accel/5→[-1,1], yaw/(π/4)→[-1,1], curvature/0.2→[-1,1].

Writes:
  <out>/ego_kinematics.json          {token: {raw, norm, hist14, n_hist}}
  <out>/ego_kinematics_manifest.json  measured-concept schema fragment (kind tag,
                                       normalization ranges) + the 14-dim history block spec

Usage:
  python scripts/mine_ego_kinematics.py \
    --impromptu_train /sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_train.json \
    --impromptu_test  /sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_test.json \
    --out outputs/diagnostics [--preview]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

DT = 0.5                      # history sampling interval (s)
MAX_SPEED = 30.0             # m/s   (KinematicExtractor.max_speed_mps)
MAX_ACCEL = 5.0             # m/s^2 (KinematicExtractor.max_accel_mps2)
MAX_YAW = np.pi / 4         # rad/s (KinematicExtractor.max_yaw_rate_radps)
MAX_CURV = 0.2              # 1/m   (~5 m radius; tight urban turn)
STOPPED = 0.5              # m/s   (heading undefined below this)
N_HIST = 7                  # full history length (3 s @ 0.5 s incl. current)

MEASURED = ["ego_speed", "ego_accel_long", "ego_yaw_rate", "ego_path_curvature"]

# Structured extraction of the `(t-Xs) [x, y]` block after the fixed template
# anchor. Not NL parsing — this pulls the bracketed coordinate arrays only.
_ANCHOR = "format [x, y]:"
_TUPLE = re.compile(r"\(t-([0-9.]+)s\)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_ego_history(prompt: str):
    """Return past positions oldest→newest as (n,2) float array, current last.

    Reads only the segment after the template anchor; each match is a
    `(t-Xs) [x, y]`. Sorted by descending t (t-3.0s → t-0.0s), so the last row
    is the current frame (≈[0,0]).
    """
    seg = prompt.split(_ANCHOR)[-1]
    hits = _TUPLE.findall(seg)
    if not hits:
        return np.zeros((0, 2), dtype=np.float64)
    rows = sorted(((float(t), float(x), float(y)) for t, x, y in hits),
                  key=lambda r: -r[0])                       # oldest (largest t) first
    return np.array([[x, y] for _, x, y in rows], dtype=np.float64)


def compute_kinematics(P: np.ndarray):
    """Finite-difference kinematics from history positions P (n,2), current last.

    Returns (raw_dict, norm_dict, hist14). Degenerate histories (n<2, or a
    stationary current step) yield 0 for the undefined quantities — heading /
    yaw / curvature are undefined at rest, per the extractor's first-frame rule.
    """
    raw = {k: 0.0 for k in MEASURED}
    n = len(P)
    if n >= 2:
        v = (P[1:] - P[:-1]) / DT                            # (n-1, 2) step velocities
        v_last = v[-1]
        speed = float(np.linalg.norm(v_last))
        raw["ego_speed"] = speed
        if n >= 3:
            # longitudinal (forward = +x) acceleration from the last two steps
            raw["ego_accel_long"] = float((v[-1, 0] - v[-2, 0]) / DT)
            # heading only where the step is non-degenerate; yaw rate = Δheading/dt
            if speed >= STOPPED and np.linalg.norm(v[-2]) >= STOPPED:
                th_last = np.arctan2(v_last[1], v_last[0])
                th_prev = np.arctan2(v[-2, 1], v[-2, 0])
                dth = float((th_last - th_prev + np.pi) % (2 * np.pi) - np.pi)
                yaw = dth / DT
                raw["ego_yaw_rate"] = yaw
                raw["ego_path_curvature"] = yaw / max(speed, 1e-6)
    norm = {
        "ego_speed": _clip(raw["ego_speed"] / MAX_SPEED, 0.0, 1.0),
        "ego_accel_long": _clip(raw["ego_accel_long"] / MAX_ACCEL, -1.0, 1.0),
        "ego_yaw_rate": _clip(raw["ego_yaw_rate"] / MAX_YAW, -1.0, 1.0),
        "ego_path_curvature": _clip(raw["ego_path_curvature"] / MAX_CURV, -1.0, 1.0),
    }
    hist14 = _hist_block(P)
    return raw, norm, hist14


def _hist_block(P: np.ndarray):
    """Raw 14-dim history (7×[x,y]); short histories left-padded with the oldest
    known position (constant-position hold → zero velocity across the pad)."""
    if len(P) == 0:
        return [0.0] * (N_HIST * 2)
    if len(P) < N_HIST:
        pad = np.repeat(P[:1], N_HIST - len(P), axis=0)
        P = np.concatenate([pad, P], axis=0)
    else:
        P = P[-N_HIST:]
    return P.reshape(-1).tolist()


def _clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


def _load_records(jsons):
    rec = {}
    for jp in jsons:
        for r in json.load(open(jp)):
            rec.setdefault(r["id"], r)
    return rec


def mine(jsons):
    rec = _load_records(jsons)
    out, n_short = {}, 0
    for tok, r in rec.items():
        P = parse_ego_history(r["messages"][0]["content"])
        if len(P) < N_HIST:
            n_short += 1
        raw, norm, hist14 = compute_kinematics(P)
        out[tok] = {"raw": raw, "norm": norm, "hist14": hist14, "n_hist": int(len(P))}
    return out, n_short


def _manifest_fragment():
    ranges = {"ego_speed": [0.0, MAX_SPEED], "ego_accel_long": [-MAX_ACCEL, MAX_ACCEL],
              "ego_yaw_rate": [-MAX_YAW, MAX_YAW], "ego_path_curvature": [-MAX_CURV, MAX_CURV]}
    return {
        "kind": "measured",
        "note": "Injected exact activations (no CBL prediction, no L_c). Included in "
                "gt_activations (self-GT, always supervised), and in the adversarial "
                "probe/disentanglement targets so the residual is pressured off them.",
        "concepts": [{"name": n, "kind": "measured", "type": "float",
                      "norm_range": ranges[n]} for n in MEASURED],
        "history_block": {"name": "ego_history_14d", "dims": N_HIST * 2,
                          "role": "A6 ceiling comparator only — never a model concept block"},
        "source": "ImpromptuVLA ego-history block (structured coord extraction); "
                  "finite differences; nuScenes ego_pose deferred to Phase-2 store mining",
    }


# ── optional local gate preview (no concept store needed) ────────────────────

def _preview(kin, jsons, horizon=6, seed=1234):
    """Concept-store-free lower bound: fit ridge(kinematics→future) and
    ridge(hist14→future) using ONLY the Impromptu JSON (past + future both in
    messages), scored with the real ST-P3 metric. train.json→train,
    test.json→eval (scene-disjoint proxy for the store's exact val∪test split)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from cb_vlam.training.dataset import parse_trajectory
    from cb_vlam.eval.metrics import trajectory_l2
    import torch

    train_toks = {r["id"] for r in json.load(open(jsons[0]))}
    rec = _load_records(jsons)
    Xk, Xh, Y, V, split = [], [], [], [], []
    for tok, r in rec.items():
        traj, mask = parse_trajectory(r["messages"][1]["content"], horizon)
        if not mask.all():
            continue                                          # complete future only
        k = kin[tok]
        Xk.append([k["norm"][n] for n in MEASURED]); Xh.append(k["hist14"])
        Y.append(traj); V.append(mask); split.append(tok in train_toks)
    Xk, Xh, Y = np.array(Xk), np.array(Xh), np.array(Y, np.float64)
    V = np.array(V); split = np.array(split)
    tr, ev = split, ~split
    alphas = np.logspace(-3, 3, 13)

    def score(X):
        sc = StandardScaler().fit(X[tr])
        reg = RidgeCV(alphas=alphas, cv=5).fit(sc.transform(X[tr]), Y[tr])
        p = reg.predict(sc.transform(X[ev]))
        n = ev.sum()
        pred = torch.from_numpy(p).reshape(n, horizon, 2)
        gt = torch.from_numpy(Y[ev]).reshape(n, horizon, 2)
        valid = torch.from_numpy(V[ev].reshape(n, horizon, 2)[..., 0].astype(np.float64))
        return trajectory_l2(pred, gt, valid, horizon)

    print(f"\n=== LOCAL PREVIEW (Impromptu-only, proxy split: train.json→train "
          f"test.json→eval, n_train={tr.sum()} n_eval={ev.sum()}) ===")
    for name, X in [("kinematics-only (4d, A5 increment)", Xk),
                    ("history-only (14d, A6 increment)", Xh)]:
        m = score(X)
        print(f"  {name:38} L2@1s={m['l2_1s']:.3f} @2s={m['l2_2s']:.3f} "
              f"@3s={m['l2_3s']:.3f} Avg={m['l2_avg']:.3f}  ADE={m['ade_m']:.3f}")
    print("  [reference: full model residual-on ST-P3 Avg ≈ 0.389 m]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impromptu_train", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_train.json"))
    ap.add_argument("--impromptu_test", type=Path,
                    default=Path("/sc-rwx-vol/cbvlam/Impromptu-VLA/nuscenes_test.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnostics"))
    ap.add_argument("--preview", action="store_true",
                    help="also run the concept-store-free ridge preview (kinematics/history → future)")
    args = ap.parse_args()

    jsons = [args.impromptu_train, args.impromptu_test]
    print(f"mining ego kinematics from {[str(j) for j in jsons]} ...")
    kin, n_short = mine(jsons)

    # distribution sanity (raw SI)
    sp = np.array([v["raw"]["ego_speed"] for v in kin.values()])
    ac = np.array([v["raw"]["ego_accel_long"] for v in kin.values()])
    yr = np.array([v["raw"]["ego_yaw_rate"] for v in kin.values()])
    print(f"  n={len(kin)}  short_history(<{N_HIST})={n_short} ({100*n_short/len(kin):.1f}%)")
    print(f"  speed  m/s: mean={sp.mean():.2f} p50={np.percentile(sp,50):.2f} "
          f"p95={np.percentile(sp,95):.2f} max={sp.max():.2f}  frac<{STOPPED}={np.mean(sp<STOPPED):.2f}")
    print(f"  accel  m/s^2: mean={ac.mean():.2f} p5={np.percentile(ac,5):.2f} "
          f"p95={np.percentile(ac,95):.2f}")
    print(f"  yawrate rad/s: p5={np.percentile(yr,5):.3f} p95={np.percentile(yr,95):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ego_kinematics.json").write_text(json.dumps(kin))
    (args.out / "ego_kinematics_manifest.json").write_text(json.dumps(_manifest_fragment(), indent=2))
    print(f"wrote {args.out/'ego_kinematics.json'} ({len(kin)} tokens) and manifest fragment")

    if args.preview:
        _preview(kin, jsons)


if __name__ == "__main__":
    main()
