"""Per-concept leaderboard from one or more eval metrics JSON files.

Accepts two JSON formats:
  - eval_gen.py output  (keys: split, n, trajectory, concepts)
  - training metrics    (keys: step, val_quality, concept_loss, …)

Prints a trajectory header (L2 @1s/2s/3s/Avg + ADE/FDE) when available,
then per-concept tables sorted best→worst by primary metric:
  continuous → R²,  binary → AUROC,  categorical → accuracy.

Single file  → full detail (primary + secondary metrics + support n).
Multiple files → primary metric as a column per file + delta(last − first).

Pure stdlib — runs anywhere.

Usage:
    python scripts/concept_leaderboard.py eval_stp3.json
    python scripts/concept_leaderboard.py eval_stp3.json eval_test.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# type -> (primary_key, secondary_keys, macro_keys)
_SPEC = {
    "continuous":  ("r2",       ["r2_floored", "std", "mae"], ["macro_mae", "macro_r2", "macro_r2_floored"]),
    "binary":      ("auroc",    ["f1", "pos_rate"],           ["macro_auroc", "macro_f1"]),
    "categorical": ("accuracy", ["macro_f1"],                 ["macro_accuracy", "macro_f1"]),
}

_L2_KEYS = [("L2@1s", "l2_1s"), ("L2@2s", "l2_2s"), ("L2@3s", "l2_3s"), ("Avg", "l2_avg")]
_TRAJ_KEYS = _L2_KEYS + [("ADE", "ade_m"), ("FDE", "fde_m")]


def _label(path: Path) -> str:
    return path.stem or str(path)


def _fmt(x: Optional[float], w: int = 7) -> str:
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'n/a':>{w}}"


def _sort_key(v: Optional[float]):
    return (v is None, -(v if isinstance(v, (int, float)) else 0.0))


def _render_trajectory(trajs: List[Optional[Dict]], labels: List[str]) -> None:
    """Print L2 @1s/2s/3s/Avg + ADE/FDE header if any file has trajectory data."""
    if not any(t for t in trajs):
        return
    multi = len(trajs) > 1
    print(f"\n{'='*72}\nTRAJECTORY  (ST-P3 cumulative L2, Impromptu q7 convention)\n{'='*72}")
    col_w = 9
    head = f"{'metric':12}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(head); print("-" * len(head))
    for display, key in _TRAJ_KEYS:
        row = f"{display:12}"
        for t in trajs:
            row += _fmt((t or {}).get(key), col_w)
        print(row)
    # n per file
    n_row = f"{'n':12}" + "".join(f"{(t or {}).get('n', 'n/a'):>{col_w}}" for t in trajs)
    print(n_row)


def _render_type(tname: str, datas: List[Dict[str, Any]], labels: List[str]) -> None:
    primary, secondary, macro_keys = _SPEC[tname]
    blocks = [d.get(tname) for d in datas]
    if not any(blocks):
        return
    first = next(b for b in blocks if b)
    names = list(first["per_concept"].keys())

    def pc(block, name):
        return (block or {}).get("per_concept", {}).get(name, {})

    names.sort(key=lambda n: _sort_key(pc(blocks[0], n).get(primary)))

    print(f"\n{'='*72}\n{tname.upper()}  (primary: {primary})\n{'='*72}")
    multi = len(datas) > 1

    if multi:
        head = f"{'concept':32}" + "".join(f"{(primary + '@' + lb):>11}" for lb in labels)
        head += f"{'Δ':>9}{'n':>7}"
        print(head); print("-" * len(head))
        for n in names:
            row = f"{n:32}"
            vals = [pc(b, n).get(primary) for b in blocks]
            row += "".join(_fmt(v, 11) for v in vals)
            delta = (vals[-1] - vals[0]) if (isinstance(vals[0], (int, float))
                                             and isinstance(vals[-1], (int, float))) else None
            row += _fmt(delta, 9)
            row += f"{pc(blocks[0], n).get('n', 0):>7}"
            print(row)
    else:
        cols = [primary] + secondary
        head = f"{'concept':32}" + "".join(f"{c:>11}" for c in cols) + f"{'n':>7}"
        print(head); print("-" * len(head))
        for n in names:
            d = pc(blocks[0], n)
            row = f"{n:32}" + "".join(_fmt(d.get(c), 11) for c in cols)
            row += f"{d.get('n', 0):>7}"
            print(row)

    for lb, b in zip(labels, blocks):
        if b:
            macro = "  ".join(f"{k}={_fmt(b.get(k), 0).strip()}" for k in macro_keys)
            tag = f" [{lb}]" if multi else ""
            print(f"  macro{tag}: {macro}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", type=Path)
    args = ap.parse_args()

    raws, datas, trajs, labels = [], [], [], []
    for p in args.metrics:
        raw = json.loads(p.read_text())
        raws.append(raw)
        # eval_gen.py wraps concept metrics under "concepts"; unwrap transparently.
        datas.append(raw.get("concepts", raw))
        trajs.append(raw.get("trajectory"))
        labels.append(_label(p))

    # Concept-loss header (present in both formats under different keys).
    closs_parts = []
    for lb, raw, d in zip(labels, raws, datas):
        v = (raw.get("trajectory") or {}).get("concept_loss") or d.get("loss")
        if v is not None:
            closs_parts.append(f"{lb}={_fmt(v, 0).strip()}")
    if closs_parts:
        print(f"concept loss:  {'  '.join(closs_parts)}")

    _render_trajectory(trajs, labels)

    for tname in ("continuous", "binary", "categorical"):
        _render_type(tname, datas, labels)


if __name__ == "__main__":
    main()
