"""Per-concept leaderboard from one or more eval metrics JSON files.

Reads the metrics_step*.json files written by training (cb_vlam.eval.metrics)
and prints, per concept type, every concept sorted best→worst by its primary
metric, so you can see which concepts the bottleneck projects well vs. which are
dead weight (near-chance / negative R²).

Primary metric per type: continuous → R², binary → AUROC, categorical → accuracy.

Single file  → full detail (primary + secondary metrics + support n).
Multiple files → primary metric as a column per file + delta(last − first), to
spotlight concepts that improved or regressed (e.g. overfitting between evals).
Concepts are sorted by the FIRST file's primary metric (None sorts last).

Pure stdlib — runs anywhere (kubectl cp the JSONs down, or run on the cluster).

Usage:
    python scripts/concept_leaderboard.py metrics_step1500.json
    python scripts/concept_leaderboard.py metrics_step1500.json metrics_step3000.json
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


def _label(path: Path) -> str:
    stem = path.stem
    return stem.replace("metrics_", "") or stem


def _fmt(x: Optional[float], w: int = 7) -> str:
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'n/a':>{w}}"


def _sort_key(v: Optional[float]):
    # None / missing sort to the bottom regardless of ascending order.
    return (v is None, -(v if isinstance(v, (int, float)) else 0.0))


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
        print(head)
        print("-" * len(head))
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
        print(head)
        print("-" * len(head))
        for n in names:
            d = pc(blocks[0], n)
            row = f"{n:32}" + "".join(_fmt(d.get(c), 11) for c in cols)
            row += f"{d.get('n', 0):>7}"
            print(row)

    # macro line(s)
    for lb, b in zip(labels, blocks):
        if b:
            macro = "  ".join(f"{k}={_fmt(b.get(k), 0).strip()}" for k in macro_keys)
            tag = f" [{lb}]" if multi else ""
            print(f"  macro{tag}: {macro}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", type=Path,
                    help="One or more metrics_step*.json files (order = column order).")
    args = ap.parse_args()

    datas, labels = [], []
    for p in args.metrics:
        raw = json.loads(p.read_text())
        # eval_gen.py nests concept metrics under "concepts"; unwrap transparently.
        datas.append(raw.get("concepts", raw))
        labels.append(_label(p))

    if any("loss" in d for d in datas):
        loss_str = "  ".join(f"{lb}={_fmt(d.get('loss'), 0).strip()}"
                             for lb, d in zip(labels, datas) if "loss" in d)
        print(f"val loss:  {loss_str}")

    for tname in ("continuous", "binary", "categorical"):
        _render_type(tname, datas, labels)


if __name__ == "__main__":
    main()
