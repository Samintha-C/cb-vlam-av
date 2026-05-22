"""Feature diagnostics for the backbone extraction smoke test.

Loads the smoke output shards and checks:
  1. Feature dimensionality and variance (non-degenerate)
  2. endprompt vs afterplan position difference (correct indexing)
  3. Cross-sample diversity (model actually runs per-sample)

Exit code 0 = all checks pass. Exit code 1 = at least one WARN.
"""

import glob
import sys

import numpy as np
from datasets import load_from_disk

SMOKE_DIR = "/sc-rwx-vol/cbvlam/outputs/impromptu_features_smoke"
FEATS = ["feat_endprompt_final", "feat_endprompt_penult", "feat_afterplan_final"]

all_ok = True

for split in ["train", "test"]:
    shards = sorted(glob.glob(f"{SMOKE_DIR}/{split}/shard_*"))
    if not shards:
        print(f"[{split}] ERROR: no shards found")
        all_ok = False
        continue

    ds = load_from_disk(shards[0])
    rows = [ds[i] for i in range(len(ds))]
    n = len(rows)
    print(f"\n[{split}] {n} rows  schema={ds.column_names}")

    vecs = {f: np.array([r[f] for r in rows]) for f in FEATS}

    # 1. Per-feature stats
    for feat, v in vecs.items():
        norms = np.linalg.norm(v, axis=1)
        print(f"  {feat}")
        print(f"    dim={v.shape[1]}  std={v.std():.4f}  norm μ={norms.mean():.2f} σ={norms.std():.3f}")
        if v.std() < 1e-6:
            print(f"    WARN: near-zero variance — features may be degenerate")
            all_ok = False

    # 2. endprompt vs afterplan difference within each sample
    ep = vecs["feat_endprompt_final"]
    ap = vecs["feat_afterplan_final"]
    ep_norms = np.linalg.norm(ep, axis=1)
    ap_norms = np.linalg.norm(ap, axis=1)
    cos_ep_ap = (ep * ap).sum(axis=1) / (ep_norms * ap_norms)
    print(f"  endprompt↔afterplan cosine:  "
          f"mean={cos_ep_ap.mean():.4f}  min={cos_ep_ap.min():.4f}  max={cos_ep_ap.max():.4f}")
    if cos_ep_ap.mean() > 0.9999:
        print("    WARN: endprompt and afterplan features are nearly identical "
              "— position indexing may be wrong")
        all_ok = False

    # 3. Cross-sample diversity
    normed = ep / ep_norms[:, None]
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0)
    off_diag = sim[sim != 0]
    print(f"  cross-sample cosine (endprompt_final):  "
          f"max={off_diag.max():.4f}  mean={off_diag.mean():.4f}")
    if off_diag.max() > 0.9999:
        print("    WARN: all features are identical — model may not be running per-sample")
        all_ok = False

print()
if all_ok:
    print("=== All feature checks passed ===")
else:
    print("=== WARNINGS above — review before launching full job ===")
    sys.exit(1)
