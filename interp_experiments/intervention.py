"""Build the ground-truth activation vector and apply concept interventions.

The FinalPredictor reads ``CBL.to_activations`` — a flat vector of NORMALIZED,
bounded per-slot concept activations (continuous value, ``sigmoid`` binary prob,
``softmax`` categorical block). Because each slot's meaning and range are known,
ground-truth injection is unambiguous:

    continuous  → GT normalized value
    binary      → GT ∈ {0, 1}        (the sigmoid saturation point)
    categorical → one-hot(GT class)  (the softmax vertex)

A concept can only be intervened where it is SUPERVISED for that sample (mask
True) — you cannot inject ground truth you do not have.

This module exposes:
  - ``build_catalog``      : ordered per-concept metadata (name, kind, columns)
  - ``concept_to_col``     : (C, A) concept→activation-column indicator
  - ``gt_activation``      : (N, A) GT activation matrix + (N, C) supervised mask
  - ``apply``              : overwrite a per-sample concept-selection mask's slots
"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def build_catalog(cbl) -> List[Dict[str, Any]]:
    """Ordered concept metadata, matching the activation-vector layout.

    Order is continuous, then binary, then categorical (same as
    ``CBL.activation_slices``). Each entry: name, kind, type_idx (index within
    its type's target tensor), k (#columns), col0 (start column in the vector).
    """
    slices = cbl.activation_slices
    catalog: List[Dict[str, Any]] = []
    for ti, n in enumerate(cbl.cont_names):
        s = slices[n]
        catalog.append({"name": n, "kind": "continuous", "type_idx": ti,
                        "col0": s.start, "k": s.stop - s.start})
    for ti, n in enumerate(cbl.bin_names):
        s = slices[n]
        catalog.append({"name": n, "kind": "binary", "type_idx": ti,
                        "col0": s.start, "k": s.stop - s.start})
    for ti, n in enumerate(cbl.cat_names):
        s = slices[n]
        catalog.append({"name": n, "kind": "categorical", "type_idx": ti,
                        "col0": s.start, "k": s.stop - s.start})
    return catalog


def concept_to_col(catalog: List[Dict[str, Any]], activation_dim: int,
                   device, dtype=torch.float32) -> torch.Tensor:
    """(C, A) 0/1 matrix mapping each concept to its activation column(s)."""
    C = len(catalog)
    m = torch.zeros((C, activation_dim), device=device, dtype=dtype)
    for c, meta in enumerate(catalog):
        m[c, meta["col0"]:meta["col0"] + meta["k"]] = 1.0
    return m


def gt_activation(catalog: List[Dict[str, Any]], targets: Dict[str, torch.Tensor],
                  n: int, activation_dim: int, device, dtype=torch.float32):
    """Construct the ground-truth activation matrix and supervised mask.

    Returns:
        a_gt: (N, A) — each concept's columns set to its GT representation
              (other entries are never read, since ``apply`` only copies
              columns of intervened, supervised concepts).
        sup:  (N, C) bool — whether each concept is supervised per sample.
    """
    cont = targets["continuous"].to(device).to(dtype)
    cont_m = targets["continuous_mask"].to(device)
    binv = targets["binary"].to(device).to(dtype)
    bin_m = targets["binary_mask"].to(device)
    catv = targets["categorical"].to(device).long()
    cat_m = targets["categorical_mask"].to(device)

    a_gt = torch.zeros((n, activation_dim), device=device, dtype=dtype)
    sup = torch.zeros((n, len(catalog)), device=device, dtype=torch.bool)

    for c, meta in enumerate(catalog):
        ti, col0, k = meta["type_idx"], meta["col0"], meta["k"]
        if meta["kind"] == "continuous":
            a_gt[:, col0] = cont[:, ti]
            sup[:, c] = cont_m[:, ti].bool()
        elif meta["kind"] == "binary":
            a_gt[:, col0] = binv[:, ti]
            sup[:, c] = bin_m[:, ti].bool()
        else:  # categorical → one-hot
            a_gt[:, col0:col0 + k] = F.one_hot(catv[:, ti], num_classes=k).to(dtype)
            sup[:, c] = cat_m[:, ti].bool()
    return a_gt, sup


def apply(a_pred: torch.Tensor, a_gt: torch.Tensor, sel: torch.Tensor,
          c2col: torch.Tensor) -> torch.Tensor:
    """Overwrite the activation columns of the selected concepts with GT.

    Args:
        a_pred: (N, A) predicted activations.
        a_gt:   (N, A) GT activations.
        sel:    (N, C) bool/float — which concepts to intervene per sample.
        c2col:  (C, A) concept→column indicator (from ``concept_to_col``).
    Returns:
        (N, A) activations with the selected concepts' columns replaced by GT.
    """
    col_sel = (sel.to(a_pred.dtype) @ c2col) > 0   # (N, A) bool: column intervened
    return torch.where(col_sel, a_gt, a_pred)
