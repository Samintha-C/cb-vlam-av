"""Concept loss L_c for CB-VLAM-AV.

Masked, per-type supervision of the CBL outputs against ConceptStore targets:

    continuous   → MSE
    binary       → BCE-with-logits (optional per-concept pos_weight for the
                   heavy class imbalance — many binary concepts are rare events)
    categorical  → cross-entropy, per concept

Masks (True = supervise) come from ConceptStore and implement the "saturated =
absent" toggles; masked-out elements contribute nothing to the loss or its
denominator. This is the only loss needed for Phase 1 (concept-projection
accuracy); the task/adversarial/sparsity terms come later.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _masked_mean(per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of per_elem over True positions in mask; 0 if none are supervised."""
    m = mask.to(per_elem.dtype)
    denom = m.sum().clamp(min=1.0)
    return (per_elem * m).sum() / denom


def concept_loss(
    pred: Dict[str, Any],
    target: Dict[str, Any],
    *,
    bin_pos_weight: Optional[torch.Tensor] = None,
    weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the masked per-type concept loss.

    Args:
        pred: CBL output — {"continuous", "binary_logits", "categorical_logits"}.
        target: batched ConceptStore targets + masks — keys "continuous",
            "continuous_mask", "binary", "binary_mask", "categorical",
            "categorical_mask" (all (B, n_*); categorical is long).
        bin_pos_weight: optional (n_binary,) positive-class weights for BCE.
        weights: optional per-type scalar weights {"continuous","binary",
            "categorical"} (default 1.0 each).

    Returns:
        dict with "continuous", "binary", "categorical" sub-losses and "total".
    """
    w = {"continuous": 1.0, "binary": 1.0, "categorical": 1.0, **(weights or {})}
    device = _first_param_device(pred)
    zero = torch.zeros((), device=device)

    # ── Continuous: MSE over supervised entries ──────────────────────────────
    if pred["continuous"].shape[1] > 0:
        se = (pred["continuous"] - target["continuous"]) ** 2
        cont_loss = _masked_mean(se, target["continuous_mask"])
    else:
        cont_loss = zero

    # ── Binary: BCE-with-logits over supervised entries ──────────────────────
    if pred["binary_logits"].shape[1] > 0:
        bce = F.binary_cross_entropy_with_logits(
            pred["binary_logits"], target["binary"].to(pred["binary_logits"].dtype),
            pos_weight=bin_pos_weight, reduction="none")
        bin_loss = _masked_mean(bce, target["binary_mask"])
    else:
        bin_loss = zero

    # ── Categorical: CE per concept over supervised rows ─────────────────────
    cat_logits: List[torch.Tensor] = pred["categorical_logits"]
    if cat_logits:
        cat_terms = []
        for i, logits in enumerate(cat_logits):
            tgt = target["categorical"][:, i].long()
            ce = F.cross_entropy(logits, tgt, reduction="none")  # (B,)
            cat_terms.append(_masked_mean(ce, target["categorical_mask"][:, i]))
        cat_loss = torch.stack(cat_terms).mean()
    else:
        cat_loss = zero

    total = (w["continuous"] * cont_loss
             + w["binary"] * bin_loss
             + w["categorical"] * cat_loss)
    return {"continuous": cont_loss, "binary": bin_loss,
            "categorical": cat_loss, "total": total}


def binary_pos_weight_from_manifest(manifest: Dict[str, Any],
                                    clamp_max: float = 50.0) -> torch.Tensor:
    """Per-binary-concept BCE pos_weight = (1 - p) / p from train pos_frac.

    Rare positives (small p) get up-weighted so the heavy class imbalance in the
    binary concepts doesn't collapse them to always-negative. Clamped to avoid
    exploding weights on near-zero-frequency concepts.
    """
    names = manifest["per_type"]["binary"]["names"]
    stats = manifest["stats"]
    out = []
    for n in names:
        p = float(stats[n]["pos_frac"])
        out.append(min((1.0 - p) / p, clamp_max) if p > 0 else clamp_max)
    return torch.tensor(out, dtype=torch.float32)


def _first_param_device(pred: Dict[str, Any]) -> torch.device:
    if pred["continuous"].shape[1] > 0:
        return pred["continuous"].device
    if pred["binary_logits"].shape[1] > 0:
        return pred["binary_logits"].device
    return pred["categorical_logits"][0].device
