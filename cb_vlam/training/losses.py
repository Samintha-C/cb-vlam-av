"""Concept loss L_c for CB-VLAM-AV.

Masked, per-type supervision of the CBL outputs against ConceptStore targets:

    continuous   → MSE
    binary       → BCE-with-logits (optional per-concept pos_weight for the
                   heavy class imbalance — many binary concepts are rare events)
    categorical  → cross-entropy, per concept

Masks (True = supervise) come from ConceptStore and implement the "saturated =
absent" toggles; masked-out elements contribute nothing to the loss or its
denominator. This is the only loss needed for concept-projection training;
the task/adversarial/sparsity terms come later.
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


# Generation training losses: trajectory task loss L_t, the adversarial probe
# loss, the residual-disentangling objective, and the sparsity penalty.
#
# Disentanglement follows CB-LLM (train_CBLLM.py:163-175), NOT a Gradient-Reversal
# Layer. CB-LLM keeps it stable by gradient *confinement*, not by a small weight:
#   1. adversary_loss trains the probe to read concepts from r.detach()
#      (backward restricted to the probe's own params);
#   2. disentanglement_loss then pushes the residual toward the probe's
#      *uninformative prior* on r computed from detached backbone features
#      (backward restricted to the residual's params).
# Because the adversarial gradient never reaches the backbone or the concept
# path, it cannot corrupt the shared trunk — which is exactly what a fused GRL
# backward did (all losses diverged as lambda ramped up). See train_gen.py.
adversary_loss = concept_loss


def disentanglement_loss(
    pred: Dict[str, Any],
    target: Dict[str, Any],
    *,
    weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """Bounded "make the probe uninformative" objective, confined to the residual.

    Minimizing this drives the adversary probe toward its uninformative prior, so
    the residual is forced to carry no recoverable concept information:

        continuous   → the batch marginal mean (regression R^2 → 0)
        binary       → maximum Bernoulli entropy (sigmoid → 0.5)
        categorical  → maximum softmax entropy (uniform over classes)

    This is the mixed-type generalization of CB-LLM's single-class neg-entropy
    term (train_CBLLM.py:172). Every component is *bounded* — that boundedness is
    what makes the adversarial pressure stable; plain gradient-ascent on the
    probe's prediction loss (the GRL form) is unbounded and runs away. Masks
    follow the concept masks, so absent/saturated concepts contribute nothing.

    Args mirror ``concept_loss``: pred is the probe's per-type output, target the
    batched ConceptStore targets + masks. Returns per-type sub-losses + "total".
    """
    w = {"continuous": 1.0, "binary": 1.0, "categorical": 1.0, **(weights or {})}
    device = _first_param_device(pred)
    zero = torch.zeros((), device=device)
    eps = 1e-6

    # ── Continuous: pull the probe toward the (detached) batch marginal mean ──
    if pred["continuous"].shape[1] > 0:
        m = target["continuous_mask"].to(pred["continuous"].dtype)
        marg = (target["continuous"] * m).sum(0) / m.sum(0).clamp(min=1.0)  # (n_cont,)
        se = (pred["continuous"] - marg.detach()) ** 2
        cont = _masked_mean(se, target["continuous_mask"])
    else:
        cont = zero

    # ── Binary: minimize Bernoulli neg-entropy → sigmoid pushed to 0.5 ───────
    if pred["binary_logits"].shape[1] > 0:
        p = torch.sigmoid(pred["binary_logits"]).clamp(eps, 1.0 - eps)
        neg_ent = p * p.log() + (1.0 - p) * (1.0 - p).log()
        binl = _masked_mean(neg_ent, target["binary_mask"])
    else:
        binl = zero

    # ── Categorical: minimize softmax neg-entropy → uniform over classes ─────
    cat_logits: List[torch.Tensor] = pred["categorical_logits"]
    if cat_logits:
        terms = []
        for i, logits in enumerate(cat_logits):
            logp = F.log_softmax(logits, dim=-1)
            neg_ent = (logp.exp() * logp).sum(-1)  # (B,) — in [-log K, 0]
            terms.append(_masked_mean(neg_ent, target["categorical_mask"][:, i]))
        catl = torch.stack(terms).mean()
    else:
        catl = zero

    total = (w["continuous"] * cont + w["binary"] * binl + w["categorical"] * catl)
    return {"continuous": cont, "binary": binl, "categorical": catl, "total": total}


def trajectory_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    *,
                    mask: Optional[torch.Tensor] = None,
                    beta: float = 1.0) -> torch.Tensor:
    """Task loss L_t for the continuous-regression arm.

    Smooth-L1 (Huber) over the flat waypoint vector — robust to the occasional
    large-displacement target without the gradient blow-up of plain MSE. ADE/FDE
    are reported separately as eval metrics; this is the optimization objective.

    Args:
        pred:   (B, traj_dim) predicted flat waypoints.
        target: (B, traj_dim) ground-truth flat waypoints.
        mask:   optional (B,) or (B, traj_dim) bool — True = include. Lets a
                sample with no valid future (padding) drop out of the loss.
        beta:   Huber transition point (meters), in the waypoints' own units.
    """
    per_elem = F.smooth_l1_loss(pred, target.to(pred.dtype), beta=beta, reduction="none")
    if mask is None:
        return per_elem.mean()
    if mask.dim() == 1:
        mask = mask[:, None].expand_as(per_elem)
    return _masked_mean(per_elem, mask)


def elastic_net_penalty(weight: torch.Tensor, alpha: float = 0.99) -> torch.Tensor:
    """Elastic-net sparsity penalty on the final predictor's concept weights.

    alpha*|W|_1 + (1-alpha)*|W|_2^2, mean-reduced — ported from CB-LLM
    (generation/utils.py:37). Applied to FinalPredictor.concept_weight so each
    output dimension wires to a *sparse* set of named concepts (interpretability).
    """
    return alpha * weight.abs().mean() + (1.0 - alpha) * weight.square().mean()


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
