"""Concept-selection criteria: the order in which concepts are intervened.

Faithful to Shin et al. (2023), "A Closer Look at the Intervention Procedure of
Concept Bottleneck Models" (§3.2). Each criterion produces a (N, C) score; the
runner intervenes highest-score-first.

  - RAND : random order (Koh et al. 2020 baseline / floor). Per-seed; the runner
           averages the curve over several seeds.
  - IMP  : importance = the model's own attribution. Our FinalPredictor is
           LINEAR, so concept c's contribution to the trajectory is exactly the
           L2 norm of its weight block ‖W[:, cols_c]‖ — a single GLOBAL ranking
           (the reportable "most important concepts"). This is Shin's CCTP with
           the linear-head simplification ``∂f_j/∂ĉ_i = W_ji`` (the optional
           per-sample ``|ĉ_i|`` scaling is available via ``scale_by_activation``).
  - LCP  : oracle — order by GT concept error ‖ĉ_c − c_c‖ (largest error first).
           Uses ground truth, so it is an UPPER BOUND, not a test-time-realistic
           strategy; it bounds how much informed selection can help.
"""

from typing import Any, Dict, List

import torch


def importance_scores(head_concept_weight: torch.Tensor,
                      catalog: List[Dict[str, Any]], n: int,
                      a_pred: torch.Tensor = None,
                      scale_by_activation: bool = False) -> torch.Tensor:
    """IMP: per-concept linear-weight norm, broadcast to (N, C).

    head_concept_weight: (output_dim, concept_dim) = FinalPredictor.concept_weight.
    If ``scale_by_activation``, multiply each concept's norm by |ĉ_c| (Shin CCTP).
    """
    device = head_concept_weight.device
    C = len(catalog)
    w = torch.zeros(C, device=device)
    for c, meta in enumerate(catalog):
        cols = head_concept_weight[:, meta["col0"]:meta["col0"] + meta["k"]]
        w[c] = cols.norm()                              # ‖W block‖_2
    scores = w.unsqueeze(0).expand(n, C).clone()        # (N, C) global
    if scale_by_activation and a_pred is not None:
        for c, meta in enumerate(catalog):
            act = a_pred[:, meta["col0"]:meta["col0"] + meta["k"]].abs().sum(-1)
            scores[:, c] = scores[:, c] * act
    return scores


def lcp_scores(a_pred: torch.Tensor, a_gt: torch.Tensor, sup: torch.Tensor,
               catalog: List[Dict[str, Any]]) -> torch.Tensor:
    """LCP oracle: per-sample GT concept error ‖ĉ_c − c_c‖_1 over the concept's columns."""
    N = a_pred.shape[0]
    scores = torch.zeros((N, len(catalog)), device=a_pred.device)
    diff = (a_pred - a_gt).abs()
    for c, meta in enumerate(catalog):
        scores[:, c] = diff[:, meta["col0"]:meta["col0"] + meta["k"]].sum(-1)
    scores = scores.masked_fill(~sup, 0.0)              # no GT → no error signal
    return scores


def random_scores(n: int, C: int, device, generator: torch.Generator) -> torch.Tensor:
    """RAND: i.i.d. uniform scores (one draw → one global-ish random order per seed)."""
    return torch.rand((n, C), device=device, generator=generator)


def rank_of(scores: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """(N, C) rank of each concept (0 = intervene first); unsupervised sort last.

    A budget-m selection is then simply ``(rank < m) & sup``.
    """
    masked = scores.masked_fill(~sup, float("-inf"))
    order = masked.argsort(dim=1, descending=True)      # best-first concept indices
    return order.argsort(dim=1)                          # position of each concept
