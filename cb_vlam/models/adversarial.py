"""Adversarial probe for disentangling the unsupervised residual.

CB-LLM keeps the concept information in the *concepts* (not the residual) so that
intervening on a concept actually steers generation. It does this WITHOUT a
gradient-reversal layer: it trains a probe to predict the class from the residual,
then trains the residual to push that probe toward maximum entropy
(generation/train_CBLLM.py:163-175). The stability comes from *confinement* —
each adversarial backward is restricted (detached features + ``inputs=``) so it
never touches the backbone or the concept path.

This module is therefore a plain probe: it predicts the concept *targets* from
the residual ``r`` and outputs the same per-type dict shape as the CBL, so the
existing ``concept_loss`` scores it directly. The minimax is orchestrated in the
training loop, not here (see cb_vlam/training/train_gen.py):

  - adversary step: ``concept_loss(adversary(r.detach()), targets)`` with the
    backward restricted to this module's params — learns to read concepts from r;
  - disentangle step: ``disentanglement_loss(adversary(residual(feats.detach())),
    targets)`` with the backward restricted to the residual's params — pushes r
    to make this probe uninformative.

The probe deliberately has a hidden trunk (it should be a *strong* probe — a weak
probe gives false confidence that r is disentangled).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn


class AdversarialDiscriminator(nn.Module):
    """Predicts the concept targets from the residual r (a plain, strong probe).

    Outputs the same per-type dict shape as the CBL, so ``concept_loss`` (probe
    training) and ``disentanglement_loss`` (residual disentangling) both consume
    it directly. No gradient reversal — the training loop confines each backward.
    """

    def __init__(self,
                 residual_dim: int,
                 layout: Dict[str, Any],
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_continuous = layout["continuous"]["n"]
        self.n_binary = layout["binary"]["n"]
        self.cat_ncats: List[int] = list(layout["categorical"]["n_categories"])

        self.trunk = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cont_head = nn.Linear(hidden_dim, self.n_continuous) if self.n_continuous else None
        self.bin_head = nn.Linear(hidden_dim, self.n_binary) if self.n_binary else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in self.cat_ncats])

    def forward(self, r: torch.Tensor) -> Dict[str, Any]:
        h = self.trunk(r.to(self.trunk[0].weight.dtype))
        B = h.shape[0]
        return {
            "continuous": (self.cont_head(h) if self.cont_head is not None
                           else h.new_zeros((B, 0))),
            "binary_logits": (self.bin_head(h) if self.bin_head is not None
                              else h.new_zeros((B, 0))),
            "categorical_logits": [head(h) for head in self.cat_heads],
        }
