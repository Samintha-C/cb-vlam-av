"""Adversarial disentanglement for the unsupervised residual.

CB-LLM keeps the class information in the *concepts* (not the residual) so that
intervening on a concept actually steers generation — they do this by training a
classifier to predict the class from the residual, then training the residual to
*maximize that classifier's entropy* (generation/train_CBLLM.py:163-175). Their
two-step alternation is specific to a single categorical class.

Our concept vector is mixed-type (continuous + binary + categorical), so we use
the equivalent single-pass formulation: a Gradient-Reversal Layer (Ganin &
Lempitsky, DANN). The adversary tries to predict the concept *targets* from the
residual ``r`` (it minimizes the standard per-type concept loss); the GRL flips
the sign of the gradient flowing back into ``r`` (and the backbone), so the
residual is simultaneously pushed to make the concepts *un*predictable. One loss
term, one backward — the adversary's own weights get normal gradients, only its
input ``r`` gets the reversed gradient.

The adversary deliberately has a hidden trunk (it should be a *strong* probe — a
weak adversary gives false confidence that r is disentangled).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Identity forward; negates (and scales by lambda_) the gradient backward."""
    return _GradientReversal.apply(x, lambda_)


class AdversarialDiscriminator(nn.Module):
    """Predicts the concept targets from the residual r (through a GRL).

    Outputs the same per-type dict shape as the CBL, so the existing
    ``concept_loss`` scores it directly — train the adversary to minimize that
    loss while the reversed gradient trains the residual to maximize it.
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

    def forward(self, r: torch.Tensor, lambda_: float = 1.0) -> Dict[str, Any]:
        r = grad_reverse(r.to(self.trunk[0].weight.dtype), lambda_)
        h = self.trunk(r)
        B = h.shape[0]
        return {
            "continuous": (self.cont_head(h) if self.cont_head is not None
                           else h.new_zeros((B, 0))),
            "binary_logits": (self.bin_head(h) if self.bin_head is not None
                              else h.new_zeros((B, 0))),
            "categorical_logits": [head(h) for head in self.cat_heads],
        }
