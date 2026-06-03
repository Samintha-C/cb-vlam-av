"""Unsupervised residual pathway for CB-VLAM-AV.

Parallel to the CBL: projects the same backbone feature into a free,
*unsupervised* residual vector ``r`` that carries whatever the 27 concepts do
not. The FinalPredictor consumes ``[concept_activations ⊕ r]``, so ``r`` lets
the trajectory head recover task-relevant information the bottleneck misses —
while the AdversarialDiscriminator (see models/adversarial.py) is trained to
keep ``r`` *concept-uninformative*, so the concepts stay the interpretable,
steerable handle on the output.

This mirrors CB-LLM's ``unsup`` linear layer (generation/modules.py:81), but is
kept modest-width by default: with only ~29 concept-activation slots, a very
wide residual would dominate the FinalPredictor and hollow out steerability.
Bump ``residual_dim`` if trajectory quality needs more residual capacity.
"""

from typing import Optional

import torch
import torch.nn as nn


class UnsupervisedResidual(nn.Module):
    def __init__(self,
                 in_dim: int,
                 residual_dim: int = 128,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 input_norm: bool = True):
        """
        Args:
            in_dim: Backbone feature dimension (CBVLAMBackbone.feature_dim) — the
                same tap the CBL reads.
            residual_dim: Width of the residual vector r. Default 128 (kept small
                relative to the concept-activation vector so concepts are not
                drowned out in the FinalPredictor — see module docstring).
            hidden_dim: Optional GELU trunk width before the projection. None
                (default) = a single linear projection, matching CB-LLM's unsup.
            dropout: Dropout on the trunk (ignored when hidden_dim is None).
            input_norm: LayerNorm the backbone feature first (same rationale as
                the CBL: raw decoder hidden states have large magnitude).
        """
        super().__init__()
        self.residual_dim = residual_dim
        self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        if hidden_dim:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            proj_in = hidden_dim
        else:
            self.trunk = nn.Identity()
            proj_in = in_dim
        self.proj = nn.Linear(proj_in, residual_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        feats = feats.to(self.proj.weight.dtype)
        return self.proj(self.trunk(self.input_norm(feats)))
