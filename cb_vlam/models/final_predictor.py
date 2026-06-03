"""Final predictor: the sparse linear layer at the end of the bottleneck.

A single ``Linear([concept_activations ⊕ residual]) → output`` — kept linear (no
hidden trunk) on purpose, so the concept→output weights are directly readable
and can be L1-sparsified (the interpretability claim: each output dimension is a
sparse linear combination of named concepts + residual). This mirrors CB-LLM's
``fc`` (generation/modules.py:82).

Two output modes, identical module, only the output dimension differs:
  - "regression"  → flat 6×2 waypoints (continuous-regression)
  - "vocab"       → next-token logits over the tokenizer vocab (autoregressive,
                    the CB-LLM-faithful per-token bottleneck)

``concept_weight`` exposes the columns acting on the concept block so the
training loop can apply the elastic-net sparsity penalty to them alone (the
residual columns are left dense).
"""

from typing import Optional

import torch
import torch.nn as nn


class FinalPredictor(nn.Module):
    def __init__(self,
                 concept_dim: int,
                 residual_dim: int,
                 output_dim: int,
                 mode: str = "regression",
                 bias: bool = True):
        """
        Args:
            concept_dim: Width of the concept-activation vector (CBL.activation_dim).
            residual_dim: Width of the residual vector r.
            output_dim: For "regression", horizon*2 (e.g. 12 for 6 waypoints);
                for "vocab", the tokenizer vocab size.
            mode: "regression" | "vocab" — bookkeeping only; the layer is identical.
            bias: Include a bias term (CB-LLM's fc has one).
        """
        super().__init__()
        if mode not in ("regression", "vocab"):
            raise ValueError(f"mode must be 'regression' or 'vocab', got {mode!r}")
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim
        self.output_dim = output_dim
        self.mode = mode
        self.fc = nn.Linear(concept_dim + residual_dim, output_dim, bias=bias)

    def forward(self, concept_vec: torch.Tensor,
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Concatenate [concept_activations ⊕ residual] and project.

        ``residual`` may be None to ablate the residual pathway (concepts-only
        predictor), in which case the residual columns are simply not fed — pass
        a zeros tensor of the right width to keep the layer shape fixed.
        """
        if residual is None:
            residual = concept_vec.new_zeros((*concept_vec.shape[:-1], self.residual_dim))
        x = torch.cat([concept_vec, residual], dim=-1).to(self.fc.weight.dtype)
        return self.fc(x)

    @property
    def concept_weight(self) -> torch.Tensor:
        """The (output_dim, concept_dim) weight block acting on the concepts."""
        return self.fc.weight[:, :self.concept_dim]
