"""Concept Bottleneck Layer (CBL).

Projects a backbone feature vector into the interpretable concept space defined
by the mining manifest's ``per_type`` layout. Produces per-type outputs so the
concept loss can apply the right objective to each:

    continuous        (B, n_continuous)         raw values  → MSE
    binary_logits     (B, n_binary)             logits      → BCE-with-logits
    categorical_logits list[(B, n_categories_i)] per concept → cross-entropy

An optional shared hidden trunk precedes the heads; with ``hidden_dim=None`` the
heads read the backbone feature directly (a single linear projection per type —
the closest analogue to CB-LLM's linear concept layer).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ConceptBottleneckLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 layout: Dict[str, Any],
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 input_norm: bool = True):
        """
        Args:
            in_dim: Backbone feature dimension (CBVLAMBackbone.feature_dim).
            layout: manifest["per_type"] — counts/categories per concept type.
            hidden_dim: If set, a shared GELU trunk of this width precedes the
                heads. If None, heads project the backbone feature directly.
            dropout: Dropout on the trunk (ignored when hidden_dim is None).
            input_norm: If True (default), LayerNorm the backbone feature before
                the heads. The raw decoder hidden states have large magnitude,
                which makes initial head outputs (and losses) hot and training
                unstable; normalizing the input fixes that.
        """
        super().__init__()
        self.n_continuous = layout["continuous"]["n"]
        self.n_binary = layout["binary"]["n"]
        self.cat_ncats: List[int] = list(layout["categorical"]["n_categories"])

        self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()

        if hidden_dim:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            head_in = hidden_dim
        else:
            self.trunk = nn.Identity()
            head_in = in_dim

        self.cont_head = nn.Linear(head_in, self.n_continuous) if self.n_continuous else None
        self.bin_head = nn.Linear(head_in, self.n_binary) if self.n_binary else None
        self.cat_heads = nn.ModuleList([nn.Linear(head_in, k) for k in self.cat_ncats])

    def forward(self, feats: torch.Tensor) -> Dict[str, Any]:
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        # Heads run in their own (float32) dtype; cast the (possibly bf16)
        # backbone feature to match so autograd casts grads back on the way out.
        ref = self._ref_param()
        if ref is not None:
            feats = feats.to(ref.dtype)

        feats = self.input_norm(feats)
        h = self.trunk(feats)
        B = h.shape[0]
        return {
            "continuous": (self.cont_head(h) if self.cont_head is not None
                           else h.new_zeros((B, 0))),
            "binary_logits": (self.bin_head(h) if self.bin_head is not None
                              else h.new_zeros((B, 0))),
            "categorical_logits": [head(h) for head in self.cat_heads],
        }

    def _ref_param(self) -> Optional[torch.Tensor]:
        for head in (self.cont_head, self.bin_head):
            if head is not None:
                return head.weight
        for head in self.cat_heads:
            return head.weight
        return None
