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

        # Concept names (canonical order, per type) — used to build the
        # name→slot map for the concept-activation vector that the downstream
        # FinalPredictor reads and that intervention overwrites.
        self.cont_names: List[str] = list(layout["continuous"]["names"])
        self.bin_names: List[str] = list(layout["binary"]["names"])
        self.cat_names: List[str] = list(layout["categorical"]["names"])

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

    # ── Concept-activation vector ────────────────────────────────────────────
    # The downstream FinalPredictor reads a single flat vector of *normalized*
    # concept activations (not raw logits), so every slot has a known meaning
    # and range — which is what makes intervention/steerability well-defined:
    #   continuous  → the predicted value as-is (already normalized at mine time)
    #   binary      → sigmoid(logit)            ∈ [0, 1]
    #   categorical → softmax(logits)           (n_categories slots, sum to 1)
    # Layout order is continuous, then binary, then each categorical block.

    @property
    def activation_dim(self) -> int:
        return self.n_continuous + self.n_binary + sum(self.cat_ncats)

    @property
    def activation_slices(self) -> Dict[str, slice]:
        """Map concept name → its slice in the activation vector (for intervention)."""
        slices: Dict[str, slice] = {}
        i = 0
        for n in self.cont_names:
            slices[n] = slice(i, i + 1); i += 1
        for n in self.bin_names:
            slices[n] = slice(i, i + 1); i += 1
        for n, k in zip(self.cat_names, self.cat_ncats):
            slices[n] = slice(i, i + k); i += k
        return slices

    def to_activations(self, out: Dict[str, Any]) -> torch.Tensor:
        """Assemble the CBL per-type output into the normalized activation vector.

        Accepts any leading batch shape (e.g. (B, D) or (B, T, D) outputs) and
        concatenates along the last dimension.
        """
        parts: List[torch.Tensor] = []
        if self.n_continuous:
            parts.append(out["continuous"])
        if self.n_binary:
            parts.append(torch.sigmoid(out["binary_logits"]))
        for logits in out["categorical_logits"]:
            parts.append(torch.softmax(logits, dim=-1))
        return torch.cat(parts, dim=-1)

    def gt_activations(self, targets: Dict[str, torch.Tensor]):
        """Ground-truth concept-activation vector + per-column supervised mask.

        Same layout/space as ``to_activations`` (continuous value, binary ∈{0,1},
        categorical one-hot), built from the dataset targets — for TEACHER-FORCING
        the FinalPredictor on ground truth (Koh-2020 "independent" training, which
        makes test-time concept intervention in-distribution for the head).

        Returns (gt_vec, sup_mask), both (B, activation_dim); ``sup_mask`` is True
        on columns whose concept has GT for that sample.
        """
        import torch.nn.functional as F
        ref = self._ref_param()
        dtype = ref.dtype if ref is not None else torch.float32
        device = (ref.device if ref is not None
                  else next(t.device for t in targets.values()))
        gt_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []
        if self.n_continuous:
            gt_parts.append(targets["continuous"].to(device, dtype))
            mask_parts.append(targets["continuous_mask"].to(device, dtype))
        if self.n_binary:
            gt_parts.append(targets["binary"].to(device, dtype))
            mask_parts.append(targets["binary_mask"].to(device, dtype))
        for ti, k in enumerate(self.cat_ncats):
            gt_parts.append(F.one_hot(targets["categorical"][:, ti].long().to(device),
                                      num_classes=k).to(dtype))
            m = targets["categorical_mask"][:, ti].to(device, dtype)
            mask_parts.append(m.unsqueeze(-1).expand(-1, k))
        return torch.cat(gt_parts, dim=-1), torch.cat(mask_parts, dim=-1) > 0.5
