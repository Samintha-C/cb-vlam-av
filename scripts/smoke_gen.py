"""CPU smoke test for the generation modules (no backbone, no store).

Validates, on random features with a fake 27-concept layout, that the new
generation modules wire together and train under CB-LLM's CONFINED disentanglement
(no gradient-reversal layer):

  1. CBL.to_activations builds the normalized concept vector at the expected
     width, and activation_slices covers it exactly.
  2. UnsupervisedResidual + AdversarialDiscriminator + FinalPredictor forward
     in both output modes (regression for continuous-regression, vocab for
     autoregressive).
  3. The MAIN loss (L_c + L_t + sparsity) backprops to backbone-features, CBL,
     residual, and head.
  4. CONFINEMENT — the firewalls that replace the GRL:
       a. the adversary step (concept_loss on r.detach(), backward(inputs=adv))
          updates ONLY the adversary — residual and feats get no gradient;
       b. the disentangle step (disentanglement_loss on residual(feats.detach()),
          backward(inputs=residual)) updates ONLY the residual — feats (the
          backbone stand-in) and the CBL get no gradient.
  5. Per-token (B, T, D) features flow through unchanged (the autoregressive shape).

Pure CPU, no deps beyond torch. Exits non-zero on any failed assertion.

    python scripts/smoke_gen.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cb_vlam.models.cbl import ConceptBottleneckLayer
from cb_vlam.models.residual import UnsupervisedResidual
from cb_vlam.models.adversarial import AdversarialDiscriminator
from cb_vlam.models.final_predictor import FinalPredictor
from cb_vlam.training.losses import (
    concept_loss, adversary_loss, disentanglement_loss, trajectory_loss,
    elastic_net_penalty)


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return total ** 0.5


def _zero(*modules_and_tensors):
    for x in modules_and_tensors:
        if isinstance(x, torch.Tensor):
            x.grad = None
        else:
            for p in x.parameters():
                p.grad = None


def _fake_layout():
    return {
        "continuous":  {"names": [f"cont{i}" for i in range(8)], "n": 8},
        "binary":      {"names": [f"bin{i}" for i in range(18)], "n": 18},
        "categorical": {"names": ["traffic_density_det"], "n_categories": [3], "n": 1},
    }


def _fake_targets(B: int):
    return {
        "continuous": torch.rand(B, 8),
        "continuous_mask": torch.ones(B, 8, dtype=torch.bool),
        "binary": (torch.rand(B, 18) > 0.5).float(),
        "binary_mask": torch.ones(B, 18, dtype=torch.bool),
        "categorical": torch.randint(0, 3, (B, 1)),
        "categorical_mask": torch.ones(B, 1, dtype=torch.bool),
    }


def main() -> None:
    torch.manual_seed(0)
    B, D, R, V = 4, 256, 64, 1000   # batch, feature dim, residual dim, fake vocab
    H = 6 * 2                       # 6 waypoints × (x, y)
    layout = _fake_layout()

    cbl = ConceptBottleneckLayer(in_dim=D, layout=layout)
    residual = UnsupervisedResidual(in_dim=D, residual_dim=R)
    adversary = AdversarialDiscriminator(residual_dim=R, layout=layout)
    head_reg = FinalPredictor(cbl.activation_dim, R, output_dim=H, mode="regression")
    head_vocab = FinalPredictor(cbl.activation_dim, R, output_dim=V, mode="vocab")

    # ── 1. Activation vector width + slice coverage ───────────────────────────
    assert cbl.activation_dim == 8 + 18 + 3, cbl.activation_dim
    slices = cbl.activation_slices
    assert len(slices) == 8 + 18 + 1, len(slices)            # 1 categorical name
    covered = sorted(i for s in slices.values() for i in range(s.start, s.stop))
    assert covered == list(range(cbl.activation_dim)), "activation_slices must tile [0, dim)"
    print(f"CBL.activation_dim = {cbl.activation_dim}, slices tile it exactly  ✓")

    # ── 2. Forward (single-tap) in both head modes ────────────────────────────
    feats = torch.randn(B, D, requires_grad=True)            # stands in for backbone h
    cbl_out = cbl(feats)
    cvec = cbl.to_activations(cbl_out)
    r = residual(feats)
    traj = head_reg(cvec, r)
    vocab = head_vocab(cvec, r)
    assert cvec.shape == (B, cbl.activation_dim) and r.shape == (B, R)
    assert traj.shape == (B, H) and vocab.shape == (B, V)
    print(f"forward: cvec{tuple(cvec.shape)} r{tuple(r.shape)} "
          f"-> traj{tuple(traj.shape)} vocab{tuple(vocab.shape)}  ✓")

    targets = _fake_targets(B)

    # ── 3. MAIN loss reaches backbone-feats, CBL, residual, head ──────────────
    Lc = concept_loss(cbl_out, targets)["total"]
    Lt = trajectory_loss(traj, torch.randn(B, H))
    Lreg = elastic_net_penalty(head_reg.concept_weight)
    Lmain = Lc + Lt + 1e-3 * Lreg
    Lmain.backward()
    for name, mod in [("cbl", cbl), ("residual", residual), ("head_reg", head_reg)]:
        assert _grad_norm(mod.parameters()) > 0, f"{name} got no gradient from main"
    assert feats.grad is not None and feats.grad.norm() > 0, "backbone feats got no main grad"
    print("main L_c + L_t + sparsity backprops to feats, CBL, residual, head  ✓")

    # ── 4a. ADVERSARY step is confined to the adversary ───────────────────────
    _zero(cbl, residual, adversary, head_reg, feats)
    adv_out = adversary(r.detach())
    Ladv = adversary_loss(adv_out, targets)["total"]
    Ladv.backward(inputs=list(adversary.parameters()))
    assert _grad_norm(adversary.parameters()) > 0, "adversary got no gradient"
    assert _grad_norm(residual.parameters()) == 0, "adversary step leaked into residual"
    assert feats.grad is None, "adversary step leaked into backbone feats"
    print("adversary step updates ONLY the adversary (residual + feats untouched)  ✓")

    # ── 4b. DISENTANGLE step is confined to the residual ──────────────────────
    _zero(cbl, residual, adversary, head_reg, feats)
    dis_out = adversary(residual(feats.detach()))
    Ldis = disentanglement_loss(dis_out, targets)["total"]
    assert torch.isfinite(Ldis), Ldis
    Ldis.backward(inputs=list(residual.parameters()))
    assert _grad_norm(residual.parameters()) > 0, "residual got no disentangle gradient"
    assert _grad_norm(adversary.parameters()) == 0, "disentangle step leaked into adversary"
    assert _grad_norm(cbl.parameters()) == 0, "disentangle step leaked into CBL"
    assert feats.grad is None, "disentangle step leaked into backbone feats"
    print("disentangle step updates ONLY the residual (feats + CBL + adversary untouched)  ✓")

    # ── 5. Per-token shape (autoregressive) ───────────────────────────────────
    T = 5
    feats_tok = torch.randn(B, T, D)
    out_tok = cbl(feats_tok.reshape(B * T, D))
    cvec_tok = cbl.to_activations(out_tok).reshape(B, T, -1)
    r_tok = residual(feats_tok.reshape(B * T, D)).reshape(B, T, R)
    logits_tok = head_vocab(cvec_tok, r_tok)
    assert logits_tok.shape == (B, T, V), logits_tok.shape
    print(f"per-token (autoregressive): (B,T,D) -> vocab logits {tuple(logits_tok.shape)}  ✓")

    # ── 6. Trajectory parsing + masked L_t ────────────────────────────────────
    from cb_vlam.training.dataset import parse_trajectory
    asst = ("<PLANNING>... formatted as [x, y]: [3.72, -0.29], [7.52, -0.73], "
            "[11.5, -1.26], [15.61, -1.9], [19.85, -2.58], [24.16, -3.28]</PLANNING>")
    tj, msk = parse_trajectory(asst, horizon=6)
    assert tj.shape == (12,) and msk.all(), (tj.shape, msk)
    assert abs(tj[0] - 3.72) < 1e-6 and abs(tj[-1] - (-3.28)) < 1e-6, tj
    tj2, msk2 = parse_trajectory("formatted as [x, y]: [1.0, 2.0]", horizon=6)
    assert msk2[:2].all() and not msk2[2:].any(), msk2
    lt = trajectory_loss(torch.zeros(1, 12), torch.from_numpy(tj)[None],
                         mask=torch.from_numpy(msk)[None])
    assert torch.isfinite(lt) and lt > 0, lt
    print(f"trajectory: parse→6 wp, masked smooth-L1={lt.item():.3f}  ✓")

    print("\nSMOKE PASSED: generation modules + CB-LLM-confined disentanglement "
          "wire together and train.")


if __name__ == "__main__":
    main()
