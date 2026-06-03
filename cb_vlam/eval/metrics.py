"""Per-type concept-projection metrics for CB-VLAM-AV.

The concept-projection question is "can we project accurately into concept space?", so the
metrics are per concept type, computed only over supervised (unmasked) entries:

    continuous   MAE, RMSE, R^2
    binary       AUROC (skipped if a concept has one class in the eval set),
                 F1@0.5, positive rate
    categorical  accuracy, macro-F1

``evaluate`` runs the backbone+CBL over a loader and returns a nested dict;
``summarize`` reduces it to the handful of headline numbers to watch.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cb_vlam.training.losses import concept_loss

try:
    from sklearn.metrics import roc_auc_score, f1_score
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


@torch.no_grad()
def evaluate(backbone, cbl, loader, device: str,
             manifest: Dict[str, Any], max_batches: Optional[int] = None,
             bin_pos_weight=None) -> Dict[str, Any]:
    backbone.model.eval()
    cbl.eval()
    layout = manifest["per_type"]

    cont_p, cont_t, cont_m = [], [], []
    bin_p, bin_t, bin_m = [], [], []
    cat_p, cat_t, cat_m = [], [], []
    loss_sum, loss_n = 0.0, 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        feats = torch.stack(
            [backbone(img, prm) for img, prm in zip(batch["images"], batch["prompts"])],
            dim=0)
        pred = cbl(feats)
        tgt = batch["targets"]

        tgt_dev = {k: v.to(device) for k, v in tgt.items()}
        bsz = feats.shape[0]
        loss_sum += float(concept_loss(pred, tgt_dev, bin_pos_weight=bin_pos_weight)["total"]) * bsz
        loss_n += bsz

        if pred["continuous"].shape[1]:
            cont_p.append(pred["continuous"].float().cpu().numpy())
            cont_t.append(tgt["continuous"].numpy())
            cont_m.append(tgt["continuous_mask"].numpy())
        if pred["binary_logits"].shape[1]:
            bin_p.append(torch.sigmoid(pred["binary_logits"]).float().cpu().numpy())
            bin_t.append(tgt["binary"].numpy())
            bin_m.append(tgt["binary_mask"].numpy())
        if pred["categorical_logits"]:
            cat_pred = np.stack(
                [lg.argmax(-1).cpu().numpy() for lg in pred["categorical_logits"]], axis=1)
            cat_p.append(cat_pred)
            cat_t.append(tgt["categorical"].numpy())
            cat_m.append(tgt["categorical_mask"].numpy())

    out: Dict[str, Any] = {}
    if loss_n:
        out["loss"] = loss_sum / loss_n
    if cont_p:
        out["continuous"] = _continuous_metrics(
            np.concatenate(cont_p), np.concatenate(cont_t), np.concatenate(cont_m),
            layout["continuous"]["names"])
    if bin_p:
        out["binary"] = _binary_metrics(
            np.concatenate(bin_p), np.concatenate(bin_t), np.concatenate(bin_m),
            layout["binary"]["names"])
    if cat_p:
        out["categorical"] = _categorical_metrics(
            np.concatenate(cat_p), np.concatenate(cat_t), np.concatenate(cat_m),
            layout["categorical"]["names"], layout["categorical"]["n_categories"])
    return out


# Target-std floor (on the normalized concept scale) below which a continuous
# concept is treated as near-constant. R² there is unreliable: the variance
# denominator is tiny, so near-mean predictions (small MAE) still yield large
# negative R². For such concepts we (a) flag low_variance, (b) report a
# variance-floored R² that can't blow up, and (c) exclude them from the headline
# macro R² so it reflects only concepts where R² is meaningful. MAE is primary.
STD_FLOOR = 0.1


def _continuous_metrics(pred, tgt, mask, names) -> Dict[str, Any]:
    per = {}
    maes, r2_reliable, r2f_all = [], [], []
    var_floor = STD_FLOOR ** 2
    n_low_var = 0
    for j, n in enumerate(names):
        m = mask[:, j].astype(bool)
        if m.sum() == 0:
            per[n] = {"mae": None, "rmse": None, "r2": None, "r2_floored": None,
                      "std": None, "low_variance": None, "n": 0}
            continue
        p, t = pred[m, j], tgt[m, j]
        mae = float(np.mean(np.abs(p - t)))
        mse = float(np.mean((p - t) ** 2))
        var = float(np.var(t))
        std = float(np.sqrt(var))
        low_var = std < STD_FLOOR
        r2 = float(1.0 - mse / var) if var > 0 else None
        # Variance-floored R²: denominator can't drop below the floor, so a
        # near-constant target can't produce a meaningless huge-negative R².
        r2_floored = float(1.0 - mse / max(var, var_floor))
        per[n] = {"mae": mae, "rmse": float(np.sqrt(mse)), "r2": r2,
                  "r2_floored": r2_floored, "std": std,
                  "low_variance": bool(low_var), "n": int(m.sum())}
        maes.append(mae)
        r2f_all.append(r2_floored)
        if low_var:
            n_low_var += 1
        elif r2 is not None:
            r2_reliable.append(r2)
    return {"per_concept": per,
            "macro_mae": float(np.mean(maes)) if maes else None,
            # Headline R² over adequate-variance concepts only (others flagged).
            "macro_r2": float(np.mean(r2_reliable)) if r2_reliable else None,
            "macro_r2_floored": float(np.mean(r2f_all)) if r2f_all else None,
            "n_low_variance": n_low_var}


def _binary_metrics(prob, tgt, mask, names) -> Dict[str, Any]:
    per = {}
    aurocs, f1s = [], []
    for j, n in enumerate(names):
        m = mask[:, j].astype(bool)
        if m.sum() == 0:
            per[n] = {"auroc": None, "f1": None, "pos_rate": None, "n": 0}
            continue
        p, t = prob[m, j], tgt[m, j].astype(int)
        pos_rate = float(t.mean())
        auroc = None
        if _HAVE_SK and 0 < t.sum() < len(t):  # needs both classes
            auroc = float(roc_auc_score(t, p))
        pred_lbl = (p >= 0.5).astype(int)
        f1 = (float(f1_score(t, pred_lbl, zero_division=0)) if _HAVE_SK
              else _f1_manual(t, pred_lbl))
        per[n] = {"auroc": auroc, "f1": f1, "pos_rate": pos_rate, "n": int(m.sum())}
        if auroc is not None:
            aurocs.append(auroc)
        f1s.append(f1)
    return {"per_concept": per,
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_f1": float(np.mean(f1s)) if f1s else None}


def _categorical_metrics(pred, tgt, mask, names, ncats) -> Dict[str, Any]:
    per = {}
    accs, f1s = [], []
    for j, n in enumerate(names):
        m = mask[:, j].astype(bool)
        if m.sum() == 0:
            per[n] = {"accuracy": None, "macro_f1": None, "n": 0}
            continue
        p, t = pred[m, j].astype(int), tgt[m, j].astype(int)
        acc = float((p == t).mean())
        if _HAVE_SK:
            f1 = float(f1_score(t, p, average="macro",
                                labels=list(range(ncats[j])), zero_division=0))
        else:
            f1 = float(np.mean([_f1_manual((t == c).astype(int), (p == c).astype(int))
                                for c in range(ncats[j])]))
        per[n] = {"accuracy": acc, "macro_f1": f1, "n": int(m.sum())}
        accs.append(acc)
        f1s.append(f1)
    return {"per_concept": per,
            "macro_accuracy": float(np.mean(accs)) if accs else None,
            "macro_f1": float(np.mean(f1s)) if f1s else None}


def _f1_manual(t: np.ndarray, p: np.ndarray) -> float:
    tp = float(((p == 1) & (t == 1)).sum())
    fp = float(((p == 1) & (t == 0)).sum())
    fn = float(((p == 0) & (t == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def summarize(metrics: Dict[str, Any]) -> str:
    """One-line headline of the key projection-quality numbers."""
    parts = []
    if "loss" in metrics:
        parts.append(f"loss={_fmt(metrics['loss'])}")
    if "continuous" in metrics:
        c = metrics["continuous"]
        nlv = c.get("n_low_variance", 0)
        lv = f" [-{nlv} low-var]" if nlv else ""
        parts.append(f"cont MAE={_fmt(c['macro_mae'])} "
                     f"R2={_fmt(c.get('macro_r2'))}{lv}")
    if "binary" in metrics:
        parts.append(f"bin AUROC={_fmt(metrics['binary']['macro_auroc'])} "
                     f"F1={_fmt(metrics['binary']['macro_f1'])}")
    if "categorical" in metrics:
        parts.append(f"cat acc={_fmt(metrics['categorical']['macro_accuracy'])} "
                     f"F1={_fmt(metrics['categorical']['macro_f1'])}")
    return "  |  ".join(parts)


def _fmt(x: Optional[float]) -> str:
    return f"{x:.3f}" if isinstance(x, float) else "n/a"
