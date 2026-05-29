"""Read-side accessor for a mined concept dataset + manifest.

ConceptStore is the single interface the training pipeline uses to fetch
supervision targets. It joins on the nuScenes ``sample_token`` shared with
Impromptu VLA's input loader, so the model-input side (images + prompt) is
never duplicated here — this store holds concept labels only.

For each ``sample_token`` it returns per-type targets and per-type masks:

    continuous      float[n_continuous]   normalized concept values
    continuous_mask bool[n_continuous]    True = include in L_c
    binary          float[n_binary]       0.0 / 1.0 targets
    binary_mask     bool[n_binary]
    categorical     int[n_categorical]    class indices
    categorical_mask bool[n_categorical]

Masks default to all-True (supervise everything). The manifest's ``mask_rules``
let specific "saturated = absent" concepts be excluded from the concept loss
when their toggle flag is enabled — e.g. enabling ``mask_lead_when_absent``
masks lead-vehicle concepts on frames where ``lead_vehicle_present == 0``. By
default no toggles are enabled, matching the project decision to supervise the
saturated sentinel as a real signal.

The numpy core (``get_numpy``) carries no torch dependency; ``get`` wraps it in
torch tensors on demand.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import numpy as np


class ConceptStore:
    def __init__(self,
                 dataset_path: Path,
                 enabled_masks: Optional[Set[str]] = None,
                 data_subdir: str = "data",
                 schema_hash: Optional[str] = None):
        """
        Args:
            dataset_path: Directory containing ``data/`` (HF Dataset) and
                ``manifest.json`` (as produced by build_manifest.py).
            enabled_masks: Set of mask toggle flags to activate (e.g.
                {"mask_lead_when_absent"}). Concepts whose mask_rule toggle is
                in this set get masked when their condition holds. Default: none
                enabled → every concept supervised.
            data_subdir: Subdirectory under dataset_path holding the HF Dataset.
            schema_hash: If given, assert the manifest's schema_hash matches —
                guards against pairing a model head layout with a dataset mined
                under a different schema.
        """
        from datasets import load_from_disk

        dataset_path = Path(dataset_path)
        self.manifest: Dict[str, Any] = json.loads(
            (dataset_path / "manifest.json").read_text())
        if schema_hash is not None and self.manifest["schema_hash"] != schema_hash:
            raise ValueError(
                f"schema_hash mismatch: store has {self.manifest['schema_hash']}, "
                f"caller expected {schema_hash}. Dataset and model head are out of sync.")

        self.ds = load_from_disk(str(dataset_path / data_subdir))
        self._row_by_token: Dict[str, int] = {
            tok: i for i, tok in enumerate(self.ds["sample_token"])}

        self.concept_order: List[str] = self.manifest["concept_order"]
        pt = self.manifest["per_type"]
        self.cont_names: List[str] = pt["continuous"]["names"]
        self.bin_names: List[str] = pt["binary"]["names"]
        self.cat_names: List[str] = pt["categorical"]["names"]
        self.cat_ncats: List[int] = pt["categorical"]["n_categories"]

        self.enabled_masks: Set[str] = set(enabled_masks or set())
        self.mask_rules: Dict[str, Any] = self.manifest.get("mask_rules", {})

    # ── dimensions ───────────────────────────────────────────────────────────
    @property
    def n_continuous(self) -> int: return len(self.cont_names)
    @property
    def n_binary(self) -> int: return len(self.bin_names)
    @property
    def n_categorical(self) -> int: return len(self.cat_names)

    # ── splits ─────────────────────────────────────────────────────────────--
    def split(self, name: str) -> List[str]:
        """Return the list of sample_tokens in split ``name`` (train/val/test).

        Filtered to tokens actually present in the loaded dataset.
        """
        toks = self.manifest["splits"][name]
        return [t for t in toks if t in self._row_by_token]

    # ── per-sample access ─────────────────────────────────────────────────────
    def _mask_for(self, name: str, concepts: Dict[str, float]) -> bool:
        """Return True if concept ``name`` should be supervised for this sample."""
        rule = self.mask_rules.get(name)
        if rule is None or rule["toggle"] not in self.enabled_masks:
            return True  # supervise
        when = rule["when"]
        ref = float(concepts[when["concept"]])
        if "equals" in when:
            cond = ref == when["equals"]
        elif "gte" in when:
            cond = ref >= when["gte"]
        else:
            raise ValueError(f"Unsupported mask predicate for {name!r}: {when}")
        return not cond  # condition holds → mask out (False); else supervise

    def get_numpy(self, sample_token: str) -> Dict[str, np.ndarray]:
        """Per-type targets + masks for one sample, as numpy arrays."""
        row = self.ds[self._row_by_token[sample_token]]
        cd = row["concepts"]

        cont = np.array([float(cd[n]) for n in self.cont_names], dtype=np.float32)
        binr = np.array([float(cd[n]) for n in self.bin_names], dtype=np.float32)
        catg = np.array([int(round(float(cd[n]))) for n in self.cat_names], dtype=np.int64)

        cont_m = np.array([self._mask_for(n, cd) for n in self.cont_names], dtype=bool)
        bin_m = np.array([self._mask_for(n, cd) for n in self.bin_names], dtype=bool)
        cat_m = np.array([self._mask_for(n, cd) for n in self.cat_names], dtype=bool)

        return {
            "sample_token": sample_token,
            "continuous": cont, "continuous_mask": cont_m,
            "binary": binr, "binary_mask": bin_m,
            "categorical": catg, "categorical_mask": cat_m,
        }

    def get(self, sample_token: str) -> Dict[str, Any]:
        """Per-type targets + masks for one sample, as torch tensors."""
        import torch
        d = self.get_numpy(sample_token)
        out: Dict[str, Any] = {"sample_token": d["sample_token"]}
        for k, v in d.items():
            if k == "sample_token":
                continue
            out[k] = torch.from_numpy(v)
        return out

    def __getitem__(self, sample_token: str) -> Dict[str, Any]:
        return self.get(sample_token)

    def __contains__(self, sample_token: str) -> bool:
        return sample_token in self._row_by_token
