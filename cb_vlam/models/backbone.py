"""LoRA-adapted Impromptu VLA backbone with a differentiable feature tap.

This is the JOINT-training backbone for CB-VLAM-AV: unlike
``cb_vlam.backbones.ImpromptuVLABackbone.extract`` (which runs under
``torch.inference_mode`` and detaches features to numpy for an offline cache),
this module keeps the forward differentiable so LoRA adapters receive gradients
from the concept loss.

The backbone (Qwen2.5-VL) base weights are frozen; only LoRA adapters train.
The forward returns a single feature vector per (image, prompt) — the
concatenation of one or more *taps* into the model's hidden states:

    endprompt_final   last layer,        last prompt token (before <PLANNING>)
    endprompt_penult  penultimate layer, same position
    afterplan_final   last layer,        the appended <PLANNING> token

The tap set is configurable (``feature_taps``); the default is
``("endprompt_final",)`` — the representation right after the model has ingested
image + prompt, which is the cleanest "scene understanding" signal for concept
projection. All three are exposed so they can be swapped/compared in experiments.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image


# tap name -> (hidden_states index from the end, position key)
_TAP_SPECS: dict = {
    "endprompt_final":  (-1, "endprompt"),
    "endprompt_penult": (-2, "endprompt"),
    "afterplan_final":  (-1, "afterplan"),
}

# Suffix-matched module names LoRA adapts. These match both the LLM decoder and
# the vision tower projections in Qwen2.5-VL, so visual + language pathways both
# adapt — desirable for grounding visual concepts. Narrow this if you want to
# adapt the language model only.
_DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class CBVLAMBackbone(nn.Module):
    PLANNING_TAG = "<PLANNING>"

    def __init__(self,
                 checkpoint_path: str,
                 feature_taps: Sequence[str] = ("endprompt_final",),
                 processor_path: str = None,
                 dtype: str = "bf16",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Sequence[str] = None,
                 gradient_checkpointing: bool = True,
                 device: str = "cuda"):
        super().__init__()
        unknown = set(feature_taps) - set(_TAP_SPECS)
        if unknown:
            raise ValueError(
                f"Unknown feature_taps {sorted(unknown)}. Valid: {sorted(_TAP_SPECS)}.")
        if not feature_taps:
            raise ValueError("feature_taps must be non-empty.")
        self.feature_taps: Tuple[str, ...] = tuple(feature_taps)
        self.device = device

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import LoraConfig, get_peft_model

        torch_dtype = _DTYPES[dtype]
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(checkpoint_path), torch_dtype=torch_dtype, device_map=device,
        )
        # Qwen2.5-VL nests the LLM params under config.text_config; the tapped
        # decoder hidden states have that text hidden size (3584 for 7B).
        text_cfg = getattr(base.config, "text_config", None)
        self.hidden_size = (getattr(text_cfg, "hidden_size", None)
                            or getattr(base.config, "hidden_size", None))
        if self.hidden_size is None:
            raise RuntimeError("Could not resolve hidden_size from the model config.")

        # Joint training: cache must be off, and (with a frozen base) the input
        # embeddings must emit grad so gradient checkpointing reconnects the
        # graph to the LoRA adapters. Set use_cache on both the top-level and the
        # nested text config.
        base.config.use_cache = False
        if text_cfg is not None:
            text_cfg.use_cache = False
        if gradient_checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules or _DEFAULT_LORA_TARGETS),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.processor = AutoProcessor.from_pretrained(processor_path or str(checkpoint_path))

    @property
    def feature_dim(self) -> int:
        """Dimension of the vector returned by ``forward`` (taps concatenated)."""
        return self.hidden_size * len(self.feature_taps)

    def trainable_parameters(self):
        """Yield the LoRA adapter parameters (the only trainable backbone params)."""
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def _build_inputs(self, image: Image.Image, user_prompt: str):
        """Replicate ImpromptuVLABackbone's prompt construction + tap positions."""
        text = user_prompt.replace("<image>", "").strip()
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + self.PLANNING_TAG

        # First call (no grad needed) measures the prompt length after image-token
        # expansion; the second builds the actual model input.
        prompt_len = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt").input_ids.shape[1]
        inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt").to(self.device)
        total_len = inputs.input_ids.shape[1]
        return inputs, {"endprompt": prompt_len - 1, "afterplan": total_len - 1}

    def forward(self, image: Image.Image, user_prompt: str) -> torch.Tensor:
        """Differentiable forward for one sample → (feature_dim,) tensor with grad."""
        inputs, pos = self._build_inputs(image, user_prompt)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states  # tuple len (n_layers+1); each (B, T, D), grad-on

        feats = []
        for tap in self.feature_taps:
            layer_idx, pos_key = _TAP_SPECS[tap]
            feats.append(hs[layer_idx][0, pos[pos_key]])  # (D,)
        return torch.cat(feats, dim=-1)  # (feature_dim,)
