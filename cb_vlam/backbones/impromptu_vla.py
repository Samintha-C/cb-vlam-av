"""Impromptu-VLA backbone wrapper.

Wraps the released `aaaaaap/ImpromptuVLAModel/7B_AD_finetune` checkpoint
(Qwen2.5-VL-7B fine-tuned on Impromptu + nuScenes) and exposes a forward
pass that caches three feature vectors per sample, all derived from a
single forward pass:

- feat_endprompt_final:   last layer, last position of (prompt + assistant prefix)
- feat_endprompt_penult:  penultimate layer, same position
- feat_afterplan_final:   last layer, after the literal '<PLANNING>' string
                          is appended via teacher-forcing

The end-of-prompt position is where, at inference time, the model is about
to emit the first token of its answer. The after-PLANNING position is
where the model has committed to producing a planning trajectory and is
about to emit the first waypoint coordinate.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from cb_vlam.backbones.base import BaseBackbone


class ImpromptuVLABackbone(BaseBackbone):
    PLANNING_TAG = "<PLANNING>"

    def __init__(self, checkpoint_path: str, dtype: str = "bf16",
                 processor_path: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.dtype_str = dtype
        # LLaMA-Factory fine-tunes often omit image_processor_type in the
        # checkpoint's preprocessor_config.json. Pass a base-model name or
        # path to load the processor from a known-good source instead.
        self.processor_path = processor_path or str(checkpoint_path)
        self.model = None
        self.processor = None
        self.device = None
        self._planning_token_ids = None

    def load(self, device: str = "cuda") -> None:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )

        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.dtype_str]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        print(f"  Loading processor from {self.processor_path} ...")
        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.device = device

        # Pre-tokenize the literal "<PLANNING>" string for teacher-forcing
        ids = self.processor.tokenizer.encode(
            self.PLANNING_TAG, add_special_tokens=False
        )
        self._planning_token_ids = torch.tensor([ids], device=device, dtype=torch.long)
        print(f"  PLANNING tag tokenizes to {len(ids)} tokens: {ids}")

    @torch.inference_mode()
    def generate(self, image: Image.Image, user_prompt: str, max_new_tokens: int = 150) -> str:
        """Run autoregressive generation; return decoded new tokens only."""
        text = user_prompt.replace("<image>", "").strip()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt"
        ).to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def extract(self, image: Image.Image, user_prompt: str) -> Dict[str, np.ndarray]:
        # The training prompt contains an inline '<image>' placeholder. The
        # native Qwen2.5-VL chat template wants images as content entries,
        # so we strip the placeholder from text and add the image separately.
        text = user_prompt.replace("<image>", "").strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Teacher-forcing: append <PLANNING> to the prompt string *before*
        # calling the processor so it builds attention masks and RoPE
        # positions over the complete sequence in one shot.
        # Concatenating tensors manually after the fact leaves image_grid_thw
        # stale, which causes Qwen2.5-VL's get_rope_index to raise an
        # IndexError when the extended mask is larger than the rope tensor.
        full_text = prompt_text + self.PLANNING_TAG

        # Two processor calls: the first (CPU-only) measures where the prompt
        # ends after image-token expansion; the second is the actual input.
        prompt_len = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt"
        ).input_ids.shape[1]

        inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt"
        ).to(self.device)

        total_len = inputs.input_ids.shape[1]
        endprompt_pos = prompt_len - 1   # last token before <PLANNING>
        afterplan_pos = total_len - 1    # last <PLANNING> token

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states: tuple of (n_layers + 1) tensors, each (B, T, D)
        last_layer = outputs.hidden_states[-1][0]      # (T, D)
        penult_layer = outputs.hidden_states[-2][0]    # (T, D)

        return {
            "feat_endprompt_final":  last_layer[endprompt_pos].float().cpu().numpy(),
            "feat_endprompt_penult": penult_layer[endprompt_pos].float().cpu().numpy(),
            "feat_afterplan_final":  last_layer[afterplan_pos].float().cpu().numpy(),
        }
