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

    def __init__(self, checkpoint_path: str, dtype: str = "bf16"):
        self.checkpoint_path = Path(checkpoint_path)
        self.dtype_str = dtype
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

        self.processor = AutoProcessor.from_pretrained(str(self.checkpoint_path))
        self.device = device

        # Pre-tokenize the literal "<PLANNING>" string for teacher-forcing
        ids = self.processor.tokenizer.encode(
            self.PLANNING_TAG, add_special_tokens=False
        )
        self._planning_token_ids = torch.tensor([ids], device=device, dtype=torch.long)
        print(f"  PLANNING tag tokenizes to {len(ids)} tokens: {ids}")

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

        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        # Length of the prompt sequence (before we append <PLANNING>)
        prompt_len = inputs.input_ids.shape[1]

        # Append <PLANNING> via teacher-forcing — one forward pass gives us
        # both end-of-prompt and after-PLANNING hidden states.
        planning_ids = self._planning_token_ids
        n_planning = planning_ids.shape[1]

        extended_ids = torch.cat([inputs.input_ids, planning_ids], dim=1)
        extended_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(planning_ids)], dim=1
        )

        forward_kwargs = dict(inputs)
        forward_kwargs["input_ids"] = extended_ids
        forward_kwargs["attention_mask"] = extended_mask

        outputs = self.model(
            **forward_kwargs,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states: tuple of (n_layers + 1) tensors, each (B, T, D)
        last_layer = outputs.hidden_states[-1][0]      # (T, D)
        penult_layer = outputs.hidden_states[-2][0]    # (T, D)

        # Position prompt_len-1 = last token of the prompt (before PLANNING)
        # Position prompt_len+n_planning-1 = last PLANNING token
        endprompt_pos = prompt_len - 1
        afterplan_pos = prompt_len + n_planning - 1

        return {
            "feat_endprompt_final":  last_layer[endprompt_pos].float().cpu().numpy(),
            "feat_endprompt_penult": penult_layer[endprompt_pos].float().cpu().numpy(),
            "feat_afterplan_final":  last_layer[afterplan_pos].float().cpu().numpy(),
        }
