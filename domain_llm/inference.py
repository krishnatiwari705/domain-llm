"""Lazy Transformers inference used by evaluation and the API."""
from __future__ import annotations

import json
from pathlib import Path

from .prompts import normalize_prediction, prompt


class IntentPredictor:
    def __init__(
        self,
        model_name: str,
        labels_path: str,
        adapter_path: str | None = None,
        device_map="auto",
        max_new_tokens=32,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.labels = json.loads(Path(labels_path).read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def _render(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "Return only the exact Banking77 intent label."},
            {"role": "user", "content": prompt(text, self.labels)},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict(self, text: str) -> tuple[str, str]:
        rendered = self._render(text)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            generated[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return normalize_prediction(raw, self.labels), raw.strip()

    def predict_batch(self, texts: list[str]) -> list[tuple[str, str]]:
        """Generate deterministic predictions for a batch of texts."""
        if not texts:
            return []

        rendered = [self._render(text) for text in texts]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            inputs = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            import torch

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = inputs.input_ids.shape[1]
            outputs = generated[:, prompt_length:]
            raws = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        return [
            (normalize_prediction(raw, self.labels), raw.strip())
            for raw in raws
        ]

    def parameter_counts(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "trainable_percentage": (trainable / total * 100.0) if total else 0.0,
        }
