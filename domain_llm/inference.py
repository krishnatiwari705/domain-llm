"""Lazy Transformers inference used by evaluation and the API."""
from __future__ import annotations
import json
from pathlib import Path
from .prompts import prompt, normalize_prediction

class IntentPredictor:
    def __init__(self, model_name: str, labels_path: str, adapter_path: str | None = None, device_map="auto", max_new_tokens=32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.labels = json.loads(Path(labels_path).read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def predict(self, text: str) -> tuple[str, str]:
        messages = [{"role": "system", "content": "Return only the exact Banking77 intent label."}, {"role": "user", "content": prompt(text, self.labels)}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return normalize_prediction(raw, self.labels), raw.strip()
