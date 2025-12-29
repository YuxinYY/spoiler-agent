from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _resolve_dtype(value: str):
    value = (value or "auto").lower().strip()
    if value in {"auto", ""}:
        return None
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    return None


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float


class QwenLocalLLM:
    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        device_map: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.model_id = model_id or _env("MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
        self.revision = revision or _env("MODEL_REVISION", "") or None
        self.device_map = device_map or _env("DEVICE_MAP", "auto")
        dtype_value = _resolve_dtype(dtype or _env("DTYPE", "auto"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=self.device_map,
            torch_dtype=dtype_value,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = config.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def load_default_llm() -> QwenLocalLLM:
    return QwenLocalLLM()


def load_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(_env("MAX_NEW_TOKENS", "256")),
        temperature=float(_env("TEMPERATURE", "0.1")),
        top_p=float(_env("TOP_P", "0.9")),
    )
