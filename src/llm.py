from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _env(name: str, default: str) -> str:
    """Read environment variable with a default."""
    return os.getenv(name, default)


def _resolve_dtype(value: str | None):
    """Map string dtype -> torch dtype."""
    if value is None:
        return None
    value = value.lower().strip()
    if value in {"", "auto"}:
        return None
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    return None


def _env_bool(name: str, default: str = "true") -> bool:
    """Read a boolean-like env var."""
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y"}


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float


class LocalCausalLM:
    """
    Thin wrapper around a local HuggingFace CausalLM model.

    Key points for your setup:
    - 默认使用本地路径 /home/ubuntu/models/Meta-Llama-3-8B-Instruct
      （你把解压好的 Llama3 目录放这里，或者在 .env 里改 MODEL_ID。）
    - local_files_only=True，避免因为服务器连不上 HuggingFace 而卡死。
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        device_map: str | None = None,
        dtype: str | None = None,
    ) -> None:
        # 默认模型路径：本地目录，而不是 HF repo 名字
        default_model_path = "/mnt/data/projects/spoiler-agent/models/Meta-Llama-3-8B-Instruct"

        # 允许通过环境变量 MODEL_ID 覆盖
        self.model_id = model_id or _env("MODEL_ID", default_model_path)

        # 你是离线场景，一般不需要 revision；保留接口以防以后换成在线模型
        self.revision = revision or _env("MODEL_REVISION", "") or None

        # 默认自动把权重 map 到 GPU
        self.device_map = device_map or _env("DEVICE_MAP", "auto")

        dtype_value = _resolve_dtype(dtype or _env("DTYPE", "auto"))

        # 是否强制只用本地文件（你现在的环境建议一直 True）
        self.local_files_only = _env_bool("LOCAL_FILES_ONLY", "true")

        # ---- Load tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )

        # 有些 Llama 模型需要显式设置 pad_token
        if self.tokenizer.pad_token is None:
            # 通常用 eos_token 作为 pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Load model ----
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=self.device_map,
            torch_dtype=dtype_value,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig) -> str:
        """
        messages: [{"role": "system" | "user" | "assistant", "content": "..."}]
        returns: generated assistant text
        """
        # 使用 tokenizer 自带的 chat_template 把 messages 拼成 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = config.temperature > 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 只取新生成部分
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def load_default_llm() -> LocalCausalLM:
    return LocalCausalLM()


def load_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(_env("MAX_NEW_TOKENS", "256")),
        temperature=float(_env("TEMPERATURE", "0.1")),
        top_p=float(_env("TOP_P", "0.9")),
    )
