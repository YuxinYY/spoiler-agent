from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .llm import load_default_llm, load_generation_config
from .prompts import build_messages
from .schema import SpoilerClassification, normalize_spoiler_type

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_LLM = None
_GEN_CONFIG = None


def _get_runtime():
    global _LLM, _GEN_CONFIG
    if _LLM is None:
        _LLM = load_default_llm()
    if _GEN_CONFIG is None:
        _GEN_CONFIG = load_generation_config()
    return _LLM, _GEN_CONFIG


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _coerce_payload(payload: dict[str, Any] | None) -> SpoilerClassification:
    if not payload:
        return SpoilerClassification(
            has_spoiler=0,
            confidence=0.0,
            spoiler_type="none",
            key_spoiler_sentence="none",
        )
    if "spoiler_type" in payload:
        payload["spoiler_type"] = normalize_spoiler_type(str(payload["spoiler_type"]))
    if "has_spoiler" in payload:
        payload["has_spoiler"] = int(payload["has_spoiler"])
    if "confidence" in payload:
        payload["confidence"] = float(payload["confidence"])
    if payload.get("has_spoiler", 0) == 0:
        payload.setdefault("spoiler_type", "none")
        payload.setdefault("key_spoiler_sentence", "none")
    return SpoilerClassification(**payload)


def classify_text(text: str, content_type: str = "unknown") -> SpoilerClassification:
    llm, config = _get_runtime()
    messages = build_messages(text, content_type)
    output = llm.generate(messages, config)
    payload = _extract_json(output)
    try:
        return _coerce_payload(payload)
    except ValidationError:
        return SpoilerClassification(
            has_spoiler=0,
            confidence=0.0,
            spoiler_type="none",
            key_spoiler_sentence="none",
        )
