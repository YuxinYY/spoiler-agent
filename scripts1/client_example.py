from __future__ import annotations

import requests


def check_spoiler(text: str, content_type: str = "movie") -> str:
    url = "http://localhost:8000/predict_spoiler"
    response = requests.post(url, json={"text": text, "content_type": content_type}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") == 200:
        result = payload.get("data", {})
        if result.get("has_spoiler") == 1 and result.get("confidence", 0) > 0.8:
            return f"[spoiler hidden] reason: {result.get('key_spoiler_sentence', '')}"
        return f"[no spoiler] {text}"
    return text


if __name__ == "__main__":
    print(check_spoiler("The hero dies at the end to save the world."))
