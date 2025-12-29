from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

from .infer import classify_text


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate_file(path: str | Path) -> dict[str, float]:
    data = load_jsonl(path)
    y_true = []
    y_pred = []
    for row in data:
        text = row.get("text", "")
        label = int(row.get("label", 0))
        result = classify_text(text)
        y_true.append(label)
        y_pred.append(int(result.has_spoiler))
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
