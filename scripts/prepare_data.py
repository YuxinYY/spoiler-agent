from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare spoiler dataset splits.")
    parser.add_argument("--input", required=True, help="Path to input CSV/TSV/JSON/JSONL")
    parser.add_argument("--text-field", default="text", help="Column name for text")
    parser.add_argument("--label-field", default="label", help="Column name for label")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    path = Path(args.input)
    df = load_table(path)
    if args.text_field not in df.columns or args.label_field not in df.columns:
        raise ValueError("Input missing required columns")

    df = df[[args.text_field, args.label_field]].dropna()
    df[args.text_field] = df[args.text_field].astype(str).map(clean_text)
    df[args.label_field] = df[args.label_field].astype(int)

    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[args.label_field],
        random_state=42,
    )
    val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[args.label_field],
        random_state=42,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(train.to_dict(orient="records"), output_dir / "train.jsonl")
    write_jsonl(val.to_dict(orient="records"), output_dir / "val.jsonl")
    write_jsonl(test.to_dict(orient="records"), output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
