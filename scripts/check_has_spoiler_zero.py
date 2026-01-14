from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for records with has_spoiler == 0 in a JSONL file."
    )
    parser.add_argument(
        "--input",
        default="data/raw/goodreads_reviews_spoiler_raw.json",
        help="Path to the JSONL file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of matching records to print",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    total = 0
    matches = 0
    samples = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get("has_spoiler")
            if value == 0:
                matches += 1
                if len(samples) < args.sample:
                    samples.append(row)

    print(f"total_records={total}")
    print(f"has_spoiler_0={matches}")
    if samples:
        print("sample_records=")
        for item in samples:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
