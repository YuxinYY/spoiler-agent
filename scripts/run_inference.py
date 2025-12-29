from __future__ import annotations

import argparse
import json

from spoiler_agent.infer import classify_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spoiler classification.")
    parser.add_argument("--text", help="Text to classify")
    parser.add_argument("--content-type", default="unknown")
    parser.add_argument("--input", help="JSONL file with text field")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    if args.text:
        result = classify_text(args.text, args.content_type)
        print(json.dumps(result.model_dump(), ensure_ascii=False))
        return

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get(args.text_field, "")
                result = classify_text(text, args.content_type)
                print(json.dumps(result.model_dump(), ensure_ascii=False))
        return

    raise SystemExit("Provide --text or --input")


if __name__ == "__main__":
    main()
