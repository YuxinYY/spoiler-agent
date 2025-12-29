from __future__ import annotations

import argparse
import json

from spoiler_agent.eval import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spoiler model on JSONL")
    parser.add_argument("--input", required=True, help="Path to JSONL with text/label")
    args = parser.parse_args()

    metrics = evaluate_file(args.input)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
