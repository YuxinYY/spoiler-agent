from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

PROMPT_TEMPLATE = """
Task: classify the input as spoiler or not.
Output JSON schema:
{{"has_spoiler":0|1,"confidence":0-1,"spoiler_type":"movie"|"novel"|"none","key_spoiler_sentence":"..."|"none"}}
Input: {text}
Output:
""".strip()


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def to_sft_records(rows: list[dict], mode: str) -> list[dict]:
    records = []
    if mode == "prompt_completion":
        for row in rows:
            prompt = row.get("prompt", "")
            completion = row.get("completion", "")
            records.append({"text": f"{prompt}\n{completion}"})
        return records

    if mode == "text_label":
        for row in rows:
            text = row.get("text", "")
            label = int(row.get("label", 0))
            payload = {
                "has_spoiler": label,
                "confidence": 1.0,
                "spoiler_type": "none" if label == 0 else "movie",
                "key_spoiler_sentence": "none" if label == 0 else text[:120],
            }
            prompt = PROMPT_TEMPLATE.format(text=text)
            records.append({"text": f"{prompt}{json.dumps(payload, ensure_ascii=False)}"})
        return records

    raise ValueError("Unsupported dataset format")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning scaffold for Qwen.")
    parser.add_argument("--input", required=True, help="Path to JSONL data")
    parser.add_argument("--dataset-format", choices=["prompt_completion", "text_label"], default="prompt_completion")
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--output-dir", default="models/qwen-qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    records = to_sft_records(rows, args.dataset_format)
    dataset = Dataset.from_list(records)

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
