from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
# === 修改 1: 只导入 SFTTrainer，不需要 SFTConfig ===
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

PROMPT_TEMPLATE = """
Task: decide whether the review sentence contains a spoiler.
You are given the review sentence (target), book title, and full review as context.
Use the full review only for context; the label is about the review sentence.
Definitions:
- spoiler: reveals ending, key plot twist, core character fate, or mystery answer.
- not spoiler: general opinions, vague praise, or non-essential details.

Output JSON schema:
{{"has_spoiler":0|1,"confidence":0-1,"spoiler_type":"movie"|"novel"|"none","key_spoiler_sentence":"..."|"none"}}

Review sentence: {review_sentence}
Book title: {book_title}
Full review: {review_text}
Output:
""".strip()


def load_jsonl(path: Path) -> list[dict]:
    print(f"正在统计文件行数: {path} ...")
    try:
        total_lines = sum(1 for _ in open(path, 'r', encoding="utf-8"))
    except Exception:
        total_lines = None

    items = []
    print(f"正在读取数据...")
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total_lines, desc="Loading JSONL", unit="lines"):
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _pick_field(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def _build_prompt(row: dict) -> tuple[str, str]:
    review_sentence = _pick_field(row, ("review_sentence", "text", "sentence"))
    book_title = _pick_field(row, ("book_titles", "book_title", "title", "book"))
    review_text = _pick_field(row, ("review", "full_review", "review_text"))
    if not review_text:
        review_text = review_sentence
    if not book_title:
        book_title = "Unknown Book"
    prompt = PROMPT_TEMPLATE.format(
        review_sentence=review_sentence,
        book_title=book_title,
        review_text=review_text,
    )
    return prompt, review_sentence


def to_sft_records(rows: list[dict], mode: str, tokenizer) -> list[dict]:
    records = []
    eos = tokenizer.eos_token 
    
    if mode == "text_label":
        for row in tqdm(rows, desc="Formatting Prompts", unit="sample"):
            prompt, review_sentence = _build_prompt(row)
            label = int(row.get("label", 0))
            payload = {
                "has_spoiler": label,
                "confidence": 1.0,
                "spoiler_type": "none" if label == 0 else "novel",
                "key_spoiler_sentence": "none" if label == 0 else review_sentence[:120],
            }
            records.append({"text": f"{prompt}{json.dumps(payload, ensure_ascii=False)}{eos}"})
        return records

    raise ValueError("Unsupported dataset format for this project")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning scaffold for Llama.")
    parser.add_argument("--input", required=True, help="Path to JSONL data")
    parser.add_argument("--dataset-format", choices=["text_label"], default="text_label")
    parser.add_argument("--model-id", default="/mnt/data/projects/spoiler-agent/models/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--output-dir", default="models/llama3-spoiler-lora")
    parser.add_argument("--epochs", type=int, default=1) 
    parser.add_argument("--batch-size", type=int, default=4) 
    parser.add_argument("--max-seq-len", type=int, default=2048) 
    args = parser.parse_args()

    # 1. Load Tokenizer
    print(f"正在加载 Tokenizer: {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 2. Prepare Data
    rows = load_jsonl(Path(args.input))
    records = to_sft_records(rows, args.dataset_format, tokenizer)
    dataset = Dataset.from_list(records)
    print(f"数据集构建完成，样本数: {len(dataset)}")

    # 3. Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 4. Load Base Model
    print("正在加载 Base Model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    # 5. LoRA Config
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 模型准备完毕。训练参数量:")
    model.print_trainable_parameters()

    # # 6. Training Args (回归到标准的 TrainingArguments)
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.epochs,tmu
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=4, 
    #     logging_steps=25,
    #     save_steps=500,
    #     learning_rate=2e-4, 
    #     fp16=False,
    #     bf16=True, 
    #     max_grad_norm=0.3,
    #     warmup_ratio=0.03,
    #     lr_scheduler_type="cosine",
    #     disable_tqdm=False, 
    #     report_to="tensorboard",
    #     # 这里不需要放 max_seq_length，因为它不是 TrainingArguments 的标准参数
    # )

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        # === 修改 1: 降小 Batch Size 防止 OOM ===
        per_device_train_batch_size=1,  # 之前是 4，降到 1
        
        # === 修改 2: 增加累积步数，保持总 Batch Size 不变 ===
        gradient_accumulation_steps=16, # 之前是 4，改为 16 (1*16 = 4*4)
        
        logging_steps=25,
        save_steps=500,
        learning_rate=2e-4, 
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        disable_tqdm=False, 
        
        # === 修改 3: 这里的 report_to="tensorboard" 需要 tensorboard 包
        # 如果你没装且不想装，改为 report_to="none"
        report_to="none", 

        # === 关键修改 4: 开启梯度检查点 (救命稻草) ===
        gradient_checkpointing=True, 
        # 为了兼容性，也可以设置 gradient_checkpointing_kwargs
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 7. Start Trainer (trl 0.12.0 风格)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # === 关键修改：改回 'tokenizer' ===
        train_dataset=dataset,
        args=training_args,
        # === 关键修改：直接传给 Trainer，不通过 Config ===
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False
    )

    print("开始训练...")
    trainer.train()
    
    print(f"保存模型至: {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()