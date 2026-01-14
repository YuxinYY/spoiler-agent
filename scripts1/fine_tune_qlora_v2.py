from __future__ import annotations

import argparse
import json
import random
import sys
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ==========================================
# === 1. Prompts 逻辑 (复刻自 src/spoiler_agent/prompts.py) ===
# ==========================================

SYSTEM_PROMPT = """
You are a spoiler classification assistant for book reviews.
Decide whether the review sentence contains spoilers.
Return ONLY a JSON object that matches the required schema.
""".strip()

USER_TEMPLATE = """
Task: decide whether the review sentence contains a spoiler.
You are given the review sentence (target), book title, and full review as context.
Use the full review only for context; the label is about the review sentence.
Definitions:
- spoiler: reveals ending, key plot twist, core character fate, or mystery answer.
- not spoiler: general opinions, vague praise, or non-essential details.

Output JSON schema:
{{
  "has_spoiler": 0 or 1,
  "confidence": 0.0-1.0,
  "spoiler_type": "movie" | "novel" | "none",
  "key_spoiler_sentence": "..." | "none"
}}

Example 1:
Review sentence: The hero dies at the end to save the world.
Book title: Unknown Book
Full review: The hero dies at the end to save the world.
Output: {{"has_spoiler":1,"confidence":0.98,"spoiler_type":"movie","key_spoiler_sentence":"The hero dies at the end to save the world."}}

Example 2:
Review sentence: The visuals are stunning and the soundtrack is great.
Book title: Unknown Book
Full review: The visuals are stunning and the soundtrack is great.
Output: {{"has_spoiler":0,"confidence":0.95,"spoiler_type":"none","key_spoiler_sentence":"none"}}

Review sentence: {review_sentence}
Book title: {book_title}
Full review: {review_text}
Content type hint: {content_type}
""".strip()

def _pick_field(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""

def _extract_inputs(input_data: str | dict) -> tuple[str, str, str]:
    if isinstance(input_data, dict):
        review_sentence = _pick_field(input_data, ("text", "review_sentence", "sentence"))
        book_title = _pick_field(input_data, ("book_titles", "book_title", "title", "book"))
        review_text = _pick_field(input_data, ("review", "full_review", "review_text"))
        if not review_text:
            review_text = review_sentence
        if not book_title:
            book_title = "Unknown Book"
        return review_sentence, book_title, review_text
    text = str(input_data).strip()
    return text, "Unknown Book", text

def build_messages(input_text: str | dict, content_type: str = "unknown") -> list[dict[str, str]]:
    """
    完全复刻 prompts.py 的逻辑，构建 System + User 消息
    """
    review_sentence, book_title, review_text = _extract_inputs(input_text)
    user_prompt = USER_TEMPLATE.format(
        review_sentence=review_sentence,
        book_title=book_title,
        review_text=review_text,
        content_type=content_type,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

# ==========================================
# === 2. 辅助函数 (输出解析) ===
# ==========================================

def parse_json_response(raw_output: str) -> dict:
    """
    与 infer.py 保持一致的解析逻辑
    """
    try:
        clean_json = raw_output.strip()
        
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].strip()
        
        start = clean_json.find('{')
        end = clean_json.rfind('}')
        
        if start != -1 and end != -1:
            clean_json = clean_json[start:end+1]
            return json.loads(clean_json)
        else:
            raise ValueError("No JSON brackets found")
            
    except Exception as e:
        return {
            "has_spoiler": 0,
            "spoiler_type": "error",
            "error_msg": str(e),
            "raw_output": raw_output
        }

# ==========================================
# === 3. 监控回调 (Monitor) - 严格一致版 ===
# ==========================================

class TrainingMonitorCallback(TrainerCallback):
    def __init__(self, tokenizer, validation_inputs, check_steps=50, num_samples=4):
        self.tokenizer = tokenizer
        self.validation_inputs = validation_inputs
        self.check_steps = check_steps
        self.num_samples = num_samples
        
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step > 0 and state.global_step % self.check_steps == 0:
            
            model.eval()
            current_batch_rows = random.sample(self.validation_inputs, min(len(self.validation_inputs), self.num_samples))
            
            print(f"\n\n🔍 [Step {state.global_step}] 中间检测 (Samples: {len(current_batch_rows)})...")
            
            error_count = 0
            spoiler_count = 0
            
            with torch.no_grad():
                for row in current_batch_rows:
                    # 1. 构建 Prompt (与 infer.py 的 build_messages 一致)
                    messages = build_messages(row, content_type="book")
                    
                    # 2. 转成文本
                    input_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)
                    
                    # 3. 生成 (参数尽量与 infer.py 默认保持一致)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    # === 4. 解码 (关键修正：与 infer.py 保持一致的切片逻辑) ===
                    # 获取输入的长度
                    input_len = inputs.input_ids.shape[1]
                    # 只取新生成的 token (去掉前面的 Prompt)
                    generated_tokens = outputs[0][input_len:]
                    # 解码
                    full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # === Debug: 打印原始输出，确保干净 ===
                    print(f"📝 [Raw]: {full_text.strip()}")
                    
                    # 5. 解析 (使用与 infer.py 一致的逻辑)
                    result = parse_json_response(full_text)
                    
                    if result.get("spoiler_type") == "error":
                        error_count += 1
                    elif result.get("has_spoiler") == 1:
                        spoiler_count += 1
            
            # 汇报
            total = len(current_batch_rows)
            err_rate = error_count / total
            valid_total = total - error_count
            spoiler_rate = spoiler_count / valid_total if valid_total > 0 else 0
            
            print(f"📊 检测结果:")
            print(f"   - 格式错误: {error_count}/{total} ({err_rate:.0%})")
            print(f"   - Spoiler占比: {spoiler_rate:.0%}")
            
            print("-" * 30 + "\n")
            model.train()

# ==========================================
# === 4. 数据处理 (Data Loading) ===
# ==========================================

def load_jsonl(path: Path) -> list[dict]:
    print(f"正在读取数据: {path} ...")
    items = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc="Loading JSONL"):
                if line.strip():
                    items.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ 文件未找到: {path}")
        sys.exit(1)
    return items

def to_sft_records(rows: list[dict], mode: str, tokenizer) -> list[dict]:
    records = []
    if mode == "text_label":
        for row in tqdm(rows, desc="Formatting Train Data", unit="sample"):
            # 1. 使用 build_messages 构建 System + User
            messages = build_messages(row, content_type="book")
            
            # 2. 构建 Assistant 回复
            label = int(row.get("label", 0))
            
            # 根据 label 动态生成 spoiler_type 和 key_sentence
            # 这样模型能学到 label 和 type 的关联
            spoiler_type = "novel" if label == 1 else "none"
            key_sentence = row.get("text", "")[:100] if label == 1 else "none"
            
            payload = {
                "has_spoiler": label,
                "confidence": 1.0,
                "spoiler_type": spoiler_type,
                "key_spoiler_sentence": key_sentence,
            }
            assistant_content = json.dumps(payload, ensure_ascii=False)

            # 3. 将 Assistant 回复加入 messages
            messages.append({"role": "assistant", "content": assistant_content})
            
            # 4. 生成最终文本
            final_text = tokenizer.apply_chat_template(messages, tokenize=False)
            records.append({"text": final_text})
        return records
    raise ValueError("Unsupported format")

# ==========================================
# === 5. Main ===
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with Chat Template & Monitor")
    parser.add_argument("--input", required=True, help="Training data (JSONL)")
    parser.add_argument("--validation-file", default=None, help="Validation data (JSONL) for monitoring.")
    
    parser.add_argument("--model-id", default="/mnt/data/projects/spoiler-agent/models/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--output-dir", default="models/llama3-spoiler-lora-v4-chat")
    
    # === 参数微调 ===
    parser.add_argument("--epochs", type=int, default=1) 
    parser.add_argument("--batch-size", type=int, default=4) 
    parser.add_argument("--max-seq-len", type=int, default=4096) 
    parser.add_argument("--lr", type=float, default=1e-4) # 降低 LR
    parser.add_argument("--dropout", type=float, default=0.1) # 增加 Dropout
    
    args = parser.parse_args()

    # 1. Tokenizer
    print(f"Loading Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 2. Data Preparation
    all_rows = load_jsonl(Path(args.input))
    
    train_rows = []
    val_rows = []
    
    if args.validation_file:
        print(f"Loading validation file: {args.validation_file}")
        train_rows = all_rows
        val_rows = load_jsonl(Path(args.validation_file))
        # 验证集只取前50条做监控
        val_rows = val_rows[:50]
    else:
        print("Splitting validation set from input...")
        random.seed(42)
        random.shuffle(all_rows)
        # 切 20 条做监控
        val_rows = all_rows[:20]
        train_rows = all_rows[20:]
        
    print(f"Train Size: {len(train_rows)}, Monitor Size: {len(val_rows)}")

    # 转换训练数据
    train_records = to_sft_records(train_rows, "text_label", tokenizer)
    train_dataset = Dataset.from_list(train_records)

    # 3. Data Collator (Loss Masking)
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 4. Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    # 5. LoRA Config
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        logging_steps=25,
        save_steps=200, 
        learning_rate=args.lr, 
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none", 
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 7. Monitor Callback
    # 注意: 这里传入原始的 val_rows，让 callback 内部调用 build_messages
    # 这样确保了 monitor 和 train 使用同一套 prompt 逻辑
    monitor_callback = TrainingMonitorCallback(
        tokenizer=tokenizer,
        validation_inputs=val_rows,
        check_steps=50, 
        num_samples=4
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        packing=False,
        callbacks=[monitor_callback]
    )

    print("开始训练...")
    trainer.train()
    
    print(f"保存模型至: {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()