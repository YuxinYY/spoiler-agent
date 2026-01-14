# run_inference.py
# 适配 infer.py v1.1 更新2
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from typing import List, Dict, Any

# 引入 tqdm 用于显示进度条
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 直接引入 classify_text，不再需要引入其他 transformers/peft 库
from src.spoiler_agent.infer import classify_text


def _pick_field(data: Dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def build_model_input(row: Dict[str, Any], text_field: str) -> Dict[str, str]:
    text = str(row.get(text_field, "")).strip()
    if not text:
        text = _pick_field(row, ("review_sentence", "text", "sentence"))
    book_title = _pick_field(row, ("book_titles", "book_title", "title", "book"))
    review = _pick_field(row, ("review", "full_review", "review_text"))
    if not review:
        review = text
    return {
        "text": text,
        "book_titles": str(book_title).strip(),
        "review": str(review).strip(),
    }


def batch_inference(input_path: str, output_path: str, args: argparse.Namespace):
    """
    批量推理主循环
    """
    batch_size = args.batch_size
    
    buffer_inputs: List[Dict[str, Any]] = []
    buffer_rows: List[Dict[str, Any]] = [] 

    print(f"Dataset: {input_path}")
    print(f"Output:  {output_path}")
    print(f"Batch Size: {batch_size}")
    if args.adapter:
        print(f"🔥 Using Adapter: {args.adapter}")
    else:
        print("🧊 Using Base Model (No Adapter)")

    if args.raw_prompt:
        print("Mode: Raw Text Prompt (Best for Fine-tuned v1)")
    else:
        print("Mode: Chat Template (Best for Base Model)")

    print("-" * 30)

    # 统计时间
    start_time = time.time()
    total_processed = 0

    # === [新增] 错误率监控计数器 ===
    monitor_total = 0
    monitor_errors = 0

    use_chat = not args.raw_prompt

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Inference"):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                model_input = build_model_input(row, args.text_field)
                
                buffer_inputs.append(model_input)
                buffer_rows.append(row) 

                # === 当缓冲区达到 Batch Size 时，触发推理 ===
                if len(buffer_inputs) >= batch_size:
                    predictions = classify_text(
                        buffer_inputs, 
                        content_type=args.content_type, 
                        adapter_path=args.adapter,
                        use_chat_template=use_chat
                    )
                    
                    # === [新增] 实时监控逻辑 ===
                    batch_err_count = 0
                    for p in predictions:
                        # 检查是否有 error 标记
                        if p.get("spoiler_type") == "error":
                            batch_err_count += 1
                    
                    monitor_errors += batch_err_count
                    monitor_total += len(predictions)
                    
                    # 计算错误率
                    error_rate = monitor_errors / monitor_total
                    
                    # 如果当前 Batch 有错，或者整体错误率过高，打印日志
                    if batch_err_count > 0:
                         tqdm.write(f"⚠️ [Monitor] 本批次发现 {batch_err_count} 个格式错误。累计错误率: {error_rate:.2%}")
                    
                    # 🚨 熔断报警：如果处理超过 32 条且错误率 > 20%，疯狂报警
                    if monitor_total > 32 and error_rate > 0.20:
                        tqdm.write("\n" + "!"*50)
                        tqdm.write(f"🚨🚨🚨 严重警告 (CRITICAL WARNING) 🚨🚨🚨")
                        tqdm.write(f"当前错误率已高达 {error_rate:.2%} (阈值 20%)")
                        tqdm.write(f"这说明 Raw Prompt 设置可能未生效，或者模型出现崩塌。")
                        tqdm.write(f"建议立即手动停止 (Ctrl+C) 并检查参数！")
                        tqdm.write("!"*50 + "\n")
                        # 稍微 sleep 一下让你能看清报错
                        time.sleep(2)

                    # 2. 将预测结果合并回原始数据
                    for original_row, pred_result in zip(buffer_rows, predictions):
                        original_row.update(pred_result)
                        f_out.write(json.dumps(original_row, ensure_ascii=False) + "\n")
                    
                    # 3. 清空缓冲区
                    buffer_inputs = []
                    buffer_rows = []
                    total_processed += batch_size

            except json.JSONDecodeError:
                continue

        # === 处理剩余不足一个 Batch 的数据 ===
        if buffer_inputs:
            predictions = classify_text(
                buffer_inputs, 
                content_type=args.content_type, 
                adapter_path=args.adapter,
                use_chat_template=use_chat
            )
            # 同样的监控逻辑
            batch_err_count = sum(1 for p in predictions if p.get("spoiler_type") == "error")
            monitor_errors += batch_err_count
            monitor_total += len(predictions)

            for original_row, pred_result in zip(buffer_rows, predictions):
                original_row.update(pred_result)
                f_out.write(json.dumps(original_row, ensure_ascii=False) + "\n")
            total_processed += len(buffer_inputs)

    end_time = time.time()
    duration = end_time - start_time
    avg_speed = total_processed / duration if duration > 0 else 0

    print("-" * 30)
    print(f"完成！总耗时: {duration:.2f}s")
    print(f"平均速度: {avg_speed:.2f} samples/sec")
    print(f"结果已保存至: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spoiler classification with batching.")
    parser.add_argument("--text", help="Single text to classify (debug mode)")
    parser.add_argument("--content-type", default="unknown", help="Content type (e.g., book, review)")
    parser.add_argument("--input", help="Input JSONL file path")
    parser.add_argument("--output", help="Output JSONL file path")
    parser.add_argument("--text-field", default="text", help="Field name containing the text in JSONL")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (default: 32)")
    
    # === 新增参数：接收 adapter 路径 ===
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter folder (optional)")
    # === Add --raw-prompt argument ===
    parser.add_argument("--raw-prompt", action="store_true", 
                        help="Use raw prompt (no chat template). REQUIRED for v1 fine-tuned model.")

    args = parser.parse_args()

    use_chat = not args.raw_prompt

    # 1. 单条测试模式
    if args.text:
        print("Running single text inference...")
        # === 关键修改：传入 adapter_path ===
        result = classify_text(
            args.text, 
            content_type=args.content_type, 
            adapter_path=args.adapter,
            use_chat_template=use_chat
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 2. 文件批量模式
    if args.input:
        if args.output:
            output_path = args.output
        else:
            output_path = "/mnt/data/projects/spoiler-agent/outputs/pred_val.jsonl"
        
        # 清空/新建输出文件
        open(output_path, "w").close()

        batch_inference(args.input, output_path, args)
        return

    raise SystemExit("Please provide --text or --input")


if __name__ == "__main__":
    main()
