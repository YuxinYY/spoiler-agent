# from __future__ import annotations

# import argparse
# import json

# from spoiler_agent.eval import evaluate_file


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Evaluate spoiler model on JSONL")
#     parser.add_argument("--input", required=True, help="Path to JSONL with text/label")
#     args = parser.parse_args()

#     metrics = evaluate_file(args.input)
#     print(json.dumps(metrics, indent=2))


# if __name__ == "__main__":
#     main()

# 12.31更新
from __future__ import annotations

import argparse
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_metrics(input_path: str) -> None:
    # 存储用于 Sklearn 计算的列表 (只包含格式正确的数据)
    y_true_valid = []
    y_pred_valid = []
    
    # 计数器
    total_count = 0
    format_error_count = 0
    
    print(f"Loading predictions from: {input_path}")
    print("-" * 40)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            total_count += 1
            try:
                row = json.loads(line)
                
                # 1. 获取真实标签 (Ground Truth)
                if "label" not in row:
                    # 如果连 True Label 都没有，这条数据是废的，不计入总数
                    total_count -= 1
                    continue
                
                truth = int(row["label"])

                # 2. 检查是否有格式错误 (Format Error)
                # 依据：spoiler_type 为 "error" 或者根本没有 has_spoiler 字段
                if row.get("spoiler_type") == "error" or "has_spoiler" not in row:
                    format_error_count += 1
                    continue  # 格式错误的数据，暂时不放入 valid 列表
                
                # 3. 获取预测结果
                pred = int(row["has_spoiler"])
                
                y_true_valid.append(truth)
                y_pred_valid.append(pred)

            except (ValueError, TypeError, json.JSONDecodeError):
                total_count -= 1 # 极端的行读取错误，忽略
                continue

    # ==========================================
    # 指标计算与展示
    # ==========================================
    
    if total_count == 0:
        print("Error: No valid data found.")
        return

    # 1. 格式错误率 (Format Compliance)
    # 这是 Fine-tuning 能够大幅改善的第一指标
    format_error_rate = format_error_count / total_count
    valid_count = len(y_true_valid)
    
    print(f"📊 [Format Analysis]")
    print(f"Total Samples: {total_count}")
    print(f"Format Errors: {format_error_count}")
    print(f"Format Error Rate: {format_error_rate:.2%}  <-- Fine-tuning 应重点降低此指标")
    print(f"Valid Valid JSONs: {valid_count}")
    print("-" * 40)

    if valid_count == 0:
        print("No valid predictions to evaluate accuracy.")
        return

    # 2. 有效样本的性能 (Model Performance on Valid Data)
    # 这代表模型在"能正常输出"时的智力水平
    print(f"🧠 [Model Intelligence] (Only on {valid_count} valid samples)")
    target_names = ["Non-Spoiler (0)", "Spoiler (1)"]
    print(classification_report(y_true_valid, y_pred_valid, target_names=target_names, digits=4))
    
    # 3. 系统整体准确率 (Overall System Accuracy)
    # 逻辑：将格式错误直接视为预测错误。这是最严苛、最真实的产品指标。
    # 计算公式：(正确预测数) / (总样本数，包含错误格式)
    
    correct_predictions = accuracy_score(y_true_valid, y_pred_valid, normalize=False)
    system_accuracy = correct_predictions / total_count
    
    print(f"🚀 [Overall System Performance]")
    print(f"System Accuracy: {system_accuracy:.4f}  (Considering errors as wrong predictions)")
    print(f"Valid Accuracy : {accuracy_score(y_true_valid, y_pred_valid):.4f}  (Ignoring errors)")
    
    print("-" * 40)
    print("Interpretation Guide:")
    print("1. If 'Format Error Rate' is high -> Model struggles with JSON structure.")
    print("2. If 'Valid Accuracy' is high but 'System Accuracy' is low -> Model is smart but formats badly.")
    print("3. Goal for Fine-tuning -> Drop 'Format Error Rate' to near 0%, boosting 'System Accuracy'.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spoiler model output.")
    parser.add_argument("--input", required=True, help="Path to pred_val.jsonl")
    args = parser.parse_args()

    evaluate_metrics(args.input)

if __name__ == "__main__":
    main()
