import json
import argparse
import sys
from pathlib import Path

def check_balance(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"正在分析文件: {file_path} ...")
    
    spoiler_count = 0
    non_spoiler_count = 0
    total_count = 0
    error_count = 0
    
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                row = json.loads(line)
                # 尝试获取标签，兼容 'label' 和 'has_spoiler'
                label = row.get('label')
                if label is None:
                    label = row.get('has_spoiler')
                
                if label == 1:
                    spoiler_count += 1
                elif label == 0:
                    non_spoiler_count += 1
                else:
                    # 标签不是 0 或 1 的情况
                    pass
                
                total_count += 1
            except json.JSONDecodeError:
                error_count += 1
                continue

    # === 统计报告 ===
    print("-" * 40)
    print(f"📊 数据分布统计")
    print("-" * 40)
    print(f"Total Rows      : {total_count}")
    if error_count > 0:
        print(f"Format Errors   : {error_count}")
    
    valid_total = spoiler_count + non_spoiler_count
    if valid_total == 0:
        print("❌ 未找到有效的 label/has_spoiler 标签 (0或1)。请检查字段名。")
        return

    print(f"Non-Spoiler (0) : {non_spoiler_count} \t({non_spoiler_count/valid_total:.1%})")
    print(f"Spoiler (1)     : {spoiler_count} \t({spoiler_count/valid_total:.1%})")
    print("-" * 40)

    # === 平衡性判断 ===
    # 计算比例差异
    ratio = spoiler_count / non_spoiler_count if non_spoiler_count > 0 else float('inf')
    
    if 0.8 <= ratio <= 1.25:
        print("✅ 状态: 数据集基本平衡 (Balance Good)")
    else:
        print("⚠️ 状态: 数据集严重失衡 (Imbalanced)")
        if spoiler_count > non_spoiler_count:
            diff = spoiler_count - non_spoiler_count
            print(f"👉 建议: Spoiler 样本过多。请删减约 {diff} 条 Spoiler 样本。")
        else:
            diff = non_spoiler_count - spoiler_count
            print(f"👉 建议: Non-Spoiler 样本过多。请删减约 {diff} 条 Non-Spoiler 样本。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset class balance.")
    # 默认路径设为你常用的路径，方便直接跑
    parser.add_argument("--input", default="/mnt/data/projects/spoiler-agent/data/train.jsonl", help="Path to input jsonl")
    args = parser.parse_args()
    
    check_balance(args.input)