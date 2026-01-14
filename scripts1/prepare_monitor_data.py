import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_balanced_dataset(input_path, output_path, total_samples=100):
    """
    从验证集中抽取平衡的样本用于监控
    total_samples: 最终输出的总样本数 (例如 100 表示 50正 + 50负)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入文件 {input_path}")
        return

    print(f"正在读取 {input_path} ...")
    
    # 1. 分桶读取
    spoilers = []
    non_spoilers = []
    
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                label = row.get("label")
                
                # 兼容可能的字段名差异
                if label is None:
                    label = row.get("has_spoiler")
                
                if label == 1:
                    spoilers.append(row)
                elif label == 0:
                    non_spoilers.append(row)
            except json.JSONDecodeError:
                continue

    print(f"\n📊 原始数据统计:")
    print(f"   - Spoiler (1)    : {len(spoilers)}")
    print(f"   - Non-Spoiler (0): {len(non_spoilers)}")

    # 2. 检查数量是否足够
    per_class = total_samples // 2
    if len(spoilers) < per_class or len(non_spoilers) < per_class:
        print(f"\n⚠️ 警告: 数据不足以凑齐 {total_samples} 条平衡数据！")
        per_class = min(len(spoilers), len(non_spoilers))
        print(f"   -> 将自动调整为每类 {per_class} 条 (总计 {per_class*2} 条)")

    # 3. 随机抽样
    random.seed(42) # 固定种子保证可复现
    sampled_spoilers = random.sample(spoilers, per_class)
    sampled_non = random.sample(non_spoilers, per_class)
    
    # 4. 合并并再次打乱
    final_data = sampled_spoilers + sampled_non
    random.shuffle(final_data)
    
    # 5. 写入文件
    print(f"\n正在写入 {output_path} ...")
    with output_path.open("w", encoding="utf-8") as f:
        for row in final_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"✅ 完成！已生成监控数据集: {per_class} 正 + {per_class} 负 = {len(final_data)} 条")
    print(f"   文件路径: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/val.jsonl", help="原始的大验证集")
    parser.add_argument("--output", default="data/val_monitor_balanced.jsonl", help="输出的监控小数据集")
    parser.add_argument("--count", type=int, default=100, help="需要的总样本数 (默认100)")
    args = parser.parse_args()
    
    prepare_balanced_dataset(args.input, args.output, args.count)