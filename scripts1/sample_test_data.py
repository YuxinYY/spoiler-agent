import json
import random
import argparse
from pathlib import Path

def sample_and_sort_test_data(input_path, output_path, target_per_class=2500):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"错误: 找不到文件 {input_path}")
        return

    print(f"正在读取 {input_path} ...")
    
    spoilers = []
    non_spoilers = []
    
    # 1. 读取并按类别分组
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # 兼容不同的标签字段名
                label = row.get("label")
                if label is None:
                    label = row.get("has_spoiler")
                
                # 必须有标签才能平衡，如果没标签(盲测集)，这个脚本可能需要修改为纯随机
                if label == 1:
                    spoilers.append(row)
                elif label == 0:
                    non_spoilers.append(row)
            except json.JSONDecodeError:
                continue

    print(f"原始分布 -> Spoiler: {len(spoilers)}, Non-Spoiler: {len(non_spoilers)}")
    
    # 2. 执行抽样
    # 如果某一类不够 2500，就取全部
    n_sp = min(len(spoilers), target_per_class)
    n_non = min(len(non_spoilers), target_per_class)
    
    random.seed(42) # 固定种子，保证每次抽的一样
    sampled_sp = random.sample(spoilers, n_sp)
    sampled_non = random.sample(non_spoilers, n_non)
    
    combined = sampled_sp + sampled_non
    print(f"抽样后 -> Spoiler: {len(sampled_sp)}, Non-Spoiler: {len(sampled_non)}, 总计: {len(combined)}")

    # 3. 【关键步骤】按长度排序 (为了加速推理)
    # 计算 (text + review) 的长度作为排序依据
    print("正在按长度排序以优化推理速度...")
    combined.sort(key=lambda x: len(str(x.get('text', ''))) + len(str(x.get('review', ''))))

    # 4. 写入文件
    print(f"正在写入 {output_path} ...")
    with output_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print("完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认读取原始的 test.jsonl (或者 test_sorted.jsonl 也可以，内容是一样的)
    parser.add_argument("--input", default="data/splits/test.jsonl") 
    parser.add_argument("--output", default="data/splits/test_5k_sorted.jsonl")
    parser.add_argument("--count", type=int, default=2500, help="每类样本数量")
    args = parser.parse_args()
    
    sample_and_sort_test_data(args.input, args.output, args.count)