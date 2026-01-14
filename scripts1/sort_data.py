import json
import argparse

def sort_jsonl(input_file, output_file):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # 读取所有行，计算长度
        data = []
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # 计算 "text" 加上 "review" 的总长度作为排序依据
            text_len = len(obj.get('text', '')) + len(obj.get('review', ''))
            data.append((text_len, line))
    
    print(f"Sorting {len(data)} lines...")
    # 按长度排序
    data.sort(key=lambda x: x[0])
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, line in data:
            f.write(line)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sort_jsonl(args.input, args.output)
