
from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
from pathlib import Path
from typing import Iterable, Set

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 引入 tqdm 显示进度

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPOILER_TAG_RE = re.compile(r"\((view|hide) spoiler\)", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_series(series: pd.Series) -> pd.Series:
    series = series.astype(str)
    series = series.str.replace(_URL_RE, "", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True)
    return series.str.strip()


def _looks_like_jsonl(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            return line.startswith("{")
    return False


def load_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if path.suffix.lower() in {".jsonl", ".json"}:
        lines = path.suffix.lower() == ".jsonl" or _looks_like_jsonl(path)
        return pd.read_json(path, lines=lines, nrows=max_rows)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_from_spoiler_tags(text: str) -> int:
    if _SPOILER_TAG_RE.search(text or ""):
        return 1
    return 0


def apply_spoiler_ids(
    df: pd.DataFrame,
    spoiler_df: pd.DataFrame,
    id_field: str,
    label_field: str,
) -> pd.DataFrame:
    if id_field not in df.columns:
        raise ValueError(f"Missing id field in main dataset: {id_field}")
    if id_field not in spoiler_df.columns:
        raise ValueError(f"Missing id field in spoiler dataset: {id_field}")
    spoiler_ids = set(spoiler_df[id_field].astype(str).tolist())
    df[label_field] = df[id_field].astype(str).apply(lambda v: 1 if v in spoiler_ids else 0)
    return df


def load_spoiler_ids(
    path: Path,
    id_field: str,
    max_rows: int | None,
    chunksize: int,
) -> Set[str]:
    print(f"Loading spoiler IDs from {path}...")
    lines = path.suffix.lower() == ".jsonl" or _looks_like_jsonl(path)
    spoiler_ids: Set[str] = set()
    try:
        for chunk in pd.read_json(path, lines=lines, chunksize=chunksize):
            if id_field not in chunk.columns:
                if "review_id" in chunk.columns:
                    id_field = "review_id"
                else:
                    raise ValueError(f"Missing id field in spoiler dataset: {id_field}")
            
            spoiler_ids.update(chunk[id_field].dropna().astype(str).unique())
            
            if max_rows is not None and len(spoiler_ids) >= max_rows:
                break
    except Exception as e:
        print(f"Warning loading spoiler IDs: {e}")
        
    print(f"Loaded {len(spoiler_ids)} spoiler IDs.")
    return spoiler_ids


def iter_review_sentences(row: dict) -> Iterable[tuple[str, int]]:
    sentences = row.get("review_sentences") or []
    for item in sentences:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = int(item[0])
        text = item[1]
        if text is None:
            continue
        yield str(text), label


# ========== MODIFIED START ==========
# 核心修改：迭代分句时，新增【整段评论full_review】+【书籍ID book_id】的传递
def iter_review_sentence_rows(
    path: Path,
    max_rows: int | None,
    book_id_field: str,
) -> Iterable[dict]:
    for idx, row in enumerate(iter_jsonl(path)):
        if max_rows is not None and idx >= max_rows:
            break
        review_id = row.get("review_id")
        # 1. 提取整段评论（核心新增）：优先取review_text，无则取review_sentences拼接
        full_review = row.get("review_text", "").strip()
        if not full_review and row.get("review_sentences"):
            full_review = " ".join([item[1] for item in row["review_sentences"] if len(item)>=2 and item[1]])
        full_review = clean_text(full_review)
        # 2. 提取书籍ID（核心新增）：从原始数据中取book_id字段
        book_id = str(row.get(book_id_field, "")).strip()
        
        for text, label in iter_review_sentences(row):
            cleaned = clean_text(text)
            if not cleaned:
                continue
            if not full_review:
                full_review = cleaned
            yield {
                "text": cleaned,       # 单句文本（原有）
                "label": label,        # 单句标签（原有）
                "review_id": review_id,# 评论ID（原有）
                "full_review": full_review, # 整段评论（新增）
                "book_id": book_id      # 书籍ID（新增）
            }
# ========== MODIFIED END ==========


def stable_bucket(value: str, buckets: int = 10000) -> int:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % buckets


# ========== ADDED START ==========
# 新增核心函数：加载书籍元数据，构建 {best_book_id: original_title} 映射字典
def load_book_metadata(
    meta_path: Path,
    title_field: str,
    desc_field: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """加载goodreads_book_works.json，返回书籍ID到书名/摘要的映射"""
    book_titles: dict[str, str] = {}
    book_descs: dict[str, str] = {}
    print(f"Loading book metadata from {meta_path}...")
    for row in tqdm(iter_jsonl(meta_path), desc="Loading book metadata"):
        bk_id = str(row.get("best_book_id", "")).strip()
        if not bk_id:
            continue
        bk_title = str(row.get(title_field, "")).strip() if title_field else ""
        bk_desc = str(row.get(desc_field, "")).strip() if desc_field else ""
        if bk_title:
            book_titles[bk_id] = bk_title
        if bk_desc:
            book_descs[bk_id] = bk_desc
    print(f"Loaded {len(book_titles)} book titles and {len(book_descs)} book descriptions.")
    return book_titles, book_descs
# ========== ADDED END ==========


# ==========================================
# 新增功能：平衡采样逻辑（原有，未改动）
# ==========================================
def run_sampling_mode(
    args,
    book_titles_map: dict[str, str],
    book_descs_map: dict[str, str],
) -> None:
    print(f"=== Running Sampling Mode ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.sample_output}")
    print(f"Target Size: {args.sample_size}")
    
    spoiler_ids = set()
    if args.spoiler_input:
        spoiler_ids = load_spoiler_ids(
            Path(args.spoiler_input),
            args.id_field,
            args.spoiler_max_rows,
            args.chunksize
        )

    target_per_class = args.sample_size // 2
    count_pos = 0
    count_neg = 0
    buffer = []
    
    path = Path(args.input)
    
    with path.open("r", encoding="utf-8") as handle:
        pbar = tqdm(total=args.sample_size, desc="Sampling")
        
        for line in handle:
            if not line.strip():
                continue

            if count_pos >= target_per_class and count_neg >= target_per_class:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(row.get("review_sentences"), list):
                review_id = row.get(args.id_field)
                full_review = row.get("review_text", "").strip()
                if not full_review and row.get("review_sentences"):
                    full_review = " ".join(
                        [item[1] for item in row["review_sentences"] if len(item) >= 2 and item[1]]
                    )
                full_review = clean_text(full_review)
                book_id = str(row.get(args.book_id_field, "")).strip()
                book_title = book_titles_map.get(book_id, "Unknown Book")
                book_desc = book_descs_map.get(book_id, "")

                for text, label in iter_review_sentences(row):
                    if count_pos >= target_per_class and count_neg >= target_per_class:
                        break

                    cleaned_text = clean_text(text)
                    if not cleaned_text:
                        continue
                    if not full_review:
                        full_review = cleaned_text

                    output_row = {
                        "text": cleaned_text,
                        "review_sentence": cleaned_text,
                        "review": full_review,
                        "label": int(label),
                        "review_id": review_id,
                        "book_id": book_id,
                        "book_titles": book_title,
                        "book_description": book_desc,
                    }

                    if label == 1:
                        if count_pos < target_per_class:
                            buffer.append(output_row)
                            count_pos += 1
                            pbar.update(1)
                    else:
                        if count_neg < target_per_class:
                            buffer.append(output_row)
                            count_neg += 1
                            pbar.update(1)
            else:
                label = 0
                if args.spoiler_input:
                    rid = str(row.get(args.id_field, ""))
                    label = 1 if rid in spoiler_ids else 0
                elif args.label_field in row:
                    val = row[args.label_field]
                    if str(val).lower() in ["1", "spoiler", "true"]:
                        label = 1
                    else:
                        label = 0
                elif args.label_mode == "spoiler_tag":
                    label = label_from_spoiler_tags(row.get(args.text_field, ""))

                text = row.get(args.text_field, "")
                cleaned_text = clean_text(text)
                if not cleaned_text:
                    continue

                book_id = str(row.get(args.book_id_field, "")).strip()
                book_title = book_titles_map.get(book_id, "Unknown Book")
                book_desc = book_descs_map.get(book_id, "")
                review_id = row.get(args.id_field)

                output_row = {
                    "text": cleaned_text,
                    "review_sentence": cleaned_text,
                    "review": cleaned_text,
                    "label": int(label),
                    "review_id": review_id,
                    "book_id": book_id,
                    "book_titles": book_title,
                    "book_description": book_desc,
                }

                if label == 1:
                    if count_pos < target_per_class:
                        buffer.append(output_row)
                        count_pos += 1
                        pbar.update(1)
                else:
                    if count_neg < target_per_class:
                        buffer.append(output_row)
                        count_neg += 1
                        pbar.update(1)
        pbar.close()

    print(f"Sampling Done. Pos: {count_pos}, Neg: {count_neg}")
    random.shuffle(buffer)
    output_path = Path(args.sample_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_jsonl(buffer, output_path)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare spoiler dataset splits.")
    
    # --- 原有参数 ---
    parser.add_argument("--input", required=True, help="Path to input CSV/TSV/JSON/JSONL")
    parser.add_argument("--spoiler-input", help="Path to spoiler-only dataset (JSON/JSONL)")
    parser.add_argument("--id-field", default="review_id", help="Field used to match spoiler ids")
    parser.add_argument("--text-field", default="text", help="Column name for text")
    parser.add_argument("--label-field", default="label", help="Column name for label")
    parser.add_argument("--label-mode", choices=["column", "spoiler_tag", "constant"], default="column")
    parser.add_argument("--label-constant", type=int, default=0)
    
    parser.add_argument("--output-dir", default="data/splits", help="Output directory for full split")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    
    parser.add_argument("--inspect", action="store_true", help="Print detected fields and exit")
    parser.add_argument("--inspect-rows", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--spoiler-max-rows", type=int, default=None)
    
    parser.add_argument("--chunked", action="store_true", help="Process JSONL data in chunks")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--review-sentences", action="store_true")
    parser.add_argument("--balanced-split", action="store_true", help="Enable balanced class splits")
    parser.add_argument(
        "--split-sample-size",
        type=int,
        default=None,
        help="Total samples to keep across train/val/test when --balanced-split is enabled",
    )
    
    # --- 新增采样参数 ---
    parser.add_argument("--sample-mode", action="store_true", help="Enable balanced sampling mode (subset generation)")
    parser.add_argument("--sample-size", type=int, default=20000, help="Total number of samples to extract")
    parser.add_argument("--sample-output", default="data/subset/sampled.jsonl", help="Output path for sampled file")

    # ========== ADDED START ==========
    # 新增书籍元数据参数
    parser.add_argument("--book-meta-input", help="Path to book meta file (goodreads_book_works.json)")
    parser.add_argument("--book-id-field", default="book_id", help="Column name for book id in review data")
    parser.add_argument("--book-title-field", default="original_title", help="Title field in book meta")
    parser.add_argument(
        "--book-desc-field",
        default="",
        help="Description/abstract field in book meta (leave empty to skip)",
    )
    # ========== ADDED END ==========

    args = parser.parse_args()

    # ========== ADDED START ==========
    # 初始化书籍元数据映射：传入了路径才加载，不影响原有逻辑
    book_titles_map: dict[str, str] = {}
    book_descs_map: dict[str, str] = {}
    if args.book_meta_input:
        desc_field = args.book_desc_field if args.book_desc_field else None
        book_titles_map, book_descs_map = load_book_metadata(
            Path(args.book_meta_input),
            args.book_title_field,
            desc_field,
        )
    # ========== ADDED END ==========

    # 如果开启采样模式，直接跳转到采样逻辑
    if args.sample_mode:
        run_sampling_mode(args, book_titles_map, book_descs_map)
        return

    path = Path(args.input)

    if args.review_sentences:
        # ========== MODIFIED START ==========
        # 核心修改：分句模式下，写入【full_review + book_title】字段
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        test_path = output_dir / "test.jsonl"

        with train_path.open("w", encoding="utf-8") as train_handle, \
             val_path.open("w", encoding="utf-8") as val_handle, \
             test_path.open("w", encoding="utf-8") as test_handle:

            total = 0
            counts = {"train": 0, "val": 0, "test": 0}
            test_cut = int(args.test_size * 10000)
            val_cut = int((args.test_size + args.val_size) * 10000)

            if args.balanced_split:
                if args.split_sample_size is None or args.split_sample_size <= 0:
                    raise ValueError("--balanced-split requires --split-sample-size > 0")

                train_ratio = 1.0 - args.test_size - args.val_size
                if train_ratio <= 0:
                    raise ValueError("Invalid split ratios; train ratio must be > 0")

                train_target = int(args.split_sample_size * train_ratio)
                val_target = int(args.split_sample_size * args.val_size)
                test_target = int(args.split_sample_size * args.test_size)
                remainder = args.split_sample_size - (train_target + val_target + test_target)
                train_target += remainder

                targets = {
                    "train": train_target // 2,
                    "val": val_target // 2,
                    "test": test_target // 2,
                }
                if any(v == 0 for v in targets.values()):
                    raise ValueError("split sample size too small to balance classes")

                class_counts = {
                    "train": {"pos": 0, "neg": 0},
                    "val": {"pos": 0, "neg": 0},
                    "test": {"pos": 0, "neg": 0},
                }

                for row in iter_review_sentence_rows(path, args.max_rows, args.book_id_field):
                    if all(
                        class_counts[split]["pos"] >= targets[split]
                        and class_counts[split]["neg"] >= targets[split]
                        for split in ("train", "val", "test")
                    ):
                        break

                    total += 1
                    review_id = row.get("review_id")
                    bucket = stable_bucket(str(review_id))
                    if bucket < test_cut:
                        subset = "test"
                    elif bucket < val_cut:
                        subset = "val"
                    else:
                        subset = "train"

                    label = int(row["label"])
                    class_key = "pos" if label == 1 else "neg"
                    if class_counts[subset][class_key] >= targets[subset]:
                        continue

                    # 关联书籍名称：匹配不到则填充Unknown Book
                    book_id = row.get("book_id", "")
                    book_title = book_titles_map.get(book_id, "Unknown Book")
                    book_desc = book_descs_map.get(book_id, "")
                    # 构造最终行：包含单句、整段、书名、标签
                    output_row = {
                        "text": row["text"],
                        "review_sentence": row["text"],
                        "review": row["full_review"],
                        "label": label,
                        "review_id": review_id,
                        "book_id": book_id,
                        "book_titles": book_title,
                        "book_description": book_desc,
                    }

                    if subset == "test":
                        test_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    elif subset == "val":
                        val_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    else:
                        train_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

                    class_counts[subset][class_key] += 1
                    counts[subset] += 1
            else:
                for row in iter_review_sentence_rows(path, args.max_rows, args.book_id_field):
                    total += 1
                    review_id = row.get("review_id")
                    bucket = stable_bucket(str(review_id))
                    # 关联书籍名称：匹配不到则填充Unknown Book
                    book_id = row.get("book_id", "")
                    book_title = book_titles_map.get(book_id, "Unknown Book")
                    book_desc = book_descs_map.get(book_id, "")
                    # 构造最终行：包含单句、整段、书名、标签
                    output_row = {
                        "text": row["text"],
                        "review_sentence": row["text"],
                        "review": row["full_review"],
                        "label": row["label"],
                        "review_id": review_id,
                        "book_id": book_id,
                        "book_titles": book_title,
                        "book_description": book_desc,
                    }
                    # 按原有逻辑写入不同文件
                    if bucket < test_cut:
                        test_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        counts["test"] += 1
                    elif bucket < val_cut:
                        val_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        counts["val"] += 1
                    else:
                        train_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        counts["train"] += 1

            print(f"review_sentence_rows={total}")
            print(f"train={counts['train']} val={counts['val']} test={counts['test']}")
        return
        # ========== MODIFIED END ==========

    if args.chunked:
        if not args.spoiler_input:
            raise ValueError("--chunked requires --spoiler-input for labeling")

        spoiler_ids = load_spoiler_ids(
            Path(args.spoiler_input),
            args.id_field,
            args.spoiler_max_rows,
            args.chunksize,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        test_path = output_dir / "test.jsonl"

        with train_path.open("w", encoding="utf-8") as train_handle, \
             val_path.open("w", encoding="utf-8") as val_handle, \
             test_path.open("w", encoding="utf-8") as test_handle:

            total = 0
            counts = {"train": 0, "val": 0, "test": 0}
            test_cut = int(args.test_size * 10000)
            val_cut = int((args.test_size + args.val_size) * 10000)

            lines = path.suffix.lower() == ".jsonl" or _looks_like_jsonl(path)
            for chunk in pd.read_json(path, lines=lines, chunksize=args.chunksize):
                if args.max_rows is not None and total >= args.max_rows:
                    break

                if args.text_field not in chunk.columns and "review_text" in chunk.columns:
                    text_field = "review_text"
                else:
                    text_field = args.text_field

                if text_field not in chunk.columns:
                    raise ValueError("Input missing required text column")
                if args.id_field not in chunk.columns:
                    raise ValueError(f"Missing id field in main dataset: {args.id_field}")

                # ========== MODIFIED START ==========
                # 新增：保留book_id字段，用于关联书名
                if args.book_id_field not in chunk.columns:
                    chunk[args.book_id_field] = ""
                chunk = chunk[[text_field, args.id_field, args.book_id_field]].dropna()
                # ========== MODIFIED END ==========

                if chunk.empty:
                    continue

                total += len(chunk)

                ids = chunk[args.id_field].astype(str)
                labels = ids.isin(spoiler_ids).astype(int)
                texts = clean_text_series(chunk[text_field])
                keep = texts != ""
                
                # ========== MODIFIED START ==========
                # 新增：关联书籍名称 + 保留整段文本
                book_ids = chunk[args.book_id_field].astype(str)
                book_titles = book_ids.map(book_titles_map).fillna("Unknown Book")
                book_descs = book_ids.map(book_descs_map).fillna("")
                chunk = pd.DataFrame({
                    "text": texts[keep],
                    "review_sentence": texts[keep],
                    "review": texts[keep],  # 非分句模式下，text即为整段
                    "label": labels[keep],
                    "review_id": ids[keep],
                    "book_id": book_ids[keep],
                    "book_titles": book_titles[keep],
                    "book_description": book_descs[keep],
                })
                # ========== MODIFIED END ==========

                buckets = pd.util.hash_pandas_object(chunk["review_id"], index=False).values % 10000
                test_mask = buckets < test_cut
                val_mask = (buckets >= test_cut) & (buckets < val_cut)
                train_mask = buckets >= val_cut

                for subset, mask, handle in (
                    ("test", test_mask, test_handle),
                    ("val", val_mask, val_handle),
                    ("train", train_mask, train_handle),
                ):
                    if mask.any():
                        # ========== MODIFIED START ==========
                        # 写入新增的review + book字段
                        out_df = chunk.loc[
                            mask,
                            [
                                "text",
                                "review_sentence",
                                "review",
                                "label",
                                "review_id",
                                "book_id",
                                "book_titles",
                                "book_description",
                            ],
                        ]
                        # ========== MODIFIED END ==========
                        out_df.to_json(handle, orient="records", lines=True, force_ascii=False)
                        counts[subset] += len(out_df)

            print(f"chunked_rows={total}")
            print(f"train={counts['train']} val={counts['val']} test={counts['test']}")
        return

    # --- 内存加载模式 (针对小文件) ---
    df = load_table(path, max_rows=args.inspect_rows if args.inspect else args.max_rows)

    if args.inspect:
        print("Columns:", list(df.columns))
        print(df.head(args.inspect_rows).to_string(index=False))
        return

    if args.text_field not in df.columns and "review_text" in df.columns:
        args.text_field = "review_text"

    if args.text_field not in df.columns:
        raise ValueError("Input missing required text column")

    if args.spoiler_input:
        spoiler_df = load_table(Path(args.spoiler_input), max_rows=args.spoiler_max_rows)
        df = apply_spoiler_ids(df, spoiler_df, args.id_field, args.label_field)
        # ========== MODIFIED START ==========
        # 新增：保留book_id字段
        if args.book_id_field not in df.columns:
            df[args.book_id_field] = ""
        if args.id_field not in df.columns:
            raise ValueError(f"Missing id field in main dataset: {args.id_field}")
        df = df[[args.text_field, args.label_field, args.book_id_field, args.id_field]].dropna()
        # ========== MODIFIED END ==========
    else:
        if args.label_mode == "column":
            if args.label_field not in df.columns:
                raise ValueError("Input missing required label column")
            # ========== MODIFIED START ==========
            if args.book_id_field not in df.columns:
                df[args.book_id_field] = ""
            if args.id_field not in df.columns:
                raise ValueError(f"Missing id field in main dataset: {args.id_field}")
            df = df[[args.text_field, args.label_field, args.book_id_field, args.id_field]].dropna()
            # ========== MODIFIED END ==========
            df[args.label_field] = df[args.label_field].astype(int)
        else:
            # ========== MODIFIED START ==========
            if args.book_id_field not in df.columns:
                df[args.book_id_field] = ""
            if args.id_field not in df.columns:
                raise ValueError(f"Missing id field in main dataset: {args.id_field}")
            df = df[[args.text_field, args.book_id_field, args.id_field]].dropna()
            # ========== MODIFIED END ==========
            if args.label_mode == "spoiler_tag":
                df[args.label_field] = df[args.text_field].astype(str).map(label_from_spoiler_tags)
            elif args.label_mode == "constant":
                df[args.label_field] = int(args.label_constant)

    df[args.text_field] = df[args.text_field].astype(str).map(clean_text)
    df["text"] = df[args.text_field]
    
    # ========== MODIFIED START ==========
    # 核心修改：关联书籍名称 + 新增review字段
    df["review"] = df[args.text_field]  # 非分句模式下，text即为整段
    df["review_sentence"] = df[args.text_field]
    df["review_id"] = df[args.id_field]
    df["book_id"] = df[args.book_id_field]
    df["book_titles"] = df[args.book_id_field].astype(str).map(book_titles_map).fillna("Unknown Book")
    df["book_description"] = df[args.book_id_field].astype(str).map(book_descs_map).fillna("")
    # ========== MODIFIED END ==========

    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[args.label_field],
        random_state=42,
    )
    val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[args.label_field],
        random_state=42,
    )

    output_dir = Path(args.output_dir)
    # ========== MODIFIED START ==========
    # 写入包含 review + book 字段的完整数据
    output_columns = [
        "text",
        "review_sentence",
        "review",
        "label",
        "review_id",
        "book_id",
        "book_titles",
        "book_description",
    ]
    write_jsonl(train[output_columns].to_dict(orient="records"), output_dir / "train.jsonl")
    write_jsonl(val[output_columns].to_dict(orient="records"), output_dir / "val.jsonl")
    write_jsonl(test[output_columns].to_dict(orient="records"), output_dir / "test.jsonl")
    # ========== MODIFIED END ==========


if __name__ == "__main__":
    main()
