# import json
# import torch
# from typing import Union, List, Any
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# from .prompts import build_messages

# # 兼容 Fine-tuning v1 的 Raw Prompt 模板
# RAW_PROMPT_TEMPLATE = """
# Task: decide whether the review sentence contains a spoiler.
# You are given the review sentence (target), book title, and full review as context.
# Use the full review only for context; the label is about the review sentence.
# Definitions:
# - spoiler: reveals ending, key plot twist, core character fate, or mystery answer.
# - not spoiler: general opinions, vague praise, or non-essential details.

# Output JSON schema:
# {{"has_spoiler":0|1,"confidence":0-1,"spoiler_type":"movie"|"novel"|"none","key_spoiler_sentence":"..."|"none"}}

# Output requirements:
# - ONLY output a SINGLE JSON object.
# - Do NOT output any explanation, text, or code fences.
# - Start directly with '{{' and end with '}}'.

# Review sentence: {review_sentence}
# Book title: {book_title}
# Full review: {review_text}
# Output:
# """.strip()

# def _pick_field(data: dict, keys: tuple[str, ...]) -> str:
#     for key in keys:
#         value = data.get(key)
#         if value is None:
#             continue
#         value = str(value).strip()
#         if value:
#             return value
#     return ""

# class SpoilerClassifier:
#     def __init__(self, model_id: str, adapter_path: str = None):
#         print(f"正在加载模型: {model_id} (使用 4-bit 量化优化)...")
        
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#         )

#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
#         # Llama 3 可能没有默认的 pad_token，我们将其设为 eos_token
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
#         # 生成任务必须使用左填充，否则生成的 token 无法接续
#         self.tokenizer.padding_side = 'left'

#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             device_map="auto",
#             torch_dtype=torch.bfloat16
#         )

#         if adapter_path:
#             print(f"🔥 正在加载微调后的 LoRA Adapter: {adapter_path}")
#             self.model = PeftModel.from_pretrained(self.model, adapter_path)

#         self.model.eval()
#         print("模型加载完成，已准备好批量处理。")

#     def _build_raw_prompt(self, input_item: Union[str, dict]) -> str:
#         """
#         仅在 use_chat_template=False 时调用。
#         完全复刻 fine_tune.py 的逻辑。
#         """
#         if isinstance(input_item, str):
#             review_sentence = str(input_item).strip()
#             book_title = "Unknown Book"
#             review_text = review_sentence
#         else:
#             review_sentence = _pick_field(input_item, ("review_sentence", "text", "sentence"))
#             book_title = _pick_field(input_item, ("book_titles", "book_title", "title", "book"))
#             review_text = _pick_field(input_item, ("review", "full_review", "review_text"))
#             if not review_text:
#                 review_text = review_sentence
#             if not book_title:
#                 book_title = "Unknown Book"
#             if not review_sentence:
#                 review_sentence = review_text

#         return RAW_PROMPT_TEMPLATE.format(
#             review_sentence=review_sentence,
#             book_title=book_title,
#             review_text=review_text
#         )

#     def predict_batch(
#         self,
#         inputs: List[Union[str, dict[str, Any]]], 
#         content_type: str = "book",
#         use_chat_template: bool = True  # <-- 新增开关，默认保持 True (不影响 Benchmark)
#         ) -> List[str]:
#         """
#         批量推理函数
#         :param inputs: 文本或结构化输入列表
#         :return: 对应的结果列表
#         """
#         # 1. 批量构建 Messages
#         # 先将每条文本转换为 Llama 3 的 prompt 格式（纯文本形式）
#         prompts = []
#         if use_chat_template:
#             # 分支 A: 走 Chat Template (Benchmark 用)
#             for input_item in inputs:
#                 msgs = build_messages(input_item, content_type)
#                 # apply_chat_template(tokenize=False) 只做格式化，不转 ID，方便后面统一 pad
#                 prompt_str = self.tokenizer.apply_chat_template(
#                     msgs, add_generation_prompt=True, tokenize=False
#                 )
#                 prompts.append(prompt_str)
#         else:
#             # 分支 B: 走 Raw Prompt (Fine-tuned v1 用)
#             # 未来如果有 prompt 压缩，可以在这里扩展逻辑，或者修改 _build_raw_prompt
#             prompts = [self._build_raw_prompt(item) for item in inputs]

#         # 2. 批量 Tokenize & Padding
#         # return_tensors='pt' 会返回 PyTorch 张量
#         # padding=True 会自动 pad 到当前 batch 中最长的序列长度
#         inputs = self.tokenizer(
#             prompts, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=4096 # 这里的长度限制可以根据显存情况调整
#         ).to(self.model.device)

#         # 3. 批量生成 (Batch Generation)
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 do_sample=False, 
#                 pad_token_id=self.tokenizer.pad_token_id
#             )

#         # 4. 批量解码
#         # generate 返回的结果包含输入部分，我们需要截断只取新生成的部分
#         input_len = inputs.input_ids.shape[1]
#         new_tokens = generated_ids[:, input_len:]
        
#         responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
#         return responses

# # 全局单例
# _classifier = None

# import re

# def extract_json_substring(text: str) -> str | None:
#     """
#     从任意文本中提取第一段形如 {...} 的 JSON 子串（支持嵌套花括号）。
#     """
#     start = text.find("{")
#     if start == -1:
#         return None

#     depth = 0
#     for i, ch in enumerate(text[start:], start=start):
#         if ch == "{":
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0:
#                 return text[start:i+1]
#     return None


# def _get_classifier(model_id: str = None, adapter_path: str = None) -> SpoilerClassifier:
#     global _classifier
#     if _classifier is None:
#         # 请确保此处路径正确
#         default_model = "/mnt/data/projects/spoiler-agent/models/Meta-Llama-3-8B-Instruct"
#         target_model = model_id or default_model
#         _classifier = SpoilerClassifier(target_model, adapter_path=adapter_path)
#     return _classifier

# def classify_text(
#     input_data: Union[str, dict[str, Any], List[Union[str, dict[str, Any]]]],
#     content_type: str = "book", 
#     adapter_path: str = None,
#     use_chat_template: bool = True # <-- 接收参数并传递
# ) -> Union[dict, List[dict]]:
#     """
#     统一入口：支持单条字符串输入，也支持字符串列表输入（Batch）。
#     """
#     classifier = _get_classifier(adapter_path=adapter_path)
    
#     # 判断是否为 Batch 模式
#     is_batch = isinstance(input_data, list)
#     inputs = input_data if is_batch else [input_data]
    
#     # 执行批量推理
#     raw_responses = classifier.predict_batch(
#         inputs, 
#         content_type=content_type,
#         use_chat_template=use_chat_template 
#     )
    
#     # 批量解析 JSON
#     results = []
#     # for raw in raw_responses:
#     #     try:
#     #         # 尝试找到 JSON 子串（简单的容错处理）
#     #         clean_json = raw.strip()
#     #         if "```json" in clean_json:
#     #             clean_json = clean_json.split("```json")[1].split("```")[0].strip()
#     #         elif "```" in clean_json:
#     #             clean_json = clean_json.split("```")[1].strip()
                
#     #         results.append(json.loads(clean_json))
#     #     except Exception as e:
#     #         # 解析失败保留原始文本方便 Debug
#     #         results.append({
#     #             "has_spoiler": 0,  # 默认值
#     #             "spoiler_type": "error",
#     #             "error_msg": str(e),
#     #             "raw_output": raw
#     #         })
#     #1.4修改：尝试提取第一个合法的json
#     for raw in raw_responses:
#         try:
#             clean = raw.strip()

#             # 先尝试从整段中抓一个 JSON 子串
#             candidate = extract_json_substring(clean)
#             if candidate is None:
#                 raise ValueError("no json object found")

#             results.append(json.loads(candidate))
#         except Exception as e:
#             results.append({
#                 "has_spoiler": 0,
#                 "confidence": 0.0,
#                 "spoiler_type": "error",
#                 "key_spoiler_sentence": "none",
#                 "error_msg": str(e),
#                 "raw_output": raw,
#             })

            
#     # 如果输入是单条，返回单个 dict；如果是 list，返回 list[dict]
#     return results if is_batch else results[0]

# 12.30 更新 (支持 LoRA)
import json
import torch
from typing import Union, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel 
from .prompts import build_messages
from llmlingua import PromptCompressor

class SpoilerClassifier:
    # === 增加 adapter_path 参数，默认为 None ===
    def __init__(self, model_id: str, adapter_path: str = None, use_compression: bool = False, compression_rate: float = 0.5):
        print(f"正在加载 Base 模型: {model_id} (使用 4-bit 量化优化)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # === 设置 Pad Token ===
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # === 设置 Left Padding (生成任务必需) ===
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
            # attn_implementation="flash_attention_2"  # <-- 可选优化，视显卡支持情况而定
        )

        # === 如果提供了 adapter_path，加载 LoRA 权重 ===
        if adapter_path:
            print(f"🔥 正在加载微调后的 LoRA Adapter: {adapter_path}")
            # 将 LoRA 权重合并到 Base Model 上 (虽然是推理模式，但 PeftModel 会自动处理)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.use_compression = use_compression
        self.compression_rate = compression_rate

        # if self.use_compression:
        #     print(f"🗜️ 正在加载 LLMLingua-2 压缩模型 (保留比例: {compression_rate})...")
        #     self.compressor = PromptCompressor(
        #         model_name="microsoft/llmlingua-2-bert-base-multilingual-cased",
        #         use_llmlingua2=True,
        #         device_map="cpu"  # 强制放在 CPU 上，节省显存给 Llama-3
        #     )
        if self.use_compression:
            # === 关键修改点：指向你的本地绝对路径 ===
            # 假设你的项目根目录是 /mnt/data/projects/spoiler-agent/
            local_model_path = "/mnt/data/projects/spoiler-agent/models/llmlingua-2-bert"
            
            print(f"🗜️ 正在从本地加载压缩模型: {local_model_path}")
            
            # 强制 device_map="cpu" 以节省显存给 Llama-3
            self.compressor = PromptCompressor(
                model_name=local_model_path,  # <--- 这里改成变量名 local_model_path
                use_llmlingua2=True,
                device_map="cpu" 
            )

        self.model.eval()
        print("模型加载完成，已准备好批量处理。")

    def predict_batch(self, inputs: List[Union[str, dict[str, Any]]], content_type: str = "book") -> List[str]:
        """
        批量推理函数
        :param inputs: 文本或结构化输入列表
        :return: 对应的结果列表
        """
        # === [新增] 压缩逻辑：在构建 Prompt 前对长文本进行压缩 ===
        if self.use_compression and hasattr(self, 'compressor'):
            compressed_inputs = []
            # 这里的 inputs 可能是字符串列表，也可能是字典列表
            for item in inputs:
                # 提取原始文本
                original_text = ""
                if isinstance(item, str):
                    original_text = item
                elif isinstance(item, dict):
                    # 优先取 review，其次取 text
                    original_text = item.get('review') or item.get('text') or ""
                
                # 如果文本太短（<200字符），跳过压缩以防出错
                if len(original_text) > 200:
                    try:
                        # 执行压缩
                        result = self.compressor.compress_prompt(
                            original_text,
                            rate=self.compression_rate,
                            force_tokens=["\n", ".", "!"] # 保留基本标点防止句子粘连
                        )
                        compressed_text = result['compressed_prompt']
                    except Exception as e:
                        print(f"⚠️ 压缩失败，使用原文本: {e}")
                        compressed_text = original_text
                else:
                    compressed_text = original_text
                
                # 将压缩后的文本回填
                if isinstance(item, str):
                    compressed_inputs.append(compressed_text)
                elif isinstance(item, dict):
                    new_item = item.copy()
                    # 同时更新 text 和 review 字段，确保 build_messages 能取到压缩后的内容
                    new_item['text'] = compressed_text
                    new_item['review'] = compressed_text
                    compressed_inputs.append(new_item)
            
            # 使用处理后的 inputs 替换原 inputs
            inputs = compressed_inputs

        # 1. 批量构建 Messages
        prompts = []
        for input_item in inputs:
            msgs = build_messages(input_item, content_type)
            prompt_str = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_str)

        # 2. 批量 Tokenize & Padding
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4096 
        ).to(self.model.device)

        # 定义停止符 (Llama 3 的 eos_token_id 和 <|eot_id|>)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 批量生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256, 
                do_sample=False, 
                pad_token_id=self.tokenizer.pad_token_id,
                # === 传入停止符 ===
                eos_token_id=terminators 
            )

        # 4. 批量解码
        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_len:]
        
        responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return responses

# 全局单例
_classifier = None

# === _get_classifier 支持传入参数 (可选) ===
def _get_classifier(model_id: str = None, adapter_path: str = None, use_compression: bool = False, compression_rate: float = 0.5) -> SpoilerClassifier:
    global _classifier
    if _classifier is None:
        # 默认路径
        default_model = "/mnt/data/projects/spoiler-agent/models/Meta-Llama-3-8B-Instruct"
        target_model = model_id or default_model
        
        _classifier = SpoilerClassifier(
            target_model, 
            adapter_path=adapter_path,
            use_compression=use_compression,
            compression_rate=compression_rate
            )
    return _classifier

def classify_text(
    input_data: Union[str, dict[str, Any], List[Union[str, dict[str, Any]]]],
    content_type: str = "book",
    adapter_path: str = None,
    use_compression: bool = False,
    compression_rate: float = 0.5
) -> Union[dict, List[dict]]:
    """
    统一入口：支持单条字符串输入，也支持字符串列表输入（Batch）。
    """
    # 注意：这里调用 _get_classifier() 不传参，意味着它会使用第一次初始化时的实例
    # 如果 run_inference.py 里手动初始化了 _classifier，这里就会直接用那个带 LoRA 的实例
    classifier = _get_classifier(
        adapter_path=adapter_path,
        use_compression=use_compression,
        compression_rate=compression_rate
        )
    
    # 判断是否为 Batch 模式
    is_batch = isinstance(input_data, list)
    inputs = input_data if is_batch else [input_data]
    
    # 执行批量推理
    raw_responses = classifier.predict_batch(inputs, content_type)
    
    # 批量解析 JSON
    results = []
    for raw in raw_responses:
        try:
            # 尝试找到 JSON 子串
            clean_json = raw.strip()
            # 处理 Markdown 代码块
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].strip()
            
            # 尝试找到最外层的花括号 (处理一些非标准输出)
            start = clean_json.find('{')
            end = clean_json.rfind('}')
            if start != -1 and end != -1:
                clean_json = clean_json[start:end+1]

            results.append(json.loads(clean_json))
        except Exception as e:
            # 解析失败保留原始文本方便 Debug
            results.append({
                "has_spoiler": 0,  # 默认无剧透，保证系统可用性
                "spoiler_type": "error",
                "error_msg": str(e),
                "raw_output": raw
            })
            
    # 如果输入是单条，返回单个 dict；如果是 list，返回 list[dict]
    return results if is_batch else results[0]