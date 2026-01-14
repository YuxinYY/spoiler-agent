import os
import traceback
from llmlingua import PromptCompressor

# 1. 本地模型路径（直接传给model_name）
LOCAL_MODEL_PATH = "/mnt/data/projects/spoiler-agent/models/llmlingua-2-bert"

print(f"Checking model path: {LOCAL_MODEL_PATH}")
if not os.path.exists(LOCAL_MODEL_PATH):
    print("❌ Error: Path does not exist!")
    exit(1)
else:
    print("✅ Path exists. Files inside:")
    print(os.listdir(LOCAL_MODEL_PATH))

print("\nAttempting to load compressor...")

try:
    # 2. 初始化（仅保留0.2.2支持的3个核心参数）
    compressor = PromptCompressor(
        model_name=LOCAL_MODEL_PATH,  # 本地路径直接传
        use_llmlingua2=True,         # 必须开启LLMLingua-2
        device_map="cpu"             # 节省显存给主模型
    )
    print("✅ Model loaded successfully!")

    # 3. 测试文本（中英混合，适配会议/书评场景）
    test_text = "This is a very long book review about Harry Potter. 这是一篇关于哈利波特的超长书评，包含大量非关键细节，比如主角的日常、配角的对话等。" * 10
    print(f"\nTest text length: {len(test_text)}")
    
    print("Compressing...")
    # 核心：仅用0.2.2支持的参数，靠force_tokens和sentence_level规避错误
    result = compressor.compress_prompt(
        test_text,
        rate=0.5,
        force_tokens=["\n", ".", "!", "。", "？", "，", ",", ";", "；", "?", ":", " "],  # 补充空格，增强词分割
        # sentence_level=False,  # 关键：彻底绕过未实现的词边界判断
        # max_token=512
    )
    
    print("✅ Compression Success!")
    print(f"Original length: {len(test_text)}")
    print(f"Compressed length: {len(result['compressed_prompt'])}")
    print(f"Compressed text: {result['compressed_prompt'][:100]}...")

except Exception:
    print("\n❌ CRITICAL ERROR CAUGHT:")
    traceback.print_exc()