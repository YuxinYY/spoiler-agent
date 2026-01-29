# spoiler-agent
An LLM based spoiler detector.

## Available Demo
Explore the live demo and how the system behaves on real reviews.

- Live demo: https://huggingface.co/spaces/Yoosine/spoiler-detector.

## Demo
Paste a review URL and the app returns a processed version of the review. Spoiler spans are hidden, while the original wording remains readable.

### How to use the demo
- Paste the review URL you want to check.
- Submit and wait for the processed version.
- Review the output with spoiler text covered.

## Model
The detector is powered by fine-tuned Llama 3 8B.

- Fine-tuning: QLoRA with loss masking and long-text grouping.
- Performance: F1 improved from 0.71 to 0.96.

## Installation & Usage
### Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Usage
Single text inference:
```bash
python scripts\run_inference.py --text "The hero dies at the end to save the world." --content-type book
```

Batch inference (JSONL files):
```bash
python scripts\run_inference.py --input data\splits\test.jsonl --output outputs\predictions.jsonl --text-field text
```



<!-- ## Detailed Workflow Notes (Chinese) -->

<!-- 基于LLM的剧透内容分类完整工作流（高可行性版）

核心目标：让论坛Agent能调用LLM精准识别小说/电影相关文本是否含剧透，输出结构化结果供网站做“隐藏/打标签”处理。工作流优先级：可行性 > 效果 > 复杂度，全程采用成熟工具（LangChain、Pydantic）和轻量化方案，降低开发门槛。

一、前置准备：数据与标注（基础且关键，决定后续效果）

核心逻辑：LLM微调需高质量标注数据，优先用“公开数据集+少量自有标注”组合，减少标注成本。

1. 数据来源（低成本获取）

- 公开数据集：优先用Kaggle的Goodreads Spoilers数据集（含137万条书评，标注“是否含剧透”及剧透内容），覆盖小说类剧透场景；电影类可补充爬取豆瓣影评（用“剧透预警”标签筛选正例，无标签且仅谈观感的为负例）。

- 自有数据：从目标论坛爬取近3个月小说/电影相关文本（1000-2000条即可），用于后续微调适配论坛话术风格。

2. 标注规范（统一标准，避免歧义）

参考公开数据集标注逻辑，制定极简标注规则（2人标注+交叉校验，确保一致性）：

- 正例（含剧透）：明确提及核心情节（如“主角最终牺牲”）、关键反转（如“反派是主角父亲”）、结局（如“男女主没在一起”）、核心谜题答案（如“凶手是管家”）。

- 负例（无剧透）：仅谈观感（如“这部电影太好哭了”）、无关细节（如“画面质感拉满”）、模糊评价（如“剧情反转超多”）。

- 校验标准：标注者间Cohen's Kappa系数≥0.7（一致性合格），分歧样本由第三人仲裁。

3. 数据预处理（适配LLM输入）

- 清洗：删除URL、特殊符号、表情；中文文本用Jieba分词（可选，开源模型需，API模型无需）。

- 划分：按7:1:2拆分训练集、验证集、测试集（分层抽样，确保各集剧透比例一致）。

- 格式：最终转为“文本+标签”对（如{"text":"主角最后死了","label":1}，1=有剧透，0=无剧透）。

二、核心环节1：Schema Definition（定义结构化输出，适配网站Agent）

目的：让LLM输出固定格式结果，避免自由文本，方便网站Agent直接解析（如“看到label=1就隐藏文本”）。采用Pydantic定义Schema（LangChain可直接关联，自动校验格式）。

1. 基础Schema（必选，满足核心需求）

from pydantic import BaseModel, Field

class SpoilerClassification(BaseModel):
    # 核心标签：是否含剧透（1=是，0=否）
    has_spoiler: int = Field(..., enum=[0, 1], description="文本是否包含剧透，仅允许输出0或1")
    # 置信度：0-1之间，供网站设置阈值（如置信度>0.8才隐藏）
    confidence: float = Field(..., ge=0.0, le=1.0, description="分类结果的置信度，保留2位小数")
    # 剧透类型：细化场景（可选，提升实用性）
    spoiler_type: str = Field(..., enum=["电影", "小说", "无"], description="剧透所属类型，无剧透时填“无”")
    # 关键剧透句：提取核心剧透内容（方便用户查看“隐藏理由”）
    key_spoiler_sentence: str = Field(..., description="若含剧透，提取1-2句核心剧透句；无剧透则填“无”")

2. Schema核心作用

- 格式约束：LLM必须输出指定字段，且has_spoiler仅能是0/1，避免无效值（如“可能有剧透”）。

- 适配Agent：网站Agent可直接解析JSON格式结果，无需额外文本处理（如“if result.has_spoiler == 1: 隐藏文本”）。

- 自动校验：LangChain会校验输出是否符合Schema，不合格则抛出错误，方便调试。

三、核心环节2：大模型选型与零/少样本调用（快速验证可行性）

优先级：先通过“零/少样本调用”验证任务可行性，再决定是否微调（避免盲目微调浪费资源）。推荐2类模型（按需选择，均支持结构化输出）：

1. 模型选型（按成本/算力分层）

模型类型

推荐模型

优势

适用场景

API模型（零算力门槛）

GPT-3.5 Turbo、通义千问Plus

无需本地算力，LangChain直接调用，结构化输出稳定

中小企业/无算力团队，快速落地

开源模型（低成本部署）

Llama 3（8B）

免费商用，部署后无调用成本

有基础算力（单卡GPU），长期使用

2. 零/少样本调用流程（用LangChain简化开发）

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # API模型；开源模型用langchain_community的对应类

# 1. 初始化模型（API模型示例，开源模型类似，需指定本地路径）
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0  # 固定为0，确保输出稳定
)

# 2. 绑定Schema，强制结构化输出
structured_llm = llm.with_structured_output(schema=SpoilerClassification)

# 3. 构建Prompt（少样本示例：给2个例子，帮助模型理解任务）
prompt = ChatPromptTemplate.from_template("""
任务：判断以下文本是否含电影/小说剧透，严格按要求输出结果。
定义：
- 有剧透：提及结局、核心反转、关键人物命运、核心谜题答案
- 无剧透：仅谈观感、无关细节、模糊评价

示例1：
文本：《流浪地球2》里刘培强最后牺牲了，只为保护地球
输出：{"has_spoiler":1,"confidence":0.98,"spoiler_type":"电影","key_spoiler_sentence":"《流浪地球2》里刘培强最后牺牲了"}

示例2：
文本：《三体》写得太精彩了，推荐大家阅读
输出：{"has_spoiler":0,"confidence":0.99,"spoiler_type":"无","key_spoiler_sentence":"无"}

待判断文本：{input_text}
""")

# 4. 构建调用链并执行
chain = prompt | structured_llm
result = chain.invoke({"input_text": "《甄嬛传》最后甄嬛成为了太后"})

# 5. 输出结果（可直接给网站Agent）
print(result.model_dump())
# 输出示例：{"has_spoiler":1,"confidence":0.99,"spoiler_type":"电影","key_spoiler_sentence":"《甄嬛传》最后甄嬛成为了太后"}
"

3. 可行性验证标准

用测试集（200条文本）评估，若满足以下条件，说明无需微调即可初步落地：

- 核心指标：F1-Score≥0.8（平衡精确率和召回率，避免“漏判剧透”或“误判无剧透”）。

- 格式稳定性：95%以上输出符合Schema，无无效值。

- 适配性：论坛常见话术（如“谁懂啊！XX最后居然黑化了”）能准确识别。

四、核心环节3：Fine-Tune（仅零/少样本效果不佳时执行）

若验证后效果不达标（如F1-Score<0.8），需微调模型。核心优化思路：跳出“仅调超参”的浅层微调，采用“任务感知+领域自适应”的深度微调策略，结合进阶轻量化技术，既保证可行性，又形成核心技术壁垒。核心目标：让模型精准捕捉剧透任务的语义特征（如核心情节泄露、反转暗示），同时适配论坛文本风格。

1. 微调数据准备（重点：适配微调格式）

将标注数据转为“Prompt-Completion”格式（API模型）或“文本-标签”格式（开源模型），示例如下：

- API模型（如GPT-3.5）：JSONL格式，每行为一条数据
        {"prompt":"判断文本是否含剧透：《庆余年》范闲其实是叶轻眉的儿子","completion":"{\"has_spoiler\":1,\"confidence\":0.99,\"spoiler_type\":\"小说\",\"key_spoiler_sentence\":\"《庆余年》范闲其实是叶轻眉的儿子\"}"}
{"prompt":"判断文本是否含剧透：《哈利波特》画面很震撼","completion":"{\"has_spoiler\":0,\"confidence\":0.99,\"spoiler_type\":\"无\",\"key_spoiler_sentence\":\"无\"}"}

- 开源模型（如Llama 3）：用JSON格式，包含text和label字段（适配Hugging Face的Trainer API）。

2. 进阶微调技术与实施步骤（核心提升点，兼顾可行性与深度）

优先采用“领域自适应预训练+QLoRA微调+判别式优化”的组合方案，相比基础LoRA微调，能提升10-15%的任务适配度，且无需额外算力成本。以下为详细步骤（以开源模型为例，API模型可迁移核心思路）：

- 工具选型：采用Hugging Face Transformers+Peft（实现QLoRA）+Accelerate（分布式训练适配），开源免费且生态成熟，无需手动搭建复杂框架。

- 核心步骤：
领域自适应预训练（关键一步，区别于通用微调）：
- 数据：爬取10万条无标注的论坛影视/小说评论（仅需文本，无需标注），补充5万条公开影视小说语料。
- 任务：采用“掩码语言建模（MLM）+句子关系预测（SRP）”双任务预训练，让模型先学习领域语义（如论坛网络用语、影视小说专属术语）。
- 意义：避免模型用通用语义理解剧透，提升对“隐晦剧透”（如“XX和XX居然是这种关系”）的识别能力。

- QLoRA轻量化微调（进阶于基础LoRA）：
- 优势：在4-bit量化基础上实现微调，显存占用降低60%，单卡GPU即可训练7B模型，且效果不劣于全参数微调。
- 超参数设计（非经验值，结合任务优化）：
学习率：采用“分层学习率”（判别式微调思路），模型底层（词嵌入层）用1e-5，中层用2e-5，顶层（分类层）用3e-5，避免底层通用语义被破坏。

- 训练轮数：采用“早停+梯度累积”，以验证集F1-Score为指标，连续3轮无提升则停止，数据量<1万条时，梯度累积步数设为4（等效提升batch size）。

- 正则化：加入“标签平滑”（平滑系数0.1），解决剧透样本不平衡导致的过拟合，提升模型泛化能力。

任务感知损失函数设计（核心创新点）：
- 基础损失：交叉熵损失（负责二分类）。
- 新增损失：实体级对比损失，将文本中的“核心角色、关键事件”（如“甄嬛”“刘培强牺牲”）作为实体，计算剧透句与非剧透句的实体语义距离，强化模型对核心信息的敏感度。
- 组合：总损失=0.7*交叉熵损失+0.3*实体级对比损失，让模型不仅判断“是否剧透”，还能理解“为何是剧透”。

模型融合与验证：训练2个不同基座模型（Llama 3-8B），采用“加权投票”融合结果，提升稳定性。

- 工具：直接用OpenAI/阿里云官方微调工具（无需手动搭环境）。

- 步骤：
       

  1. 上传微调数据：按平台要求上传JSONL文件，平台自动校验格式。

  2. 设置超参数（按经验值，无需复杂调试）：
           

    - 学习率：2e-5（通用最优值，避免过拟合）。

    - 训练轮数（epoch）：3-5（数据量<1万条时，3轮足够）。

    - 批量大小（batch size）：16（API平台默认，无需修改）。

  3. 启动微调：平台自动训练，耗时约0.5-2小时（视数据量而定）。

  4. 获取微调后模型：训练完成后，平台返回模型ID，直接替换调用环节的模型名即可。

3. 微调后验证（关键：确认效果提升）

用同一测试集评估，核心看4点（体现技术提升）：① F1-Score≥0.9（较基础微调提升5个百分点）；② 隐晦剧透识别准确率≥85%（新增指标，区分于基础微调）；③ 对抗测试通过率≥80%（测试用谐音、缩写剧透文本，如“甄huan最后成了太hou”）；④ 格式稳定性≥99%。若未达标，优先优化预训练数据质量（补充隐晦剧透样本），而非单纯增加标注数据。

五、核心环节4：输出与Agent集成（落地最后一步）

目标：让网站Agent能快速调用模型，拿到结果后执行“隐藏/打标签”逻辑，优先采用HTTP接口形式（通用性最强）。

1. 封装为HTTP接口（用FastAPI，极简开发）

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. 初始化FastAPI应用
app = FastAPI(title="剧透识别API")

# 2. 定义输入模型（Agent传入的参数）
class SpoilerInput(BaseModel):
    text: str = Field(description="待识别的论坛文本")
    content_type: str = Field(enum=["电影", "小说"], description="文本所属类型")

# 3. 加载微调后的模型（复用前面的chain）
# （此处省略模型初始化和chain构建代码，与环节三一致）

# 4. 定义API接口
@app.post("/predict_spoiler")
def predict_spoiler(input_data: SpoilerInput):
    # 调用模型获取结果
    result = chain.invoke({"input_text": input_data.text})
    # 补充返回内容类型（方便Agent细化处理）
    result_dict = result.model_dump()
    result_dict["content_type"] = input_data.content_type
    # 返回结构化结果
    return {
        "code": 200,
        "message": "success",
        "data": result_dict
    }

# 5. 启动服务（端口：8000）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"

2. 网站Agent调用逻辑（极简示例）

import requests

# Agent传入待识别文本
def check_spoiler(text, content_type):
    url = "http://你的服务器IP:8000/predict_spoiler"
    headers = {"Content-Type": "application/json"}
    data = {"text": text, "content_type": content_type}
    response = requests.post(url, json=data).json()
    
    # 核心逻辑：根据结果做处理
    if response["code"] == 200:
        result = response["data"]
        # 置信度>0.8且含剧透，隐藏文本
        if result["has_spoiler"] == 1 and result["confidence"] > 0.8:
            return f"[剧透隐藏] 理由：{result['key_spoiler_sentence']}"
        # 否则显示原文，可选打“无剧透”标签
        else:
            return f"[无剧透] {text}"
    else:
        return text  # 接口异常时显示原文，避免影响用户体验
"

3. 输出结果说明

Agent最终拿到的结果含明确逻辑，可直接落地：

- 含剧透（置信度>0.8）：返回“[剧透隐藏] 理由：XXX”，网站隐藏原文，显示提示。

- 无剧透/低置信度：返回“[无剧透] 原文”，正常显示。

六、项目深度提升：跳出微调，打造核心竞争力

除进阶微调技术外，从“场景适配、任务深化、系统设计”三个维度提升项目级别，形成差异化核心贡献：

1. 任务深化：从“二分类”到“细粒度+多任务”

- 细粒度剧透分级（提升实用性与技术深度）：
- 扩展标签体系：0=无剧透，1=轻度剧透（提及非核心情节，如“XX前期很虐”），2=中度剧透（提及关键情节但不涉及结局，如“XX背叛了主角”），3=重度剧透（提及结局/核心反转，如“甄嬛最终成为太后”）。
- 价值：网站可根据分级处理（轻度打标签、重度隐藏），提升用户体验，区别于简单“隐藏/显示”的初级方案。

- 多任务联合学习（提升模型泛化能力）：
- 新增关联任务：“剧透文本溯源”（识别剧透内容来自哪部作品）、“情感极性判断”（判断剧透是否带负面引导）。
- 实现：共享底层预训练参数，顶层设置多任务头，训练时同步优化多任务损失，让模型同时具备“识别+溯源+情感判断”能力，适配论坛复杂场景。

2. 场景适配：针对论坛场景的鲁棒性优化（工程化核心贡献）

- 对抗训练（抵御剧透规避手段）：
- 生成对抗样本：对标注数据进行“同义替换”“谐音改写”“分段隐藏”（如“甄→真，嬛→环，最后成了太hou”“主角最后…（省略）…牺牲了”），构建对抗训练集。
- 训练策略：采用“FGSM对抗训练”，在训练中动态加入对抗样本，提升模型对规避性剧透的识别能力，这是纯调参方案无法实现的。

- 实时性优化（适配论坛高并发）：
- 模型蒸馏：用微调后的7B大模型作为教师模型，蒸馏到1.3B小模型（如Phi-2、Llama 3.2 1B），推理速度提升3倍，显存占用降低70%。
- 缓存策略：针对高频重复文本（如热门作品的经典剧透），采用“Redis+本地双重缓存”，缓存命中率≥30%，降低服务器压力。

3. 可解释性与交互设计（提升项目落地价值）

- 可解释性模块（区别于“黑盒模型”）：
- 实现：结合“注意力可视化+关键短语提取”，输出模型判断剧透的依据（如“判定为重度剧透，关键依据：‘甄嬛最终成为太后’（提及结局）”）。
- 价值：方便网站运营者审核，也让用户理解“为何被判定为剧透”，降低投诉率。

- 用户反馈闭环（打造可持续迭代系统）：
- 设计“用户申诉接口”：用户可对判定结果申诉（如“此内容不是剧透”），申诉数据自动标记为“高价值样本”，定期回流到训练集进行增量微调。
- 迭代机制：每月自动统计误判案例，分析误判原因（如新增作品的剧透、新规避手段），更新预训练数据与微调策略，形成“模型-用户-迭代”闭环。

4. 多模态扩展（预留技术升级空间）：

- 适配论坛图文内容：新增图片剧透识别分支（如电影截图、小说情节插画），采用“CLIP模型+微调后的文本模型”跨模态融合，识别图片中的剧透信息（如结局场景截图）。
- 价值：覆盖论坛多模态内容，提升项目的全面性，区别于纯文本识别的初级方案。

1. 成本控制

- 数据成本：优先用公开数据集，自有数据标注控制在2000条内（人工标注约2-3天）。

- 算力成本：API模型微调1000条数据约50-200元；开源模型用单卡GPU（如3090）微调，电费可忽略。

2. 效果优化（快速迭代）

- 错误分析：定期收集误判样本（如“把‘剧情反转’误判为剧透”），补充到标注数据中微调。

- Prompt优化：若微调成本高，可增加Prompt中的示例数量（如从2个增至5个），提升零样本效果。

3. 风险控制

- 接口稳定性：加缓存机制（如Redis），相同文本1小时内不重复调用模型。

- 格式异常：增加异常捕获逻辑，若模型输出不符合Schema，默认按“无剧透”处理，避免网站崩溃。

七、工作流总结与核心贡献（突出差异化价值）

优化后的工作流核心贡献的不再是“调参”，而是“任务感知的微调策略+论坛场景深度适配+可持续迭代系统”，落地步骤更具技术深度：

1. 1-4天：数据准备（含10万条领域无标注数据+2000条细粒度标注数据）+ Schema扩展（支持分级剧透）。

2. 3-5天：领域自适应预训练+QLoRA进阶微调（含损失函数设计）。

3. 2天：对抗训练+模型蒸馏（兼顾效果与实时性）。

4. 2天：封装API（集成可解释性模块）+ 与Agent集成（含用户申诉接口）。

5. 持续迭代：每月基于用户反馈与新场景数据，进行增量预训练与微调。

核心价值：从“能用”升级为“好用、抗造、可迭代”，区别于纯调参的初级方案，形成技术壁垒与场景竞争力。

1. 1-3天：准备数据（爬取+标注），定义Schema。

2. 1天：搭建零/少样本调用链，验证可行性。

3. 1-2天：若效果不佳，执行轻量化微调并验证。

4. 1天：封装API接口，与网站Agent集成。

5. 持续迭代：收集用户反馈，补充数据微调模型。


## Implementation Scaffold

This repo now includes runnable scaffolding for data prep, local Llama inference, evaluation, fine-tuning, and an API.

### Layout
- `src/spoiler_agent/`: schema, prompts, inference, evaluation, API
- `scripts/`: data prep, inference, evaluation, QLoRA fine-tuning
- `configs/`: default runtime settings
- `data/`: place raw and processed datasets (not committed)

### Quick Start
1) Create a venv and install deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
2) Copy env template:
```bash
copy .env.example .env
```
3) Run a single inference:
```bash
python scripts\run_inference.py --text "The hero dies at the end to save the world."
```

### Data Preparation
```bash
python scripts\prepare_data.py --input data\raw\goodreads.csv --text-field review_text --label-field has_spoiler
```
Outputs JSONL splits under `data/splits/`.

### Evaluation
```bash
python scripts\evaluate.py --input data\splits\test.jsonl
```

### API Service
```bash
python -m spoiler_agent.api
```
POST to `http://localhost:8000/predict_spoiler` with JSON:
```json
{"text": "...", "content_type": "movie"}
```

### Fine-Tuning (QLoRA)
Install optional deps:
```bash
pip install -r requirements-finetune.txt
```
Run QLoRA training:
```bash
python scripts\fine_tune_qlora.py --input data\splits\train.jsonl --dataset-format text_label --load-in-4bit
```
If you already have `prompt` + `completion` JSONL, use `--dataset-format prompt_completion`.

### Notes
- Default model is `meta-llama/Meta-Llama-3-8B-Instruct` (set `MODEL_ID` in `.env`).
- For GPU usage, keep `DEVICE_MAP=auto` and use `DTYPE=bf16` or `DTYPE=fp16` if supported.
- Llama 3 models are gated on Hugging Face; set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) to download.
- If the model output is invalid JSON, the service defaults to `has_spoiler=0`.

### Client Example
```bash
python scripts\client_example.py
```
 -->
