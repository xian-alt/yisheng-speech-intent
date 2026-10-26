# 颐声：养老护理需求意图识别

颐声通过语音接收饮水、用餐、如厕、用药提醒和紧急求助等护理需求。系统使用Qwen3-ASR进行语音转写，再由经过LoRA微调的Qwen2.5-0.5B-Instruct识别当前需求，生成候选护理任务，交由护理人员确认。

本仓库提供论文实验相关的核心代码、提示词、训练配置和匿名评测数据，主要比较基础语言模型与LoRA微调模型在七类护理需求上的识别表现。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| `recognize.py` | 单条文本或音频识别入口 |
| `yisheng/models.py` | 意图模型加载与推理 |
| `yisheng/intents.py` | 意图标签及输出解析 |
| `yisheng/qwen_asr.py` | Qwen3-ASR语音转写 |
| `scripts/intent_v2_research.py` | 数据检查、LoRA训练与比较评测 |
| `reproduce_metrics.py` | 根据归档预测复算论文指标 |
| `data/evaluation.sqlite3` | 匿名转写文本、参考标签及预测记录 |
| `data/training/` | V2.1训练语料及训练、推理共用的中文提示词 |
| `prompts/initial_7class.txt` | 初始实验的七分类提示词 |
| `configs/` | 训练参数、实验协议、软件版本与模型校验信息 |
| `tests/` | 指标复算与错误计数检查 |

数据库表结构见`data/schema.sql`，文件来源与哈希记录见`metadata/source_manifest.json`。完整网页、ESP32固件及模型权重未包含在本仓库中。

## 识别类别

| 标签 | 统一英文名称 | 护理需求 |
|---|---|---|
| `REQUEST_WATER` | Drinking water | 饮水、倒水或饮水协助 |
| `REQUEST_FOOD` | Eating | 用餐、送饭或进食协助 |
| `REQUEST_TOILET` | Toileting | 如厕及相关协助 |
| `CALL_CAREGIVER` | Calling a caregiver | 呼叫护理人员或一般护理协助 |
| `MEDICINE_REMINDER` | Medication-related requests | 用药提醒及原定服药安排相关请求 |
| `SOS` | SOS | 当前紧急求助 |
| `UNKNOWN` | UNKNOWN | 没有当前有效需求，或超出本次评测范围的表达 |

模型根据完整语义判断当前请求。例如，“我要吃药”表达当前需求，“我已经吃过药了”描述过去发生的事情，“现在不用吃药”则表示否定。提示词据此说明各类含义，并要求模型只输出一个标签。

原型支持13种意图，论文选取上述七类进行评测。`recognize.py`输出转写文本、模型原始回答、解析标签和输出合法性检查结果。

## 运行环境

- Python 3.10或更高版本。
- 模型推理与训练使用PyTorch、Transformers、PEFT等依赖。
- 本地Qwen2.5-0.5B-Instruct模型，以及需要使用的LoRA适配器。
- 识别音频时，另需本地Qwen3-ASR-1.7B模型。

实验电脑采用Intel Core i7-13700HX和NVIDIA GeForce RTX 5060 Laptop GPU（约8GB显存）。初始语言模型比较使用CPU float32，后续紧凑提示词回归比较使用GPU float16。软件版本记录在`configs/recorded_versions.json`。

以下命令均在`recognize.py`所在目录运行。目前该目录位于GitHub仓库的`yisheng-speech-intent/`子文件夹中。

安装模型相关依赖：

```sh
python -m pip install -r requirements.txt
```

处理音频时另行安装：

```sh
python -m pip install -r requirements-asr.txt
```

## 使用方法

将命令中的`/path/to/local/...`替换为自己的模型和适配器路径。

### 文本识别

```sh
python recognize.py --base /path/to/local/Qwen2.5-0.5B-Instruct --adapter /path/to/local/adapter --text "请给我倒杯水" --device cpu
```

省略`--adapter`使用基础模型；GPU推理将设备改为`--device cuda:0`。默认提示词为`data/training/prompt.txt`，也可通过`--prompt`指定其他提示词文件。

### 音频识别

```sh
python recognize.py --base /path/to/local/Qwen2.5-0.5B-Instruct --adapter /path/to/local/adapter --audio /path/to/request.wav --asr-model /path/to/local/Qwen3-ASR-1.7B --device cuda:0
```

### LoRA训练

```sh
python scripts/intent_v2_research.py train --data data/training --output runs/my-v21 --base /path/to/local/Qwen2.5-0.5B-Instruct
```

V2.1使用371条合成训练样本和75条合成开发样本。训练样本由275条新编写AI合成文本和96条去重后的历史合成文本组成。训练与推理使用相同的中文提示词，仅对助手答案部分计算损失，并按合成开发集损失选择检查点。

主要训练参数如下，完整记录见`configs/train_v21.json`：

| 参数 | 设置 |
|---|---|
| LoRA rank / alpha | 8 / 16 |
| Dropout | 0.05 |
| Epochs | 5 |
| Batch size | 2 |
| 梯度累积 | 4 |
| 学习率 | 0.0002 |

训练命令不使用评测SQLite数据库进行梯度训练。适配器版本与提示词版本分别记录，初始实验和后续回归比较的配置见`configs/protocols.json`。

## 实验结果

数据集包含两位志愿者使用ESP32-S3连接的麦克风录制并通过Wi-Fi上传的700条语音，七类各100条，每人每类50条。音频由ESP32-S3上传至本地电脑，电脑端完成ASR转写与意图识别。实验分别使用原始ASR文本与参与者校对文本，比较转写条件对意图识别的影响。

在紧凑提示词协议下，原始ASR文本的回归评估结果如下。所有数值均为百分比，P、R和F1采用七类宏平均。

| 方法 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|
| 基础语言模型 | 48.86 | 53.93 | 48.86 | 45.70 |
| 原始LoRA V1 | 81.14 | 88.40 | 81.14 | 82.07 |
| LoRA V2.1 | 98.43 | 98.58 | 98.43 | 98.50 |

V2.1正确识别689条请求，准确率比同协议下的原LoRA V1提高17.29个百分点，比基础语言模型提高49.57个百分点。使用校对文本后，正确识别693条，准确率为99.00%，宏平均F1为99.07%。对100条SOS样本，两种输入条件下均未出现漏判；原始ASR输入下出现1次误报，校对后该误报消除。

### 复算指标

```sh
python reproduce_metrics.py
python -m unittest discover -s tests -v
```

复算仅使用Python标准库，无需下载模型。脚本只读数据库，计算总体与各类指标、混淆计数和SOS误报／漏报，并核对14组归档比较。输出分数乘以100可换算为百分比。无效输出与失败样本保留为错误，宏平均固定覆盖七类，未定义分量按零处理。

数据库中的9800条预测来自同一批700条语音的14组归档比较，保留完整历史实验记录；论文选取其中10组基础语言模型与LoRA模型的比较。`initial_v1`对应论文表1—4，`compact_regression`对应表5—7；两阶段配置不同，分别报告，不能把阶段间的全部差异归因于LoRA重新训练。复用的预测已记录来源。

## 数据与使用说明

匿名转写文本已取得参与者公开许可。公开数据保留原始ASR文本、校对文本、参考标签和模型输出，不包含原始录音、姓名、联系方式或业务数据库中的其他信息。参考标签采用AI辅助标注。




