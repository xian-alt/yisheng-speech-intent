# 颐声语音意图识别

面向养老陪护场景的语音需求识别。使用Qwen3-ASR将录音转成文字，再由Qwen2.5-0.5B-Instruct判断意图，支持基础模型与LoRA微调模型的比较。

本目录包含论文使用的核心代码、提示词、训练配置和匿名评测数据。ESP32-S3为终端设计部分，本轮700条评测录音由电脑浏览器采集；完整网页与ESP32固件未包含在内。

## 文件

- `recognize.py`：单条文本或音频的识别入口。
- `yisheng/models.py`：语言模型加载与推理。
- `yisheng/intents.py`：意图标签及输出解析。
- `yisheng/qwen_asr.py`：Qwen3-ASR语音转写。
- `scripts/intent_v2_research.py`：数据检查、LoRA训练与模型比较。
- `reproduce_metrics.py`：读取归档预测，重新计算论文指标。
- `data/evaluation.sqlite3`：匿名转写文本、参考标签和各组预测结果。
- `data/training/`：V2.1训练数据与完整中文提示词。
- `prompts/initial_7class.txt`：初始实验的七分类提示词。
- `configs/`：实验设置、训练参数和软件版本记录。

数据库结构见`data/schema.sql`，文件来源与哈希见`metadata/source_manifest.json`。

## 运行环境

指标复算使用Python 3.10或更高版本，无需安装模型依赖。

模型推理和训练需要PyTorch、Transformers、PEFT等依赖，以及本地Qwen2.5-0.5B-Instruct模型。处理新音频还需要Qwen3-ASR-1.7B。模型权重需自行准备。

```sh
python -m pip install -r requirements.txt
```

音频识别另需安装：

```sh
python -m pip install -r requirements-asr.txt
```

记录的实验环境为Intel Core i7-13700HX、NVIDIA GeForce RTX 5060 Laptop GPU（约8GB显存）。初始语言模型比较使用CPU float32，V2.1回归比较使用GPU float16。具体软件版本见`configs/recorded_versions.json`。

## 配置与运行

以下命令均在`recognize.py`所在目录执行。将`/path/to/local/...`替换为自己的模型或适配器路径。

文本识别：

```sh
python recognize.py --base /path/to/local/Qwen2.5-0.5B-Instruct --adapter /path/to/local/adapter --text "请给我倒杯水" --device cpu
```

省略`--adapter`可使用基础模型；GPU推理使用`--device cuda:0`。默认提示词位于`data/training/prompt.txt`，也可以通过`--prompt`指定其他文件。

音频识别：

```sh
python recognize.py --base /path/to/local/Qwen2.5-0.5B-Instruct --adapter /path/to/local/adapter --audio /path/to/request.wav --asr-model /path/to/local/Qwen3-ASR-1.7B --device cuda:0
```

LoRA训练：

```sh
python scripts/intent_v2_research.py train --data data/training --output runs/my-v21 --base /path/to/local/Qwen2.5-0.5B-Instruct
```

V2.1采用rank 8、alpha 16、dropout 0.05、5 epochs、batch size 2、梯度累积4、学习率0.0002。训练和推理使用同一份中文提示词。仅计算助手答案部分的训练损失，按合成开发集损失选择检查点。训练配置记录见`configs/train_v21.json`；训练命令不使用评测SQLite数据库。

训练语料为371条合成训练样本（275条新编写AI样本与96条去重历史合成样本），另有75条新编写合成开发样本，尚未完成人工审核。这75条与旧文件`lora-v2.json`中的历史提示词消融记录不同；提示词版本和适配器版本分别记录。

## 识别方式与输出

论文评测七种意图：

| 标签 | 含义 |
|---|---|
| `REQUEST_WATER` | 饮水 |
| `REQUEST_FOOD` | 用餐 |
| `REQUEST_TOILET` | 如厕 |
| `CALL_CAREGIVER` | 呼叫护理人员或一般护理协助 |
| `MEDICINE_REMINDER` | 用药提醒及原定服药安排相关请求 |
| `SOS` | 当前紧急求助 |
| `UNKNOWN` | 无当前有效需求，或超出本次评测范围 |

模型判断的是当前请求，而非句子里是否出现某个词。例如“我已经吃过药了”不等于“现在需要吃药”。原型支持13种标签，本次论文评测其中七类。

`recognize.py`输出JSON，包括转写文本、模型原始输出、解析结果及输出是否合法。推理最多读取500字符、生成16个token，关闭采样。格式非法的输出即使解析时回退为UNKNOWN，计分仍算错误。识别结果作为候选护理任务交由人工确认，不用于自动诊断或决定用药。

## 指标与测试数据

```sh
python reproduce_metrics.py
python -m unittest discover -s tests -v
```

脚本只读数据库，复算Accuracy、宏平均Precision、Recall、F1、各类结果及SOS误报和漏报，并与归档指标核对。输出分数乘以100可换算为百分比；未定义分量按零处理，无效输出保留为错误。

数据库包含两位志愿者的700条录音对应文本，每人每类50条，共七类；9800条预测对应这批录音的14组比较，不是9800条独立录音。初始协议`initial_v1`对应论文表1—4，`compact_regression`对应表5—7，两阶段需要分开统计。复用预测已记录来源，不能用于测量时延。

参考标签采用AI辅助标注。项目负责人报告已复核录音、校正12条ASR文本，未进行独立盲法人工标注。匿名转写文本已取得参与者公开许可；原始录音、姓名、联系方式与业务数据库其他信息未公开。

700条录音中发现的错误曾用于指导V2.1优化。因此，98.43%的原始ASR输入准确率和99.00%的校对输入准确率属于开发阶段回归结果，不代表对新说话人、新表达的独立测试表现，也不能证明真实养老场景中的零漏报。

已核对归档指标复算，未在全新环境中重新完成模型训练与GPU推理验证。代码和数据的使用权限见`LICENSING.md`；目前尚未选定统一开源许可证，第三方模型和依赖遵守各自许可证。
