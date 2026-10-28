# 颐声：养老护理需求意图识别

颐声将语音转写成文本，再识别饮水、用餐、如厕、呼叫护理人员、用药相关请求和紧急求助等当前需求。转写使用Qwen3-ASR，意图识别使用Qwen2.5-0.5B-Instruct及护理领域LoRA适配器，识别结果作为候选护理任务交由护理人员确认。

仓库保存论文实验使用的核心代码、中文提示词、训练配置和匿名评测数据库。目前论文比较基础模型与LoRA模型，按ASR和参与者校对两种文本条件报告结果，共四张表。

**完整流程、文件位置和运行命令见：[实验复现说明](yisheng-speech-intent/docs/实验复现说明.md)。**

## 代码与数据

程序位于本仓库的`yisheng-speech-intent/`子目录。运行命令前先进入该目录。

| 文件 | 用途 |
|---|---|
| `recognize.py` | 单条文本或音频识别 |
| `yisheng/qwen_asr.py` | Qwen3-ASR转写 |
| `yisheng/models.py` | 基础模型和LoRA模型的加载、推理及输出解析 |
| `yisheng/intents.py` | 标签定义和解析规则 |
| `scripts/intent_v2_research.py` | 已有的数据处理、LoRA训练和批量文本评测 |
| `reproduce_metrics.py` | 从已保存预测复算论文指标 |
| `data/training/prompt.txt` | 训练、推理共用的七类中文提示词 |
| `data/training/dataset.json` | 371条训练样本和75条开发样本 |
| `data/evaluation.sqlite3` | 700条匿名转写、参考标签和归档预测 |
| `configs/` | 训练配置、实验分组、版本和模型哈希 |
| `tests/test_reproduction.py` | 数据及计分规则检查 |

已有代码保持原样，没有增加测试入口。完整网页、ESP32固件及模型权重不包含在本代码目录中。

## 七类标签

| 标签 | 英文名称 | 含义 |
|---|---|---|
| `REQUEST_WATER` | Drinking water | 饮水、倒水或饮水协助 |
| `REQUEST_FOOD` | Eating | 用餐、送饭或进食协助 |
| `REQUEST_TOILET` | Toileting | 如厕及相关协助 |
| `CALL_CAREGIVER` | Calling a caregiver | 一般护理协助 |
| `MEDICINE_REMINDER` | Medication-related requests | 用药提醒及原定服药安排相关请求 |
| `SOS` | SOS | 当前紧急求助 |
| `UNKNOWN` | UNKNOWN | 无有效当前请求，或超出定义范围 |

## 先核对论文成绩

以下命令在代码子目录运行，只需Python标准库：

```powershell
python reproduce_metrics.py > metrics-all.json
python -m unittest discover -s tests -v
```




## 重新运行模型

模型训练、推理依赖见`requirements.txt`，音频识别另需`requirements-asr.txt`。实验计算机为Intel Core i7-13700HX、NVIDIA GeForce RTX 5060 Laptop GPU，约8GB显存；当前论文语言模型比较使用GPU float16。

单条识别示例，替换本地模型路径后运行：

```powershell
python recognize.py --base "D:/models/Qwen2.5-0.5B-Instruct" --adapter "D:/models/care-intent-adapter" --text "请给我倒杯水" --device cuda:0
```

去掉`--adapter`即使用基础模型。批量评测、原脚本兼容参数和可选训练步骤见复现说明。模型权重尚未包含在仓库；只有代码时可以复算指标，重新运行论文适配器还需要相应权重。

## 数据说明

评测包含两位参与者的700条录音对应文本，七类各100条，每人每类50条。原始ASR文本与参与者校对文本使用相同录音编号和参考标签，12条转写经校正。训练和开发语料与录音评测集分开保存。


源文件校验信息见`metadata/source_manifest.json`，数据库结构见`data/schema.sql`，第三方组件及使用条款见`LICENSING.md`。数据库中的历史预测和原始元数据保留用于核对。
