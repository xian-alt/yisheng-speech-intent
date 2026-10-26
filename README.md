# 颐声：养老陪护语音意图识别

颐声尝试把口头表达的护理需求转换为明确的任务，例如“帮我倒杯水”“扶我去厕所”或“快来帮帮我”。音频先由 Qwen3-ASR 转成文字，再交给 Qwen2.5-0.5B-Instruct 判断意图，识别结果作为候选护理任务，交由护理人员确认。

这里整理了论文实验用到的核心代码、提示词、训练配置和匿名测试数据，方便查看实现，也方便重新计算表格中的指标。完整网页和 ESP32 固件暂未放入。

## 这次实验识别哪些需求

| 标签 | 对应需求 |
|---|---|
| `REQUEST_WATER` | 饮水、倒水或饮水协助 |
| `REQUEST_FOOD` | 用餐、送饭或进食协助 |
| `REQUEST_TOILET` | 如厕及相关协助 |
| `CALL_CAREGIVER` | 呼叫护理人员或一般护理协助 |
| `MEDICINE_REMINDER` | 用药提醒及原定服药安排相关请求 |
| `SOS` | 当前紧急求助 |
| `UNKNOWN` | 没有当前有效需求，或超出上述范围的表达 |

判断依据是整句话的意思。“我要吃药”“我已经吃过药了”和“现在不用吃药”不能只因为都出现了“吃药”就得到同一个结果。

原型支持13种标签，论文选取其中七类进行评测。ESP32-S3是系统的终端设计部分；这次700条实验录音是在电脑浏览器中采集的，尚未用这批数据验证ESP32-S3麦克风采集和上传的完整链路。

## 文件放在哪里

- `yisheng/` 和 `recognize.py`：模型调用、意图解析，以及文本和音频识别入口。
- `scripts/intent_v2_research.py`：数据检查、LoRA训练和模型比较。
- `data/evaluation.sqlite3`：匿名测试数据及各组预测结果；表结构见 `data/schema.sql`。
- `data/training/`：V2.1的训练语料及训练、推理共用的中文提示词。
- `prompts/initial_7class.txt`：初始七分类实验的提示词。
- `configs/`：训练参数、实验设置、软件版本及适配器和提示词的哈希。
- `reproduce_metrics.py` 和 `tests/`：指标复算与检查。
- `metadata/source_manifest.json`：整理文件时保留的来源与校验记录。

适配器版本和提示词版本分别记录，文件名中的“v2”不一定表示第二版适配器。

## 数据说明

测试录音来自两位志愿者，按预设护理场景录制，每人每类50条，共700条。数据库保留了原始ASR文本、校对文本、参考标签和9800条预测。9800条是同一批录音在不同方法和输入条件下的预测数量，不是9800条独立录音。

参考意图标签采用AI辅助标注。项目负责人报告已复核全部录音，校正了12条ASR文本；没有进行独立盲法人工标注。匿名转写文本已取得参与者公开许可，原始录音、姓名、联系方式及业务数据库中的其他信息未公开。

V2.1训练语料另放在 `data/training/dataset.json`，包括371条合成训练样本和75条合成开发样本。371条由275条新编写AI合成样本与96条去重后的历史合成样本组成；75条开发样本也是新编写的合成文本，尚未完成人工审核。它们与旧文件 `lora-v2.json` 中的75条历史提示词消融记录不是同一份数据。

## 复算论文中的结果

准备Python 3.10或更高版本，在这个文件夹中运行：

```sh
python reproduce_metrics.py
python -m unittest discover -s tests -v
```

这一步只读取数据库，不需要模型、显卡或额外下载。输出包括Accuracy、宏平均Precision、Recall、F1、各类结果及SOS误报和漏报情况，分数乘以100即可换算成百分比。

脚本会核对14组比较与归档指标是否一致。每组都包含700条，无效输出和失败样本也计入错误；宏平均固定覆盖七类，未定义分量按零处理。

论文表1—4对应 `initial_v1`，表5—7对应 `compact_regression`。两阶段的提示词和实验设置不同，需要分开看。部分匹配预测沿用了已保存的结果，数据库中记录了来源，不能用这些缓存结果测量时延。

## 训练和识别

重新运行模型需要自行准备Qwen2.5-0.5B-Instruct基础模型和相应的LoRA权重。这里没有附带模型权重。以下命令中的 `/path/to/local/...` 请替换成自己的实际路径。

安装依赖并训练V2.1：

```sh
python -m pip install -r requirements.txt
python scripts/intent_v2_research.py train --data data/training --output runs/my-v21 --base /path/to/local/Qwen2.5-0.5B-Instruct
```

训练使用GPU，主要参数为rank 8、alpha 16、dropout 0.05、5 epochs、batch size 2、梯度累积4、学习率0.0002。训练损失只计算助手答案部分，按合成开发集损失选择检查点；训练命令不使用测试SQLite数据库。

输入一句话进行识别：

```sh
python recognize.py --base /path/to/local/Qwen2.5-0.5B-Instruct --adapter /path/to/local/adapter --text "请给我倒杯水" --device cpu
```

导出测试输入，并重新运行模型比较：

```sh
python reproduce_metrics.py --export-inputs runs/references.json
python scripts/intent_v2_research.py evaluate --data data/training --references runs/references.json --output runs/new-comparison --base /path/to/local/base --v1 /path/to/local/v1-adapter --v2 /path/to/local/v21-adapter --device cuda:0
```

历史结果重跑需要匹配的适配器和设置。推理封装最多读取500字符、生成16个token，关闭采样；格式非法的输出即使在解析时回退为UNKNOWN，计分仍算错误。

如果需要识别新录音，还要安装ASR依赖并准备本地Qwen3-ASR-1.7B：

```sh
python -m pip install -r requirements-asr.txt
python recognize.py --base /path/to/local/base --adapter /path/to/local/adapter --audio /path/to/request.wav --asr-model /path/to/local/Qwen3-ASR-1.7B --device cuda:0
```

软件版本记录在 `configs/recorded_versions.json`。这次整理已核对归档指标的复算结果，没有在全新环境中重新完成模型训练和GPU推理验证。

## 结果应当怎样理解

这700条录音中发现的错误曾用于指导V2.1优化。因此，原始ASR输入下98.43%的准确率、校对输入下99.00%的准确率，反映的是这批数据上的开发回归表现，不能直接说明对新说话人、新表达也有相同效果。

目前的实验也不能证明真实老人群体中的效果、紧急求助零漏报、硬件端到端时延或断开公网运行情况。模型输出仍需要人工确认，不用于诊断或自动决定用药。

代码和数据的使用权限见 `LICENSING.md`。目前尚未选择统一的开源许可证，第三方模型与依赖遵守各自许可证。
