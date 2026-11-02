# 运行与复现

当前目录为程序目录，以下命令均在此运行。

## 复算已保存成绩

基础模型和LoRA的历史预测保存在SQLite；新增SVM预测保存在 `results/svm/predictions.jsonl`。只复算分数需要Python标准库：

```powershell
python reproduce_metrics.py > metrics-llm.json
python scripts/reproduce_svm_metrics.py > metrics-svm.json
```

语言模型取 `compact_regression/base/{asr,proofread}` 和 `compact_regression/v21/{asr,proofread}`，其中 `v21` 对应论文LoRA模型。SVM结果取 `asr` 与 `proofread`。两种转写条件来自同700条录音，分别计分。

| 输入 | 方法 | 正确数 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| ASR | Prompt基础模型 | 342/700 | 48.86 | 53.93 | 48.86 | 45.70 |
| ASR | TF-IDF + Linear SVM | 658/700 | 94.00 | 94.10 | 94.00 | 94.02 |
| ASR | LoRA | 689/700 | 98.43 | 98.58 | 98.43 | 98.50 |
| 校对 | Prompt基础模型 | 341/700 | 48.71 | 54.70 | 48.71 | 45.63 |
| 校对 | TF-IDF + Linear SVM | 659/700 | 94.14 | 94.21 | 94.14 | 94.16 |
| 校对 | LoRA | 693/700 | 99.00 | 99.15 | 99.00 | 99.07 |

数值为百分比。训练/开发集为371/75条合成文本；录音曾用于开发诊断，结果按回归评估解释。

## 运行SVM

记录环境为Python 3.12。SVM依赖与GPU语言模型依赖分开安装：

```powershell
python -m pip install -r requirements-svm.txt
python -m unittest discover -s tests -v
python scripts/svm_baseline.py train --source . --data data/training/dataset.json --output runs/svm-train
python scripts/svm_baseline.py evaluate --source . --database data/evaluation.sqlite3 --frozen runs/svm-train --output runs/svm-evaluation
python scripts/svm_baseline.py benchmark --source . --data data/training/dataset.json --frozen runs/svm-train --output runs/svm-resources
```

`train`只读取训练/开发文本，TF-IDF仅在训练集拟合，按开发集Macro F1选择C并保存完整模型及 `frozen.json`。`evaluate`加载该模型，对两个转写条件各700条录音评分；`benchmark`使用开发文本测量CPU单条推理。各输出目录必须是新目录，重复运行请换名称。测速请与其他计算任务分开运行。

## 语言模型与音频

```powershell
python -m pip install -r requirements.txt
python recognize.py --base "D:/models/Qwen2.5-0.5B-Instruct" --adapter "D:/models/care-intent-adapter" --text "请给我倒杯水" --device cuda:0
```

替换为实际模型路径；去掉 `--adapter` 使用基础模型。音频转写另需 `requirements-asr.txt` 及本地Qwen3-ASR模型。批量推理、LoRA训练和文件对应关系见 [实验复现说明](docs/实验复现说明.md)。
