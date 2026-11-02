# 颐声：养老护理需求意图识别

颐声通过ESP32-S3采集并上传音频，在本地计算机使用Qwen3-ASR转写，再识别七类护理需求。分类结果作为候选任务交由护理人员确认。语言模型采用Qwen2.5-0.5B-Instruct及护理领域LoRA适配器，传统分类基线采用字符TF-IDF与Linear SVM。

## 实验结果

同一批700条录音、两位参与者、七类各100条。下表均为百分比，F1为七类宏平均。

| 方法 | ASR Accuracy | ASR Macro F1 | 校对文本 Accuracy | 校对文本 Macro F1 |
|---|---:|---:|---:|---:|
| Prompt基础模型 | 48.86 | 45.70 | 48.71 | 45.63 |
| TF-IDF + Linear SVM | 94.00 | 94.02 | 94.14 | 94.16 |
| LoRA模型 | 98.43 | 98.50 | 99.00 | 99.07 |

训练集为371条合成文本，开发集为75条；SVM按开发集Macro F1选择C=1。700条录音的原ASR文本与参与者校对文本配对评估，12条转写经校正。

## 代码与复现

程序位于 `yisheng-speech-intent/` 子目录。

- [运行命令](yisheng-speech-intent/README.md)：依赖、指标复算和SVM训练/评估。
- [实验复现说明](yisheng-speech-intent/docs/实验复现说明.md)：数据、语言模型推理、LoRA训练及评分口径。
- [SVM配置](yisheng-speech-intent/configs/svm_baseline.json)：训练参数、开发集选择、版本和模型哈希。
- [SVM结果](yisheng-speech-intent/results/svm/)：逐条预测、指标、延迟明细和资源记录。

七类标签为 `REQUEST_WATER`、`REQUEST_FOOD`、`REQUEST_TOILET`、`CALL_CAREGIVER`、`MEDICINE_REMINDER`、`SOS`、`UNKNOWN`。

模型权重需另行准备；SVM训练和评估可在CPU运行。原始录音见仓库的录音目录，文本和归档预测见 `data/evaluation.sqlite3`。第三方组件及使用条款见 [LICENSING.md](yisheng-speech-intent/LICENSING.md)。
