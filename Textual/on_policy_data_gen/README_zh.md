# On-Policy 偏好数据生成

[English](/Textual/on_policy_data_gen/README.md) | **中文**

我们提供了一套脚本，用于生成实验中使用的 on-policy 偏好数据集（例如 `Llama-3-8B-Instruct-UltraFeedback-ArmoRM`）。

## 环境要求

请先安装 `vllm` 以进行解码。如果你计划使用 `gemma-2` 系列模型进行解码，还需要额外安装 `flashinfer`。

## 数据生成流程

### 1. 生成多个回复

```bash
python decode.py --data_dir $DATASET_DIR --seed $SEED
```

该命令会针对指定随机种子，为数据集中的每个 prompt 生成一个回复。你需要提供一个包含 prompts 的数据集（默认：`HuggingFaceH4/ultrafeedback_binarized`）。

解码超参数可通过命令行参数自定义（默认采样温度为 `0.8`）。

为了获得每个 prompt 的多样化回复，请使用**多个不同随机种子**运行该命令（默认：`13, 21, 42, 79, 100`）。

### 2. 后处理生成结果

```bash
python post_process.py
```

该步骤会合并不同随机种子生成的回复，并删除所有输出完全相同的样本。

### 3. 标注偏好标签

```bash
python reward_model_annotate.py --reward_model $MODEL
```

该脚本会使用奖励模型为每条生成回复打分（默认：`RLHFlow/ArmoRM-Llama3-8B-v0.1`）。随后数据将被二值化处理：

- 得分最高的回复 → **chosen**
- 得分最低的回复 → **rejected**
