# Uni-DPO 多模态理解 <!-- omit in toc -->

[English](/Multimodal/README.md) | **中文**

本文档提供 Uni-DPO 多模态理解任务的训练与测试详细指南。

## 目录 <!-- omit in toc -->

- [训练](#训练)
  - [安装必要依赖](#安装必要依赖)
  - [添加 Uni-DPO 适配](#添加-uni-dpo-适配)
  - [准备训练数据](#准备训练数据)
  - [启动训练](#启动训练)
- [测试](#测试)

## 训练

训练流程基于 [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) 框架，环境搭建可参考以下步骤：

### 安装必要依赖

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
git checkout 92fa3df

conda create -n Uni-DPO-Multimodal-train python=3.11.14 -y
conda activate Uni-DPO-Multimodal-train

pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4 qwen_vl_utils
```

### 添加 Uni-DPO 适配

将本项目中 `Multimodal/LlamaFactory` 文件夹内的文件复制至 `LlamaFactory` 目录下，并**覆盖**原文件。新增文件包含对训练代码与配置的最小侵入式修改，以支持 Uni-DPO 多模态理解训练。

<details>
<summary>文件详细说明</summary>

对于 `python` 文件，我们使用以下注释标识为适配 Uni-DPO 所新增或修改的代码段：

```bash
#! Below this line are additions for Uni-DPO.
这里是 Uni-DPO 的修改内容
#! Above this line are additions for Uni-DPO.
```

</details>

### 准备训练数据

1. 准备图片数据

我们使用 [MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) 数据集中的图片部分。请先下载数据并解压至本地。下载命令如下：

```bash
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/long.zip
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/short.zip
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/mcq.zip
```

2. 准备偏好样本对

前往 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) 下载 **Uni-DPO 多模态训练数据集**，并将其放置于 `LlamaFactory/data` 目录下，目录结构示例如下：

```bash
- LlamaFactory
  - data
    - dataset_info.json # 数据集信息文件
    - uni_dpo_image_only_mcq_short_long_50k.json # 训练数据文件
```

训练数据格式示例如下：

```json
{
  "instruction": "[str] <image>这里是问题",
  "chosen": "[str] 较好的回答文本",
  "rejected": "[str] 较差的回答文本",
  "score_chosen": "[float] 较好的回答得分",
  "score_rejected": "[float] 较差的回答得分",
  "images": ["/your/path/to/MM-RLHF/long/..."]
}
```

请使用 [`change_data_image_path.py`](/Multimodal/LlamaFactory/scripts/change_data_image_path.py) 将 `images` 字段中的路径替换为本地绝对路径，使其正确指向已下载并解压的图片目录。

### 启动训练

修改并运行训练脚本：

```bash
bash examples/uni_dpo/Qwen2_VL_2B_uni_dpo.sh
```

## 测试

测试流程基于 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。请先按照原项目说明完成环境配置，然后安装以下依赖：

```bash
pip install qwen_vl_utils  vllm==0.8.2
```

测试前请将 [model.py](/Multimodal/VLMEvalKit/vlm/qwen2_vl/model.py) 覆盖原文件，以适配预训练模型的推理流程。

测试所使用的配置文件为 [config.json](/Multimodal/VLMEvalKit/config.json)
