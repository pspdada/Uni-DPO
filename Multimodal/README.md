# Uni-DPO Multimodal Understanding

[中文](/Multimodal/README_zh.md) | **English**

This document provides a detailed guide for training and testing the Uni-DPO multimodal understanding task.

## Training

The training pipeline is based on the [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) framework. You can follow the steps below to set up the environment:

### Install Required Dependencies

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
git checkout 92fa3df

conda create -n Uni-DPO-Multimodal-train python=3.11.14 -y
conda activate Uni-DPO-Multimodal-train

pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4 qwen_vl_utils
```

### Add Uni-DPO Adaptation

Copy the files from the `Multimodal/LlamaFactory` folder in this project into the `LlamaFactory` directory and **overwrite** the original files. The added files contain minimally invasive modifications to training code and configurations to support Uni-DPO multimodal understanding training.

<details>
<summary>File Details</summary>

For `python` files, we mark code sections that were added or modified for Uni-DPO using the following comments:

```bash
#! Below this line are additions for Uni-DPO.
Here are the modifications for Uni-DPO
#! Above this line are additions for Uni-DPO.
```

</details>

### Prepare Training Data

1. Prepare image data

We use the image portion of the [MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) dataset. Please download and extract the data locally first. Download commands:

```bash
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/long.zip
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/short.zip
wget -c https://huggingface.co/datasets/yifanzhang114/MM-RLHF/resolve/main/mcq.zip
```

2. Prepare preference pairs

Go to [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) to download the **Uni-DPO multimodal training dataset**, and place it under the `LlamaFactory/data` directory. The directory structure should look like this:

```bash
- LlamaFactory
  - data
    - dataset_info.json # Dataset info file
    - uni_dpo_image_only_mcq_short_long_50k.json # Training data file
```

Example training data format:

```json
{
  "instruction": "[str] <image>Here is the question",
  "chosen": "[str] Better response text",
  "rejected": "[str] Worse response text",
  "score_chosen": "[float] Score of the better response",
  "score_rejected": "[float] Score of the worse response",
  "images": ["/your/path/to/MM-RLHF/long/..."]
}
```

Please use [`change_data_image_path.py`](/Multimodal/LlamaFactory/scripts/change_data_image_path.py) to replace the paths in the `images` field with local absolute paths so they correctly point to the downloaded and extracted image directory.

### Start Training

Modify and run the training script:

```bash
bash examples/uni_dpo/Qwen2_VL_2B_uni_dpo.sh
```

## Evaluation

The testing pipeline is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Please first follow the original project instructions to complete environment setup, then install the following dependencies:

```bash
pip install qwen_vl_utils  vllm==0.8.2
```

Before testing, replace the original file with [model.py](/Multimodal/VLMEvalKit/vlm/qwen2_vl/model.py) to adapt the inference pipeline for the pretrained model.

The configuration file used for testing is [config.json](/Multimodal/VLMEvalKit/config.json)
