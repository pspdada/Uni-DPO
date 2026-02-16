# Uni-DPO Math Reasoning <!-- omit in toc -->

[中文](/Math/README_zh.md) | **English**

## Contents <!-- omit in toc -->

- [Training](#training)
  - [Install Required Dependencies](#install-required-dependencies)
  - [Prepare Training Data](#prepare-training-data)
  - [Start Training](#start-training)
- [Evaluation](#evaluation)
  - [Requirements](#requirements)
  - [Prepare Evaluation Data](#prepare-evaluation-data)
  - [Run Evaluation](#run-evaluation)
  - [Merge Evaluation Results](#merge-evaluation-results)
- [Data generation](#data-generation)

## Training

The training pipeline is built on the [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) framework. Follow the steps below to set up the environment:

### Install Required Dependencies

> Note: Uni-DPO text understanding and math reasoning share the same training dependency (conda env: `Uni-DPO-alignment`), so only one environment setup is required.

```bash
conda create -n Uni-DPO-alignment python=3.10.19 -y
conda activate Uni-DPO-alignment

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook && git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41

# install requirements
pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

python -m pip install .

pip install accelerate==0.33.0 huggingface-hub==0.24.7 transformers==4.42.2 peft==0.7.1 deepspeed==0.15.4 trl==0.9.6 wandb pebble==5.1.1 timeout_decorator==0.5.0 matplotlib bitsandbytes rich

pip install --no-build-isolation flash-attn==2.8.3
# or use the following wheel to install flash-attn
# wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Prepare Training Data

Go to [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) to download the Uni-DPO math reasoning training dataset (the `Math` folder), and place it under the `Math/train/data` directory. The directory structure should look like this:

```bash
- Math
  - train
    - data
      - Train_Qwen_2_5_math_7B.jsonl
```

### Start Training

Modify and run the training script:

```bash
bash train/run.sh
```

## Evaluation

### Requirements

You can install the required packages with the following command:

```bash
conda create -n Uni-DPO-Math-eval python=3.10.19 -y
conda activate Uni-DPO-Math-eval

cd latex2sympy
pip install -e .
cd ..

pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt # Math/evaluation/requirements.txt

pip install --no-build-isolation flash-attn==2.8.3
# or use the following wheel to install flash-attn
# wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

### Prepare Evaluation Data

Go to [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) to download the Uni-DPO math reasoning evaluation dataset (the [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math_eval_data.zip) `Math_eval_data.zip` file), and place it under the `Math/evaluation/data` directory. The directory structure should look like this:

```bash
- Math
  - evaluation
    - data
      - aime24
      - ...
```

### Run Evaluation

Use the [batch_eval.sh](/Math/evaluation/batch_eval.sh) script to evaluate the model’s performance on math reasoning tasks in batch.

### Merge Evaluation Results

You can use the [merge_results.py](/Math/evaluation/merge_results.py) script to merge the evaluation results into a single file for easier analysis.

## Data generation

If you want to generate the training data for math reasoning yourself, you can use [this script](/Math/data_generation/generate_data.sh).
