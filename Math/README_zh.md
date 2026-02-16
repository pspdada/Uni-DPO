# Uni-DPO 数学推理

[English](/Math/README.md) | **中文**

## 训练

训练流程基于 [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) 框架构建。请按照以下步骤配置环境：

### 安装依赖

> 注意：Uni-DPO 的文本理解与数学推理任务使用相同的训练依赖（conda 环境 `Uni-DPO-alignment`），因此只需构建一次环境即可

```bash
conda create -n Uni-DPO-alignment python=3.10.19 -y
conda activate Uni-DPO-alignment

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook && git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41

# 安装依赖
pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

python -m pip install .

pip install accelerate==0.33.0 huggingface-hub==0.24.7 transformers==4.42.2 peft==0.7.1 deepspeed==0.15.4 trl==0.9.6 wandb pebble==5.1.1 timeout_decorator==0.5.0 matplotlib bitsandbytes rich

pip install --no-build-isolation flash-attn==2.8.3
# 或使用下面的 wheel 安装 flash-attn
# wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 准备训练数据

前往 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) 下载 Uni-DPO 数学推理训练数据集（`Math` 文件夹），并将其放置到 `Math/train/data` 目录下。目录结构应如下所示：

```bash
- Math
  - train
    - data
      - Train_Qwen_2_5_math_7B.jsonl
```

### 开始训练

修改并运行训练脚本：

```bash
bash train/run.sh
```

## 评测

### 环境要求

你可以使用以下命令安装所需依赖：

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
# 或使用下面的 wheel 安装 flash-attn
# wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

### 准备评测数据

前往 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) 下载 Uni-DPO 数学推理评测数据集（[🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math_eval_data.zip) `Math_eval_data.zip` 文件），并将其放置到 `Math/evaluation/data` 目录下。目录结构应如下所示：

```bash
- Math
  - evaluation
    - data
      - aime24
      - ...
```

### 运行评测

使用 [batch_eval.sh](/Math/evaluation/batch_eval.sh) 脚本批量评测模型在数学推理任务上的表现。

你可以使用 [merge_results.py](/Math/evaluation/merge_results.py) 脚本将评测结果合并到一个文件中，以便更方便地进行分析。
