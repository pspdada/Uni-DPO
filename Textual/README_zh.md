# Uni-DPO 文本理解

[English](/Textual/README.md) | **中文**

## 训练

训练流程基于 [SimPO](https://github.com/princeton-nlp/SimPO) 框架构建。请按照以下步骤配置环境：

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

前往 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) 下载 Uni-DPO 文本理解训练数据集（`Textual` 文件夹）。

### 开始训练

修改 [`configs`](/Textual/train/configs) 文件夹中的配置文件，然后修改训练脚本 [`run.sh`](/Textual/train/run.sh) 并运行：

```bash
bash train/run.sh
```
