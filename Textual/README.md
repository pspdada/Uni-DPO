# Uni-DPO Textual Understanding

[中文](/Textual/README_zh.md) | **English**

## Training

The training pipeline is built on the [SimPO](https://github.com/princeton-nlp/SimPO) repository. Follow the steps below to set up the environment:

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

Go to [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) to download the Uni-DPO text understanding training dataset (the `Textual` folder).

### Start Training

Modify the training config in the [`configs`](/Textual/train/configs) folder. Then modify the training script [`run.sh`](/Textual/train/run.sh) and run:

```bash
bash train/run.sh
```
