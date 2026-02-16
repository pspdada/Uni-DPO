# On-Policy Preference Data Generation

[中文](/Textual/on_policy_data_gen/README_zh.md) | **English**


We provide scripts for generating on-policy preference datasets (e.g., `Llama-3-8B-Instruct-UltraFeedback-ArmoRM`) used in our experiments.

## Requirements

Install the [`vllm`](https://github.com/vllm-project/vllm) package for decoding. If you plan to run decoding with `gemma-2` models, you will need to also install `flashinfer`.

## Generation Pipeline

### 1. Generate multiple responses

```bash
python decode.py --data_dir $DATASET_DIR --seed $SEED
```

This command produces one response per prompt for the specified random seed. You must supply a dataset containing prompts (default: `HuggingFaceH4/ultrafeedback_binarized`). Decoding hyperparameters can be customized through command-line arguments (default sampling temperature: `0.8`).

To obtain diverse responses for each prompt, run this command with **multiple different seeds** (default: `13, 21, 42, 79, 100`).

### 2. Post-process generated outputs

```bash
python post_process.py
```

This step merges responses generated under different seeds and removes samples where all outputs are identical.

### 3. Annotate preference labels

```bash
python reward_model_annotate.py --reward_model $MODEL
```

This script scores each generated response using a reward model (default: `RLHFlow/ArmoRM-Llama3-8B-v0.1`). The dataset is then binarized by selecting the highest-scoring response as the **chosen** sample and the lowest-scoring one as the **rejected** sample.
