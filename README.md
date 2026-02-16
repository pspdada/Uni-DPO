<div align="center">

# Uni-DPO: A Unified Paradigm for <br> Dynamic Preference Optimization of LLMs <!-- omit in toc -->

<a href='https://arxiv.org/abs/2506.10054'>
<img src='https://img.shields.io/badge/Paper-Arxiv-purple'></a>
<a href='https://huggingface.co/datasets/psp-dada/Uni-DPO'>
<img src='https://img.shields.io/badge/Datasets-HF-Green'></a>
<a href='https://huggingface.co/collections/psp-dada/uni-dpo'>
<img src='https://img.shields.io/badge/Models-HF-orange'></a>
<a href='https://huggingface.co/papers/2506.10054'>
<img src='https://img.shields.io/badge/Discussion-HF-blue'></a>
<a href='https://github.com/pspdada/Uni-DPO/blob/main/LICENSE'>
<img src='https://img.shields.io/badge/LICENSE-Apache_2.0-yellow'></a>

<a href='https://modelscope.cn/datasets/pspdada/Uni-DPO'>
<img src='https://img.shields.io/badge/Datasets-🤖ModelScope-pink'></a>
<a href='https://modelscope.cn/collections/pspdada/Uni-DPO'>
<img src='https://img.shields.io/badge/Models-🤖ModelScope-red'></a>

<a href="/docs/README_zh.md">中文</a> | <b>English</b>

</div>

## 🎊 News <!-- omit in toc -->

- [2025.02.16] 📖 Code, data, and models are released!
- [2026.01.26] 🎉 Our Uni-DPO is accepted by **ICLR 2026**!

## 🚀 Overview <!-- omit in toc -->

**Uni-DPO** introduces a unified dynamic preference optimization paradigm for training large language models (LLMs) from preference data. Unlike prior DPO-based methods that treat all preference pairs equally, Uni-DPO jointly considers **intrinsic data quality** and **model learning dynamics**, enabling more effective and robust preference learning.

**Key advantages:**

- **Quality-aware**: Adaptively prioritizes high-quality preference pairs while down-weighting ambiguous ones.
- **Dynamics-aware**: Shifts training focus toward under-fitted samples to mitigate overfitting.
- **Unified & lightweight**: Seamlessly integrates dual-perspective weighting and calibrated NLL into standard DPO with minimal overhead.

## 📌Contents <!-- omit in toc -->

- [🔑 Key Features](#-key-features)
- [📚 Dataset](#-dataset)
  - [Textual Understanding](#textual-understanding)
  - [Mathematical Reasoning](#mathematical-reasoning)
  - [Multimodal Understanding](#multimodal-understanding)
- [📦 Model Weights](#-model-weights)
- [💻 Environment Setup](#-environment-setup)
  - [Textual Understanding](#textual-understanding-1)
  - [Mathematical Reasoning](#mathematical-reasoning-1)
  - [Multimodal Understanding](#multimodal-understanding-1)
- [📝 Citation](#-citation)

## 🔑 Key Features

- **Dual-perspective dynamic weighting for preference optimization**.
  Uni-DPO jointly models _what data is worth learning_ (intrinsic quality) and _what the model still struggles with_ (learning dynamics). By combining a quality-aware weight and a performance-aware weight, Uni-DPO dynamically reallocates training focus throughout optimization.

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure1.png" width="80%" />
  </p>
</table>

- **Quality-aware weighting filters ambiguous preference pairs**.
  Preference data varies widely in reliability. Uni-DPO leverages score margins between preferred and rejected responses to assign higher weights to clear, high-quality pairs while suppressing noisy or ambiguous ones.

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure2.png" width="80%" />
  </p>
</table>

- **Performance-aware weighting mitigates overfitting during training**.
  High-quality samples are not always the most informative once the model has already mastered them. Uni-DPO introduces a stabilized focal-style performance weight that down-weights well-fitted pairs and emphasizes hard-but-informative examples, effectively reducing overfitting.

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure3.png" width="80%" />
  </p>
</table>

- **Decoupling data quality from learning difficulty**.
  Empirical analysis reveals that data quality (score margin) and learning difficulty (reward margin) are weakly correlated. Uni-DPO explicitly models this mismatch, ensuring that optimization is guided by both dimensions rather than relying on either alone.

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure4.png" width="80%" />
  </p>
</table>

- **State-of-the-art performance across text, math, and multimodal benchmarks**.
  Uni-DPO consistently outperforms DPO and SimPO across diverse settings.

<table align="center">
  <p align="center">
    <img src="/docs/figures/table1.png" width="80%" />
  </p>
</table>

## 📚 Dataset

We present the [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) **Uni-DPO Dataset**, which contains preference pairs for training Uni-DPO across three key domains: textual instruction following, mathematical reasoning, and multimodal understanding.

### Textual Understanding

The [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Textual) `Textual` folder contains training data used for Uni-DPO textual understanding experiments, including data used in both v0.1 and v0.2 settings. The exact mapping can be found in the training config folder.

To generate these data yourself, refer to [this document](/Textual/on_policy_data_gen/README.md).

<details>
<summary>Process of generating data</summary>

1. Download [🤗](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) `HuggingFaceH4/ultrafeedback_binarized` dataset
2. Run `decode.py` to generate policy outputs and clean them using `post_process.py`
3. Run `reward_model_annotate.py` to obtain reward scores

</details>

### Mathematical Reasoning

Training data for math reasoning is located in the [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Math) `Math` folder.

If you need to generate these training data yourself, you can refer to [this document](/Math/data_generation/README.md) and use [this script](Math/data_generation/generate_data.sh).

<details>
<summary>Process of generating data</summary>

1. Download math question dataset [🤗](https://huggingface.co/datasets/RLHFlow/numia_prompt_dpo1) `RLHFlow/numia_prompt_dpo1`
2. Run `gen_samples.py` to generate model responses
3. Score with `verifiable_reward_labeling.py` and `progress_reward_labeling.py`
4. Build preference pairs using `get_uni_dpo_data.py`

</details>

Evaluation data are in [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math_eval_data.zip) `Math_eval_data.zip`. See [this document](Math/evaluation/README.md) for evaluation details.

### Multimodal Understanding

Training data are in the [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Multimodal) `Multimodal` folder.

## 📦 Model Weights

| Base Model                                                                           | Training Data                                                                                                                        | Training Setup |                                                                                           Uni-DPO Model                                                                                            |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | :------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-GPT-4o/train.jsonl)                            |      v0.1      |                [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO) Llama-3-8B-Base-SFT-Uni-DPO                |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-Qwen2_5_72B/train.jsonl)                       |      v0.2      |    [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen) Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen    |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Textual)                                                             |      v0.2      |  [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4) Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4   |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-GPT-4o/train.jsonl)        |      v0.1      |                [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO) Llama-3-8B-Instruct-Uni-DPO                |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-ArmoRM/train.jsonl)        |      v0.2      | [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM) Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-ArmoRM-GPT-4o/train.jsonl) |      v0.2      | [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o) Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o |
| [🤗](https://huggingface.co/google/gemma-2-9b-it) Gemma2-9B-IT                       | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Gemma-2-9B-IT-UltraFeedback-GPT-4o/train.jsonl)              |      v0.1      |                          [🤗](https://huggingface.co/psp-dada/Gemma2-9B-IT-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Gemma2-9B-IT-Uni-DPO) Gemma2-9B-IT-Uni-DPO                           |
| [🤗](https://huggingface.co/Qwen/Qwen2.5-7B) Qwen2.5-7B                              | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-GPT-4o/train.jsonl)                            |      v0.1      |                           [🤗](https://huggingface.co/psp-dada/Qwen2.5-7B-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Qwen2.5-Math-7B-Uni-DPO) Qwen2.5-7B-Uni-DPO                           |
| [🤗](https://huggingface.co/Qwen/Qwen2.5-Math-7B) Qwen2.5-7B                         | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math/Train_Qwen_2_5_math_7B.jsonl)                                   |      v0.1      |                      [🤗](https://huggingface.co/psp-dada/Qwen2.5-Math-7B-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Qwen2.5-Math-7B-Uni-DPO) Qwen2.5-Math-7B-Uni-DPO                      |

## 💻 Environment Setup

To ensure fair comparison with prior work, we align training and testing environments with the original implementations whenever possible. Below is a brief introduction to the environments used for each task.

### Textual Understanding

Training environment: See [this document](/Textual/README.md) for details.

- Built based on the [SimPO](https://github.com/princeton-nlp/SimPO) repository.
- Mainly rely on alignment-handbook and uses the `Trainer` class from the `transformers` library to construct a `UniDPOTrainer` class for implementing Uni-DPO training.

For evaluation, the metrics reported in the main paper strictly align with the following four evaluation environments: [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto), [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval), [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [SedarEval](https://github.com/wwn1233/sedareval). For downstream task evaluation in the appendix, we use the configuration from [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Mathematical Reasoning

Our training and evaluation environments are built based on the [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) repository. See [this document](/Math/README.md) for details.

- Training data construction: relies on vLLM for model deployment and inference
- Training: also depends on [alignment-handbook](https://github.com/huggingface/alignment-handbook) and uses the `Trainer` class from `transformers` to build the UniDPOTrainer class for Uni-DPO training
- Evaluation: the evaluation codebase is based on [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason).

### Multimodal Understanding

Following [MM-RLHF](https://github.com/Kwai-YuanQi/MM-RLHF). See [this document](/Multimodal/README.md) for details.

- Training: our training environment is built based on [LlamaFactory](https://github.com/hiyouga/LLaMAFactory), and we provide a minimal modified version and necessary training scripts
- Evaluation: our evaluation environment is built based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and we provide the required evaluation scripts and necessary documentation for running the evaluation.

## 📝 Citation

If you find our model/code/data/paper helpful, please consider citing our papers 📝 and starring us ⭐️！

```bibtex
@article{peng2025omni,
  title={Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs},
  author={Peng, Shangpin and Wang, Weinong and Tian, Zhuotao and Yang, Senqiao and Wu, Xing and Xu, Haotian and Zhang, Chengquan and Isobe, Takashi and Hu, Baotian and Zhang, Min},
  journal={arXiv preprint arXiv:2506.10054},
  year={2025}
}
```

## 📧 Contact us <!-- omit in toc -->

If you have any questions, comments, or suggestions, please do not hesitate to submit an issue or PR to help advance research in this area.

## 🙏 Acknowledgement <!-- omit in toc -->

We thank the following projects for their open-source code and datasets, which greatly facilitated our research:

- Training data generation: [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), [RLHFlow/numia_prompt_dpo1](https://huggingface.co/datasets/RLHFlow/numia_prompt_dpo1), [MM-RLHF](https://github.com/Kwai-YuanQi/MM-RLHF)
- Training: [SimPO](https://github.com/princeton-nlp/SimPO), [alignment-handbook](https://github.com/huggingface/alignment-handbook), [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1), [LlamaFactory](https://github.com/hiyouga/LLaMAFactory)
- Evaluation
  - Textual understanding: [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto), [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval), [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [SedarEval](https://github.com/wwn1233/sedareval), [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - Math reasoning: [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
  - Multimodal understanding: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

## License <!-- omit in toc -->

[Apache License 2.0](/LICENSE)
