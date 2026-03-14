<div align="center">

# Uni-DPO: A Unified Paradigm for <br> Dynamic Preference Optimization of LLMs <!-- omit in toc -->

<a href='https://arxiv.org/abs/2506.10054'>
<img src='https://img.shields.io/badge/论文-Arxiv-purple'></a>
<a href='https://huggingface.co/datasets/psp-dada/Uni-DPO'>
<img src='https://img.shields.io/badge/数据集-HF-Green'></a>
<a href='https://huggingface.co/collections/psp-dada/uni-dpo'>
<img src='https://img.shields.io/badge/模型-HF-orange'></a>
<a href='https://huggingface.co/papers/2506.10054'>
<img src='https://img.shields.io/badge/讨论区-HF-blue'></a>
<a href='https://github.com/pspdada/Uni-DPO/blob/main/LICENSE'>
<img src='https://img.shields.io/badge/许可证-Apache_2.0-yellow'></a>

<a href='https://modelscope.cn/datasets/pspdada/Uni-DPO'>
<img src='https://img.shields.io/badge/数据集-🤖ModelScope-pink'></a>
<a href='https://modelscope.cn/collections/pspdada/Uni-DPO'>
<img src='https://img.shields.io/badge/模型-🤖ModelScope-red'></a>

<a href="/README.md">English</a> | <b>中文</b>

</div>

## 🎊 新闻 <!-- omit in toc -->

- [2026.03.15] 🖼️ 我们已上传 [Poster](/docs/poster.pdf)，欢迎查看！
- [2026.02.16] 📖 代码、数据与模型已发布！
- [2026.01.26] 🎉 我们的 Uni-DPO 被 **ICLR 2026** 接收！

## 🚀 概览 <!-- omit in toc -->

**Uni-DPO** 提出一种统一的动态偏好优化范式，用于基于偏好数据训练大语言模型。不同于以往将所有偏好样本等同处理的 DPO 方法，Uni-DPO 同时考虑：**偏好数据自身质量**与**模型学习动态**，从而实现更有效、更稳健的偏好学习。

**核心优势：**

- **数据质量感知**：自适应地提升高质量样本权重，降低模糊样本影响
- **训练动态感知**：动态地关注模型尚未学会的样本，缓解过拟合
- **统一且轻量**：无缝将双视角动态加权机制和校准 NLL 损失集成到标准 DPO 训练流程，额外开销极小

## 📌 目录 <!-- omit in toc -->

- [🔑 主要特性](#-主要特性)
- [📚 数据集](#-数据集)
  - [文本理解](#文本理解)
  - [数学推理](#数学推理)
  - [多模态理解](#多模态理解)
- [📦 模型权重](#-模型权重)
- [💻 环境配置](#-环境配置)
  - [文本理解](#文本理解-1)
  - [数学推理](#数学推理-1)
  - [多模态理解](#多模态理解-1)
- [📝 引用](#-引用)

## 🔑 主要特性

- **面向偏好优化的双视角动态加权。**
  Uni-DPO 联合建模了*哪些数据值得学习*（内在质量）和*模型仍存在哪些困难*（学习动态）。通过结合质量感知权重和性能感知权重，Uni-DPO 在整个优化过程中动态重新分配训练焦点。

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure1.png" width="80%" />
  </p>
</table>

- **质量感知加权过滤模糊的偏好对。**
  偏好数据的可靠性差异很大。Uni-DPO 利用偏好回答与拒绝回答之间的分数差值，为清晰、高质量的偏好对分配更高的权重，同时抑制嘈杂或模糊的样本。

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure2.png" width="80%" />
  </p>
</table>

- **性能感知加权缓解训练过程中的过拟合。**
  对于模型已经掌握的样本，即使它们质量很高，也并非总是最具信息量的。Uni-DPO 引入了一种类似焦点损失的稳定化性能权重，它会降低已拟合良好样本的权重，而强调那些困难但信息量大的样本，从而有效减少过拟合。

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure3.png" width="80%" />
  </p>
</table>

- **解耦数据质量与学习难度。**
  实证分析表明，数据质量（分数差值）和学习难度（奖励差值）之间的相关性很弱。Uni-DPO 显式地对这种不匹配进行建模，确保优化过程同时受到这两个维度的指导，而不是单独依赖其中任何一个。

<table align="center">
  <p align="center">
    <img src="/docs/figures/figure4.png" width="80%" />
  </p>
</table>

- **在文本、数学和多模态基准测试中达到业界领先性能。**
  Uni-DPO 在多种设置下始终优于 DPO 和 SimPO。

<table align="center">
  <p align="center">
    <img src="/docs/figures/table1.png" width="80%" />
  </p>
</table>

## 📚 数据集

我们发布 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO) [🤖](https://modelscope.cn/datasets/pspdada/Uni-DPO) **Uni-DPO 数据集**，包含三类训练数据：_文本理解_、_数学推理_、_多模态理解_。

### 文本理解

数据集目录下的 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Textual) `Textual` 文件夹包含*文本理解*任务的训练数据，涵盖 v0.1 和 v0.2 两个训练设置。若想要自己生产这些训练数据，可参考此[文档](/Textual/on_policy_data_gen/README_zh.md)。

<details>
<summary>生成数据流程</summary>

1. 下载 [🤗](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) `HuggingFaceH4/ultrafeedback_binarized` 数据集；
2. 使用 `decode.py` 生成输出并使用 `post_process.py` 清理；
3. 使用 `reward_model_annotate.py` 进行打分。

</details>

### 数学推理

*数学推理*训练数据位于数据集目录下的 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Math) `Math` 文件夹。若想要自己生产这些训练数据，可参考此[文档](/Math/README_zh.md)并使用此[脚本](/Math/data_generation/generate_data.sh)。

<details>
<summary>生成数据流程</summary>

1. 下载数学问题数据集 [🤗](https://huggingface.co/datasets/RLHFlow/numia_prompt_dpo1) `RLHFlow/numia_prompt_dpo1`；
2. 运行 `gen_samples.py` 以获得模型输出；
3. 使用规则奖励 `verifiable_reward_labeling.py` 与过程奖励模型 `progress_reward_labeling.py` 打分；
4. 运行 `get_uni_dpo_data.py` 构建偏好对。

</details>

评测数据位于 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math_eval_data.zip) `Math_eval_data.zip`。评测细节见[文档](/Math/README_zh.md)。

### 多模态理解

*多模态理解*的训练数据位于 [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Multimodal) `Multimodal` 文件夹。使用方式请参考[文档](/Multimodal/README_zh.md)。

## 📦 模型权重

我们开源了基于 **Uni-DPO** 方法训练的两种版本模型权重：**v0.1** 与 **v0.2**，覆盖多模型系列，包括 Llama3-8B，Gemma-2-9B-IT，以及 Qwen2.5。

| 基础模型                                                                             | 训练数据                                                                                                                             | 训练设置 |                                                                                            Uni-DPO 模型                                                                                            |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-GPT-4o/train.jsonl)                            |   v0.1   |                [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO) Llama-3-8B-Base-SFT-Uni-DPO                |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-Qwen2_5_72B/train.jsonl)                       |   v0.2   |    [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen) Llama-3-8B-Base-SFT-Uni-DPO-v2-Qwen    |
| [🤗](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) Llama-3-8B-Base-SFT   | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/tree/main/Textual)                                                             |   v0.2   |  [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4) Llama-3-8B-Base-SFT-Uni-DPO-v2-GPT-4   |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-GPT-4o/train.jsonl)        |   v0.1   |                [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO) Llama-3-8B-Instruct-Uni-DPO                |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-ArmoRM/train.jsonl)        |   v0.2   | [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM) Llama-3-8B-Instruct-Uni-DPO-v2-ArmoRM |
| [🤗](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Llama-3-8B-Instruct | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Llama-3-8B-Instruct-UltraFeedback-ArmoRM-GPT-4o/train.jsonl) |   v0.2   | [🤗](https://huggingface.co/psp-dada/Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o) [🤖](https://modelscope.cn/models/pspdada/Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o) Llama-3-8B-Instruct-Uni-DPO-v2-GPT-4o |
| [🤗](https://huggingface.co/google/gemma-2-9b-it) Gemma2-9B-IT                       | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/Gemma-2-9B-IT-UltraFeedback-GPT-4o/train.jsonl)              |   v0.1   |                          [🤗](https://huggingface.co/psp-dada/Gemma2-9B-IT-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Gemma2-9B-IT-Uni-DPO) Gemma2-9B-IT-Uni-DPO                           |
| [🤗](https://huggingface.co/Qwen/Qwen2.5-7B) Qwen2.5-7B                              | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Textual/UltraFeedback-GPT-4o/train.jsonl)                            |   v0.1   |                             [🤗](https://huggingface.co/psp-dada/Qwen2.5-7B-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Qwen2.5-7B-Uni-DPO) Qwen2.5-7B-Uni-DPO                              |
| [🤗](https://huggingface.co/Qwen/Qwen2.5-Math-7B) Qwen2.5-Math-7B                    | [🤗](https://huggingface.co/datasets/psp-dada/Uni-DPO/blob/main/Math/Train_Qwen_2_5_math_7B.jsonl)                                   |   v0.1   |                      [🤗](https://huggingface.co/psp-dada/Qwen2.5-Math-7B-Uni-DPO) [🤖](https://modelscope.cn/models/pspdada/Qwen2.5-Math-7B-Uni-DPO) Qwen2.5-Math-7B-Uni-DPO                      |

## 💻 环境配置

为确保与之前的研究进行公平的对比，我们尽可能地让训练和测试环境与原始实现保持一致。以下是各个任务所用环境的简要介绍。

### 文本理解

训练环境：参见[文档](/Textual/README_zh.md)。

- 基于 [SimPO](https://github.com/princeton-nlp/SimPO) 仓库构建
- 依赖 [alignment-handbook](https://github.com/huggingface/alignment-handbook)并使用 `transformers` 库中的 `Trainer` 类来构建 `UniDPOTrainer` 类，以实现 Uni-DPO 训练。

评测环境：主论文中报告的指标严格遵循以下四个评测环境：[Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto)、[AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval)、[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)、[SedarEval](https://github.com/wwn1233/sedareval)。附录中的下游任务评测则使用 [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 的配置。

### 数学推理

我们的训练与评测环境基于 [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) 仓库构建。详情请参见[文档](/Math/README_zh.md)。

- **训练数据构建：** 依赖 vLLM 进行模型部署与推理
- **训练：** 同样依赖 [alignment-handbook](https://github.com/huggingface/alignment-handbook)，并使用 `transformers` 的 `Trainer` 类构建 `UniDPOTrainer` 类以执行 Uni-DPO 训练
- **评测：** 评测代码基于 [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)

### 多模态理解

遵循 [MM-RLHF](https://github.com/Kwai-YuanQi/MM-RLHF)。详情请参见[文档](/Multimodal/README_zh.md)。

- **训练：** 我们的训练环境基于 [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) 构建，并提供最小修改版本及必要训练脚本
- **评测：** 我们的评测环境基于 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 构建，并提供运行评测所需脚本与必要文档

## 📝 引用

如果我们的模型/代码/数据/论文对您有帮助，请引用我们的论文并为我们点 ⭐️！

```bibtex
@inproceedings{peng2026unidpo,
  title     = {Uni-{DPO}: A Unified Paradigm for Dynamic Preference Optimization of {LLM}s},
  author    = {Shangpin Peng and Weinong Wang and Zhuotao Tian and Senqiao Yang and Xing W and Haotian Xu and Chengquan Zhang and Takashi Isobe and Baotian Hu and Min Zhang},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=G7DBGlgjjp}
}
```

## 📧 联系我们 <!-- omit in toc -->

如果您有任何问题、意见或建议，欢迎提交 issue 或 PR，共同推动该方向的研究进展。

## 🙏 致谢 <!-- omit in toc -->

感谢以下项目提供支持：

- 数据来源：[ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) / [RLHFlow/numia_prompt_dpo1](https://huggingface.co/datasets/RLHFlow/numia_prompt_dpo1) / [MM-RLHF](https://github.com/Kwai-YuanQi/MM-RLHF)
- 训练框架：[SimPO](https://github.com/princeton-nlp/SimPO) / [alignment-handbook](https://github.com/huggingface/alignment-handbook) / [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) / [LlamaFactory](https://github.com/hiyouga/LLaMAFactory)
- 评测工具：
  - 文本理解：[Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) / [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval) / [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) / [SedarEval](https://github.com/wwn1233/sedareval) / [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - 数学推理：[simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
  - 多模态理解：[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

## 许可证 <!-- omit in toc -->

[Apache License 2.0](/LICENSE)
