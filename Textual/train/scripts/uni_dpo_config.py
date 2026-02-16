from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class UniDPOConfig(TrainingArguments):
    r"""
    UniDPOConfig collects all training arguments related to the [`UniDPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 2.0):
            The beta factor in SimPO loss.
        gamma_beta_ratio (`float`, defaults to 0.25):
            The ratio between the target reward margin (gamma) and beta in SimPO loss.
        sft_weight (`float`, defaults to 0.0):
            SFT loss weight added to the SimPO loss (0.0 is not using SFT).
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, defaults to `sigmoid`):
            The type of loss to use. This argument is required if you want to use the default data collator.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[dict]`, *optional*):
            dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """

    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_samples: Optional[int] = None  # max number of samples to use for training , useful for debugging

    use_ref_model: bool = False
    ref_model: str = None

    gamma_beta_ratio: float = 0.25

    label_smoothing: float = 0
    disable_dropout: bool = True
    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None
    model_init_kwargs: Optional[dict] = None
    dataset_num_proc: Optional[int] = None

    # Uni-DPO configs
    pref_loss: Literal["simpo", "uni_dpo"] = "uni_dpo"
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    dpo_length_norm: bool = True  # whether to apply length normalization in DPO loss
    beta: float = 2.0
    uni_dpo_qual_eta: float = 0.1
    uni_dpo_tau_ref: float = 0.8
    uni_dpo_perf_gamma: float = 1.0
    uni_dpo_nll_lambda: float = 0.01
    uni_dpo_tau_good: float = 7.0
