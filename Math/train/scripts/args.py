from dataclasses import dataclass, field
from typing import Optional

from alignment import DPOConfig


@dataclass
class ScriptArgs:
    """
    The arguments for the Uni-DPO training script.
    """

    ref_model: str = field(metadata={"help": "the location of the SFT model name or path"})
    train_data_path: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the training dataset name or path"},
    )
    eval_data_path: Optional[str] = field(
        default="/export/home/hanze/project/vllm-gen/uf_split0_offline_reward.json",
        metadata={"help": "the location of the eval dataset name or path"},
    )
    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "Whether to only train on 1000 samples"})
    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})


@dataclass
class TrainingArgs(DPOConfig):
    # Basic arguments
    loss_type: str = field(default="sigmoid")

    # SimPO hyperparameters
    simpo_gamma: float = field(default=1.0)

    # Uni-DPO hyperparameters
    dpo_length_norm: bool = field(
        default=True,
        metadata={"help": "Whether to average token-level log-probabilities when computing the DPO reward margin."},
    )
    uni_dpo_qual_eta: float = field(
        default=0.7,
        metadata={
            "help": "The scaling factor eta applied to the score margin when computing the quality weight w_qual."
        },
    )
    uni_dpo_tau_ref: float = field(
        default=2.0,
        metadata={
            "help": "The reference performance margin tau_ref used to calibrate the performance-based weight w_perf."
        },
    )
    uni_dpo_perf_gamma: float = field(
        default=3.0,
        metadata={"help": "The focal exponent gamma controlling how strongly well-fitted samples are down-weighted."},
    )
    uni_dpo_nll_lambda: float = field(
        default=0.001,
        metadata={
            "help": "The coefficient lambda of the calibrated NLL loss applied to difficult high-quality positives."
        },
    )
    uni_dpo_tau_good: float = field(
        default=2.5,
        metadata={"help": "The quality score threshold tau_good for activating the calibrated NLL loss."},
    )

    # Save model
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.35)
    # save_only_model: bool = field(default=True)
    # save_strategy: str = field(default="no")

    # wandb
    wandb_key: str = field(default="", metadata={"help": "the wandb key"})

    def __post_init__(self):
        super().__post_init__()
        self.use_ref_model: bool = self.loss_type not in ["orpo", "simpo"]
        self.gamma_beta_ratio: float = self.simpo_gamma / self.beta
