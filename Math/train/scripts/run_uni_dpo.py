# ruff: noqa: E402
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


import torch
from alignment import H4ArgumentParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.trainer_utils import TrainOutput
from trl import ModelConfig

sys.path.append(str(Path(__file__).resolve().parents[3]))
from Math.train.scripts.args import ScriptArgs, TrainingArgs
from Math.train.scripts.data import prepare_data
from Math.train.scripts.trainer import UniDPOTrainer
from Math.train.scripts.utils import TRAINER_STATE_NAME, plot_train_dynamics


def main() -> None:
    parser = H4ArgumentParser((ScriptArgs, TrainingArgs, ModelConfig))
    args: tuple[ScriptArgs, TrainingArgs, ModelConfig] = parser.parse()
    script_args: ScriptArgs = args[0]
    training_args: TrainingArgs = args[1]
    model_config: ModelConfig = args[2]
    del args, parser

    if training_args.wandb_key:
        try:
            import wandb

            wandb.login(key=training_args.wandb_key)
            training_args.report_to = ["wandb"]
        except Exception as e:
            print(f"Failed to login to wandb: {e}")
            training_args.report_to = []
    else:
        training_args.report_to = []

    # 1. load a pretrained model
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.train()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ref_name: str = script_args.ref_model if script_args.ref_model else model_config.model_name_or_path

    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        ref_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    ref_model.requires_grad_(False)
    ref_model.eval()

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        ref_model.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

    tokenizer.bos_token = tokenizer.eos_token

    # 2. Load the paired dataset
    train_dataset = prepare_data(
        data_path=script_args.train_data_path,
        sanity_check=script_args.sanity_check,
        eot_token=script_args.eot_token,
    )

    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = None
    # eval_dataset = prepare_data(
    #     data_path=script_args.eval_data_path,
    #     sanity_check=True,
    #     eot_token=script_args.eot_token,
    # )

    # 4. initialize the DPO trainer
    dpo_trainer = UniDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 5. train
    train_result: TrainOutput = dpo_trainer.train()

    # 6. save
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dpo_trainer.save_model(output_dir)
    dpo_trainer.model.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint"))

    dpo_trainer.log_metrics("train", train_result.metrics)
    dpo_trainer.save_metrics("train", train_result.metrics)
    dpo_trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    plot_train_dynamics(output_dir)


if __name__ == "__main__":
    main()
