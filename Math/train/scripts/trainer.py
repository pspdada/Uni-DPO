import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from typing_extensions import override

sys.path.append(str(Path(__file__).resolve().parents[3]))
from Math.train.scripts.args import TrainingArgs
from Math.train.scripts.utils import rank0_print


class UniDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TrainingArgs] = None,
        data_collator=None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[dict] = None,
        ref_model_init_kwargs: Optional[dict] = None,
        model_adapter_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
        )

        # dpo hyperparams
        self.use_ref_model = args.use_ref_model
        self.label_smoothing = args.label_smoothing
        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.loss_type = str(args.loss_type).lower().replace("-", "_")

        self.dpo_length_norm = args.dpo_length_norm
        self.qual_eta = args.uni_dpo_qual_eta
        self.tau_ref = args.uni_dpo_tau_ref
        self.perf_gamma = args.uni_dpo_perf_gamma
        self.nll_lambda = args.uni_dpo_nll_lambda
        self.tau_good = args.uni_dpo_tau_good

        rank0_print(f"[UniDPOTrainer] Use {self.loss_type=}")
        rank0_print(f"[UniDPOTrainer] Use {self.beta=}")
        rank0_print(f"[UniDPOTrainer] Use {self.dpo_length_norm=}")
        rank0_print(f"[UniDPOTrainer] Use {self.qual_eta=}")
        rank0_print(f"[UniDPOTrainer] Use {self.tau_ref=}")
        rank0_print(f"[UniDPOTrainer] Use {self.perf_gamma=}")
        rank0_print(f"[UniDPOTrainer] Use {self.nll_lambda=}")
        rank0_print(f"[UniDPOTrainer] Use {self.tau_good=}")

    def compute_preference_loss(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        reference_chosen_logps: Optional[Tensor],
        reference_rejected_logps: Optional[Tensor],
        score_chosen: Optional[Tensor] = None,
        score_rejected: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Computes loss for preference learning.
        """
        # 初始化返回值（有些 dpo 类型不会返回其中某些值，因此初始化为 None）
        losses, chosen_rewards, rejected_rewards = None, None, None
        logits, pi_logratios, ref_logratios = None, None, None
        w_qual, w_perf, nll_coeffi = None, None, None
        dpop_penalty_term = None

        if not self.use_ref_model:
            if self.loss_type.lower() == "simpo":
                losses, pi_logratios, logits = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type.lower() == "uni_dpo":
            (
                losses,
                chosen_rewards,
                rejected_rewards,
                logits,
                pi_logratios,
                ref_logratios,
                w_qual,
                w_perf,
                nll_coeffi,
            ) = self.uni_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                score_chosen=score_chosen,
                score_rejected=score_rejected,
            )
        elif self.loss_type == "dpo_positive":
            losses, chosen_rewards, rejected_rewards, dpop_penalty_term = self.dpo_positive_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        else:  # vanilla dpo, go to trl implementation
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        return (
            losses,
            chosen_rewards,
            rejected_rewards,
            logits,
            pi_logratios,
            ref_logratios,
            w_qual,
            w_perf,
            nll_coeffi,
            dpop_penalty_term,
        )

    def simpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        device: torch.device = self.accelerator.device

        # policy logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = (policy_chosen_logps - policy_rejected_logps).to(device)  # Shape: (batch_size,)
        logits = pi_logratios - self.gamma_beta_ratio
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss, pi_logratios, logits

    def compute_w_qual(self, score_chosen: "torch.Tensor", score_rejected: "torch.Tensor", qual_eta: float = 1.0):
        tao_judge = F.sigmoid(qual_eta * (score_chosen - score_rejected))
        return tao_judge

    def uni_dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        score_chosen: Optional[Tensor],
        score_rejected: Optional[Tensor],
    ) -> tuple:
        w_qual, w_perf, nll_coeffi = None, None, None
        device: torch.device = self.accelerator.device

        if isinstance(score_chosen, Tensor) and isinstance(score_rejected, Tensor):
            score_chosen = score_chosen.to(device)
            score_rejected = score_rejected.to(device)

        # Shape: (batch_size,)
        pi_logratios = (policy_chosen_logps - policy_rejected_logps).to(device)
        ref_logratios = (reference_chosen_logps - reference_rejected_logps).to(device)

        # (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
        # Shape: (batch_size,)
        logits: Tensor = pi_logratios - ref_logratios

        # quality weight, shape: (batch_size,)
        w_qual: Tensor = self.compute_w_qual(score_chosen, score_rejected, qual_eta=self.qual_eta)

        # performance weight
        w_perf: Tensor = (1 - F.sigmoid(self.beta * pi_logratios - self.tau_ref)) ** self.perf_gamma

        losses: Tensor = -w_qual * w_perf * F.logsigmoid(self.beta * logits)

        if self.nll_lambda > 0.0:
            # Only generate positive loss when reference model probability is higher than policy model
            nll_coeffi = self.sign(reference_chosen_logps - policy_chosen_logps)
            # Filter out low-quality samples to ensure loss only comes from high-quality samples
            nll_coeffi = torch.where(score_chosen >= self.tau_good, nll_coeffi, 0)
            losses -= self.nll_lambda * nll_coeffi * policy_chosen_logps

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return (
            losses,
            chosen_rewards,
            rejected_rewards,
            logits,
            pi_logratios,
            ref_logratios,
            w_qual,
            w_perf,
            nll_coeffi,
        )

    # Modified from https://github.com/abacusai/smaug/issues/2#issuecomment-2075172150
    def dpo_positive_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        device = self.accelerator.device

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = Tensor([0], dtype=pi_logratios.dtype, device=device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(device)
        ref_logratios = ref_logratios.to(device)

        logits = pi_logratios - ref_logratios

        # torch.clamp(x, min=0) 和 torch.maximum(torch.zeros_like(x), x) 在功能上是等价的
        dpop_penalty_term = torch.clamp(reference_chosen_logps - policy_chosen_logps, min=0)

        positive_sft_margin = self.nll_lambda * dpop_penalty_term

        losses = (
            -F.logsigmoid(self.beta * logits - positive_sft_margin) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits + positive_sft_margin) * self.label_smoothing
        )

        chosen_rewards = self.beta * (policy_chosen_logps.to(device) - reference_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(device) - reference_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards, dpop_penalty_term

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        device: torch.device = self.accelerator.device
        batch_size = batch["chosen_labels"].shape[0]

        concatenated_batch: dict[str] = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=device,
        )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        )
        all_logits: Tensor = outputs.logits  # shape: (batch_size * 2, seq_len, vocab_size)

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"].clone(),
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        if self.dpo_length_norm is True:
            all_logps = all_logps / size_completion
        elif self.loss_type.lower() in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / size_completion

        if self.args.rpo_alpha is not None:

            def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
                if not self.is_encoder_decoder:
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                # Enable model parallelism
                labels = labels.to(logits.device)
                loss = loss_fct(logits, labels)
                return loss

            labels = concatenated_batch["concatenated_labels"].clone()
            nll_loss = cross_entropy_loss(all_logits[:batch_size], labels[:batch_size])
        else:
            nll_loss = None  # noqa

        chosen_logps, rejected_logps = all_logps[:batch_size], all_logps[batch_size:]
        chosen_logits, rejected_logits = all_logits[:batch_size], all_logits[batch_size:]
        chosen_length, rejected_length = size_completion[:batch_size], size_completion[batch_size:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_length,
            rejected_length,
            nll_loss,
        )

    def ref_model_forward(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Computes log probabilities of the reference model.
        """
        if not self.use_ref_model:
            return None, None

        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (ref_chosen_logps, ref_rejected_logps, *_) = self.concatenated_forward(self.model, batch)
            else:
                (ref_chosen_logps, ref_rejected_logps, *_) = self.concatenated_forward(self.ref_model, batch)

        return (ref_chosen_logps, ref_rejected_logps)

    def DMC(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Detach, Mean, and CPU."""
        return input_tensor.detach().mean().cpu()

    def sign(self, input_tensor: torch.Tensor):
        """Limit the result to non-negative values"""
        return torch.clamp(torch.sign(input_tensor), min=0)

    @override
    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Compute the DPO loss and other metrics for the given batch of inputs for train or test.

        Return:
            - losses: The DPO loss for the batch.
            - metrics: A dictionary of metrics computed for the batch.
        """
        # rank0_print("Starting policy model forward pass...")
        (
            policy_chosen_logps,  # Shape: (batch_size,)
            policy_rejected_logps,  # Shape: (batch_size,)
            policy_chosen_logits,  # Shape: (batch_size, seq_len, vocab_size)
            policy_rejected_logits,  # Shape: (batch_size, seq_len, vocab_size)
            chosen_length,  # Shape: (batch_size,)
            rejected_length,  # Shape: (batch_size,)
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)
        # rank0_print("Finished policy model forward pass.")

        reference_chosen_logps, reference_rejected_logps = self.ref_model_forward(batch)
        # rank0_print("Finished reference model forward pass.")

        if "score_chosen" in batch and "score_rejected" in batch:
            score_chosen: Tensor = torch.tensor(batch["score_chosen"], dtype=policy_chosen_logps.dtype)
            score_rejected: Tensor = torch.tensor(batch["score_rejected"], dtype=policy_chosen_logps.dtype)
        else:
            score_chosen, score_rejected = None, None

        # rank0_print("Starting to compute preference loss...")
        (
            losses,
            chosen_rewards,
            rejected_rewards,
            logits,
            pi_logratios,
            ref_logratios,
            w_qual,
            w_perf,
            nll_coeffi,
            dpop_penalty_term,
        ) = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            score_chosen=score_chosen,
            score_rejected=score_rejected,
        )
        # rank0_print("Computed preference loss.")

        if self.args.rpo_alpha is not None:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        # 记录指标和中间变量，如果为 None 则不保存，防止后续报错
        metrics = {}

        scalar_metrics = {
            "policy_logps/chosen": policy_chosen_logps,
            "policy_logps/rejected": policy_rejected_logps,
            "reference_logps/chosen": reference_chosen_logps,
            "reference_logps/rejected": reference_rejected_logps,
            "policy_logits/chosen": policy_chosen_logits.mean(),
            "policy_logits/rejected": policy_rejected_logits.mean(),
            "logits": logits,
            "pi_logratios": pi_logratios,
            "ref_logratios": ref_logratios,
            "w_qual": w_qual,
            "w_perf": w_perf,
            "nll_coeffi": nll_coeffi,
            "dpop_penalty_term": dpop_penalty_term,
        }
        if isinstance(score_chosen, Tensor) and isinstance(score_rejected, Tensor):
            scalar_metrics["score_chosen"] = score_chosen
            scalar_metrics["score_rejected"] = score_rejected
            scalar_metrics["accuracy"] = (score_chosen > score_rejected).float()
            scalar_metrics["score_margin"] = score_chosen - score_rejected

        for key, value in scalar_metrics.items():
            if value is not None and isinstance(value, Tensor):
                metrics[key] = self.DMC(value)

        if isinstance(chosen_rewards, Tensor) and isinstance(rejected_rewards, Tensor):
            metrics["rewards/chosen"] = self.DMC(chosen_rewards)
            metrics["rewards/rejected"] = self.DMC(rejected_rewards)
            metrics["rewards/accuracies"] = self.DMC((chosen_rewards > rejected_rewards).float())
            metrics["rewards/margins"] = self.DMC(chosen_rewards - rejected_rewards)

        # rank0_print("Computed all metrics for the batch.")
        # rank0_print(f"Losses: {losses}")
        return losses.mean(), metrics
