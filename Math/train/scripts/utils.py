import json
import math
import os

import torch.distributed as dist
from matplotlib import pyplot as plt

TRAINER_STATE_NAME = "trainer_state.json"

KEY_TO_PLOT = [
    "loss",
    "policy_logps/chosen",
    "policy_logps/rejected",
    "policy_logits/chosen",
    "policy_logits/rejected",
    "ref_logps/chosen",
    "ref_logps/rejected",
    "grad_norm",
    "positive_sft_coeffi",
    "fix_coeffi",
    "focal_item_coeffi",
    "logits",
    "policy_logratios",
    "ref_logratios",
]


def smooth(scalars: list[float]) -> list[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


DIST_AVAILABLE: bool = False


def rank0_print(*args, **kwargs):
    """Print only on rank 0 process in distributed training."""
    global DIST_AVAILABLE

    if DIST_AVAILABLE or dist.is_available() and dist.is_initialized():
        DIST_AVAILABLE = True
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def plot_train_dynamics(save_dictionary: str, keys: list[str] = KEY_TO_PLOT) -> None:
    """
    Plots the training dynamics of the specified keys from the trainer state JSON file.
    """
    try:
        print(f"Plotting training dynamics for {save_dictionary}...")
        plt.switch_backend("agg")
        file_path: str = os.path.join(save_dictionary, TRAINER_STATE_NAME)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return

        with open(file_path, encoding="utf-8") as f:
            data: dict[str] = json.load(f)

        print(f"Save figure to {save_dictionary}.")
        for key in keys:
            try:
                steps: list[int] = []
                metrics: list = []
                for i in range(len(data["log_history"])):
                    if key in data["log_history"][i]:
                        steps.append(data["log_history"][i]["step"])
                        metrics.append(data["log_history"][i][key])
                if not steps or not metrics:
                    print(f"No data found for {key}.")
                    continue

                plt.figure()
                plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
                plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
                plt.title(f"training {key} of {save_dictionary}")
                plt.xlabel("step")
                plt.ylabel(key)
                plt.legend()
                figure_path = os.path.join(save_dictionary, f"training_{key.replace('/', '_')}.pdf")
                plt.savefig(figure_path, format="pdf")
                print(f"Figure of training {key} saved at {figure_path}")
                plt.close()
            except Exception as e:
                print(f"Error plotting {key}: {e}")
                continue
        print("Plotting completed.")
    except Exception as e:
        print(f"Error plotting training dynamics: {e}")
