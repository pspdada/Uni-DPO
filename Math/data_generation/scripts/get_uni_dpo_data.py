import json
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm import tqdm
from transformers import HfArgumentParser

"""
Each line is a JSON object containing the following fields:
- problem: Problem description
- prompt: Prompt information
- gt: Ground truth answer
- responses: Multiple model responses
- correct: Whether each response is correct (list of bool)
- min_rewards: Minimum score for each response
- avg_rewards: Average score for each response

The fields responses, correct, min_rewards, and avg_rewards have equal length and correspond by position.
"""

TRY_MAX_CNT = 200  # Maximum number of attempts


@dataclass
class MyArguments:
    input_file: str = field(required=True, metadata={"help": "Input file path"})
    output_file: str = field(required=True, metadata={"help": "Output file path"})
    score_margin: Optional[float] = field(
        default=5, metadata={"help": "Minimum score diff between positive and negative samples"}
    )
    reward_type: Optional[str] = field(
        default="min",
        metadata={"help": "Options: min or avg, use min_rewards or avg_rewards as the basis for construction"},
    )
    pos_strategy: Optional[str] = field(
        default="max", metadata={"help": 'Positive sample selection strategy, options: "max" or "rand"'}
    )
    neg_strategy: Optional[str] = field(
        default="positive",
        metadata={"help": 'Negative sample selection strategy, options: "min", "rand", or "positive"'},
    )
    max_data_num: Optional[int] = field(default=10000, metadata={"help": "Maximum number of data samples"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


def load_jsonl(file_path: str) -> list[dict[str]]:
    """Load JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_positive_sample(correct: list[bool], rewards, strategy):
    """Select positive sample"""
    valid_indices = [i for i, is_correct in enumerate(correct) if is_correct and rewards[i] > 6]
    if not valid_indices:
        return None

    if strategy == "max":
        return max(valid_indices, key=lambda i: rewards[i])
    elif strategy == "rand":
        return random.choice(valid_indices)


def get_negative_sample(correct: list[bool], rewards, strategy, positive_score, score_margin):
    """Select negative sample"""
    invalid_indices = [
        i for i, is_correct in enumerate(correct) if not is_correct and rewards[i] <= (positive_score - score_margin)
    ]

    if strategy == "min":
        if invalid_indices:
            return min(invalid_indices, key=lambda i: rewards[i])

    elif strategy == "rand":
        if invalid_indices:
            return random.choice(invalid_indices)

    elif strategy == "positive":
        # First, try to select low-scoring correct responses
        low_score_valid_indices = [
            i for i, is_correct in enumerate(correct) if is_correct and rewards[i] <= (positive_score - score_margin)
        ]
        if low_score_valid_indices:
            return random.choice(low_score_valid_indices)

        # If no qualifying correct responses, randomly select a low-scoring incorrect response
        low_score_indices = [i for i in invalid_indices if rewards[i] <= (positive_score - score_margin)]
        if low_score_indices:
            return random.choice(low_score_indices)

    return None


def normalize_score_margin(dataset: list[dict[str]]) -> list[dict[str]]:
    margins = []
    for item in dataset:
        if "score_margin" in item:
            margins.append(item["score_margin"])
        else:
            margins.append(item["score_chosen"] - item["score_rejected"])

    margin_mean = np.mean(margins)
    margin_std = np.std(margins) + 1e-6

    for i, item in enumerate(dataset):
        dataset[i]["score_margin_normalized"] = (margins[i] - margin_mean) / margin_std
    return dataset


def construct_dataset(
    data: list[dict[str]],
    reward_type: str,
    pos_strategy: str,
    neg_strategy: str,
    max_data_num: int,
    score_margin: float,
) -> list[dict[str]]:
    """Construct preference training dataset"""
    dataset: list[dict[str]] = []
    try_cnt = 0
    with tqdm(total=max_data_num, desc="Building Dataset", unit="samples") as pbar:
        while True:
            while len(dataset) < max_data_num and try_cnt <= TRY_MAX_CNT:
                for item in data:
                    try:
                        responses: list[str] = item["responses"]
                        correct: list[bool] = item["correct"]
                        rewards: list[float] = item["min_rewards"] if reward_type == "min" else item["avg_rewards"]
                        if not responses or not correct or not rewards:
                            continue

                        if max(rewards) <= 1.1:
                            rewards = [r * 10 for r in rewards]  # Align scores

                        # Select positive sample index
                        chosen_idx = get_positive_sample(correct, rewards, pos_strategy)
                        if chosen_idx is None:
                            continue

                        # Select negative sample index
                        positive_score = rewards[chosen_idx]
                        rejected_idx = get_negative_sample(correct, rewards, neg_strategy, positive_score, score_margin)
                        if rejected_idx is None:
                            continue

                        if rewards[chosen_idx] < rewards[rejected_idx]:
                            continue

                        # Build data dictionary
                        dataset.append(
                            {
                                "prompt": item["prompt"],
                                "chosen": responses[chosen_idx],
                                "rejected": responses[rejected_idx],
                                "score_chosen": rewards[chosen_idx],
                                "score_rejected": rewards[rejected_idx],
                                "score_margin": rewards[chosen_idx] - rewards[rejected_idx],
                                "correct_chosen": correct[chosen_idx],
                                "correct_rejected": correct[rejected_idx],
                            }
                        )
                        pbar.update(1)

                        # Remove used content (sort to avoid index errors)
                        indices_to_remove = sorted([chosen_idx, rejected_idx], reverse=True)
                        for idx in indices_to_remove:
                            del item["responses"][idx]
                            del item["correct"][idx]
                            del item["min_rewards"][idx]
                            del item["avg_rewards"][idx]

                        # Stop if target quantity is reached
                        if len(dataset) >= max_data_num:
                            return dataset
                    except Exception:
                        pass

                try_cnt += 1
            score_margin -= 0.5
            try_cnt = 0

            if score_margin <= 0:
                print("Failed to find enough samples after multiple attempts.")
                return dataset
            print(f"Failed to find enough samples, trying again with score_margin: {score_margin}")


if __name__ == "__main__":
    parser = HfArgumentParser(MyArguments)
    args = parser.parse_args_into_dataclasses()
    args: MyArguments = args[0]

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data: list[dict[str]] = load_jsonl(args.input_file)

    # Construct dataset
    dataset = construct_dataset(
        data,
        reward_type=args.reward_type,
        pos_strategy=args.pos_strategy,
        neg_strategy=args.neg_strategy,
        max_data_num=args.max_data_num,
        score_margin=args.score_margin,
    )

    dataset = normalize_score_margin(dataset)

    # Save dataset to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset construction completed. Total samples: {len(dataset)}")
    print(f"Output saved to: {args.output_file}")
