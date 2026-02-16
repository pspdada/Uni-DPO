"""
This file is used for rule-based correctness judgment of mathematical responses. Input a dataset containing:
{"prompt": "the prompt", "responses": ["response1", "response2" ...], "gt": ["gt1", "gt2" ...]}

This file will calculate the correctness of each input-output pair and output a new dataset where each sample contains:
{"prompt": "the prompt", "responses": [...], "rewards": [...], "correct": [...], "pred": [...]}
where the "correct" field is a list of bool indicating whether each output is correct

Use this to debug:
python -um reward_labeling.verifiable_reward_labeling_faster --dataset_name_or_path "output/Train_Qwen_numina_iter_1_data.jsonl" --output_file "output/Train_Qwen_numina_iter_1_data_v.jsonl"
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import HfArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[3]))
from Math.data_generation.scripts.utils.evaluate import get_batch_scores


@dataclass
class ScriptArguments:
    dataset_name_or_path: str = field(default="data.jsonl", metadata={"help": "path of the dataset name or path"})
    output_file: str = field(default="data_v.jsonl", metadata={"help": "path of the output file"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    output_file: str = script_args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset: Dataset = load_dataset("json", data_files=script_args.dataset_name_or_path, split="train")
    response_lists, ground_truths = [], []

    time1 = time.time()

    # Prepare data
    for sample in dataset:
        response_lists.append(sample["responses"])
        ground_truths.append(sample["gt"])

    # Calculate correctness
    results: list[dict[str]] = get_batch_scores(response_lists, ground_truths)

    processed_samples = []
    filtered_hard_cnt, filtered_empty_pred_cnt = 0, 0

    for i, sample in enumerate(dataset):
        eval_result: dict[str] = results[i]
        correct: list[bool] = eval_result["correct"]
        pred: list[str] = eval_result["pred"]
        responses = sample["responses"]

        # If all rewards are False, the problem is too hard (all incorrect), filter it out
        if not any(correct):
            filtered_hard_cnt += 1
            continue

        # Synchronized filtering: if pred contains empty strings, remove corresponding elements
        filtered_correct, filtered_pred, filtered_responses = [], [], []
        for p, c, r in zip(pred, correct, responses):
            if p.strip() != "":
                filtered_correct.append(c)
                filtered_pred.append(p)
                filtered_responses.append(r)
            else:
                filtered_empty_pred_cnt += 1

        # Handle case where PRM is applied before verifiable labeling
        if "progress_rewards" in sample:
            progress_rewards = sample["progress_rewards"]
            min_rewards = sample["min_rewards"]
            avg_rewards = sample["avg_rewards"]
            filtered_progress_rewards, filtered_min_rewards, filtered_avg_rewards = [], [], []
            for p, progress_r, min_r, avg_r in zip(pred, progress_rewards, min_rewards, avg_rewards):
                if p.strip() != "":
                    filtered_progress_rewards.append(progress_r)
                    filtered_min_rewards.append(min_r)
                    filtered_avg_rewards.append(avg_r)

        # Update fields in sample
        sample.update({"correct": filtered_correct, "pred": filtered_pred, "responses": filtered_responses})
        if "progress_rewards" in sample:
            sample.update(
                {
                    "progress_rewards": filtered_progress_rewards,
                    "min_rewards": filtered_min_rewards,
                    "avg_rewards": filtered_avg_rewards,
                }
            )
        processed_samples.append(sample)

    ## Save as jsonl
    with open(output_file, "w", encoding="utf8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Print statistics
    print(f"Samples filtered due to being too hard: {filtered_hard_cnt}")
    print(f"Predictions filtered due to empty strings: {filtered_empty_pred_cnt}")
    print(f"Number of saved samples: {len(processed_samples)}")
    print("Runtime:", time.time() - time1)
