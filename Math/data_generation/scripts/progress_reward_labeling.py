"""
Use Process Reward Model (PRM) to label the generated dataset
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, TokensPrompt

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

GPU_UTILS = 0.92


@dataclass
class MyArguments:
    input_file: str = field(metadata={"help": "the location of the dataset name or path, xxx/xxx.jsonl"})
    output_name: str = field(metadata={"help": "the name of the output file, without .jsonl suffix!! xxx/xxx"})
    local_index: int = field(metadata={"help": "the local index of the agent"})
    my_world_size: int = field(metadata={"help": "the total number of the agents"})

    PRM_model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-PRM-7B", metadata={"help": "the PRM model"})
    batch_size: Optional[int] = field(default=5, metadata={"help": "Batch size for processing"})

    gpu_utils: Optional[float] = field(default=GPU_UTILS)
    seed: Optional[int] = field(default=42, metadata={"help": "the random seed"})
    dataset_key: Optional[str] = field(default="problem", metadata={"help": "the key of the dataset"})
    eos_ids: list[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


def main():
    parser = HfArgumentParser(MyArguments)
    args = parser.parse_args_into_dataclasses()
    args: MyArguments = args[0]

    # Set random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data from JSONL file
    input_file = args.input_file
    with open(input_file, "r", encoding="utf8") as f:
        data = [json.loads(line) for line in f]

    data_size = len(data)
    one_num_share = int(data_size / args.my_world_size)
    data = data[args.local_index * one_num_share : (args.local_index + 1) * one_num_share]

    output_file = args.output_name + "_" + str(args.local_index) + ".jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_prompts = set()
    qs_key = args.dataset_key

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf8") as f:
            for line in f:
                processed_data = json.loads(line)
                processed_prompts.add(processed_data[qs_key])

    # Filter out already processed samples
    data = [sample for sample in data if sample[qs_key] not in processed_prompts]
    data_size = len(data)
    print(f"Remaining samples to process: {data_size}")
    if data_size == 0:
        return

    # Load model and tokenizer
    model_dir = os.getenv("MODEL_PATH")
    model_path: str = args.PRM_model_name_or_path
    print("Model path:", model_path)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tokenizer=model_path,
        task="reward",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_utils,
        download_dir=model_dir,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)

    batch_size = args.batch_size
    token1 = tokenizer.encode("<extra_0>")[0]
    token2 = tokenizer.encode("<|im_end|>")[0]
    token3 = tokenizer.encode("<|endoftext|>")[0]

    # Print start time
    start_time = datetime.now()
    print(f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for i in tqdm(range(0, data_size, batch_size), desc="Processing Batches"):
        batch: list[dict[str]] = data[i : i + batch_size]

        # Construct prompts
        prompts: list[TokensPrompt] = []
        for sample in batch:
            problem = sample["problem"]
            for response in sample["responses"]:
                response = response.split("\n\n")
                prompt: list[int] = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": "<extra_0>".join(response) + "<extra_0>"},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=4090,
                )
                if prompt[-3] != token1 or prompt[-2] != token2 or prompt[-1] != token3:
                    prompt[-2] = token1
                    prompt[-1] = token2
                    prompt.append(token3)
                prompts.append(TokensPrompt(prompt_token_ids=prompt))

        # Call LLM to generate results
        outputs = llm.encode(prompts, use_tqdm=False)

        # Process generated results
        output_rewards: list[list[float]] = [[s[1] for s in out.outputs.data.tolist()] for out in outputs]

        output_idx = 0
        for sample in batch:
            problem: str = sample["problem"]
            responses: list[str] = sample["responses"]

            num_responses = len(responses)
            current_output_rewards: list[list[float]] = output_rewards[output_idx : output_idx + num_responses]
            output_idx += num_responses

            # Update sample fields
            sample["progress_rewards"] = current_output_rewards
            sample["min_rewards"] = [round(min(r), 3) for r in current_output_rewards]
            sample["avg_rewards"] = [round(np.mean(r), 3) for r in current_output_rewards]

            # Save results
            with open(output_file, "a", encoding="utf8") as f:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

    print(f"Index {args.local_index} of {args.my_world_size} finished processing.")
    print(f"Results saved to {output_file}")

    # Print end time
    end_time = datetime.now()
    print(f"Processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {end_time - start_time}")


if __name__ == "__main__":
    main()
