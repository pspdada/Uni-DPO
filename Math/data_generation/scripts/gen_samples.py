import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

GPU_UTILS = 0.90  # GPU memory utilization ratio for vLLM
ENFORCE_EAGER = False  # Whether to use eager mode


@dataclass
class MyArgs:
    model_name_or_path: str = field(metadata={"help": "the location of the SFT model name or path"})
    dataset_name_or_path: str = field(metadata={"help": "the location of the dataset name or path"})
    output_dir: str = field(metadata={"help": "the location of the output file, without .jsonl suffix!!"})
    local_index: int = field(metadata={"help": "the local index of the agent"})
    my_world_size: int = field(metadata={"help": "the total number of the agents"})

    K: Optional[int] = field(default=8, metadata={"help": "the number of generations per prompt"})
    max_new_tokens: Optional[int] = field(default=4096, metadata={"help": "the maximum length of the new tokens"})
    gpu_utils: Optional[float] = field(default=GPU_UTILS)
    seed: Optional[int] = field(default=42, metadata={"help": "the random seed"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "the temperature"})
    dataset_key: Optional[str] = field(default="problem", metadata={"help": "the key of the dataset"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    batch_size: Optional[int] = field(default=10, metadata={"help": "Batch size for processing"})
    truncate_length: Optional[int] = field(
        default=4000,
        metadata={
            "help": "The maximum model output length of token ids considered when building the training dataset. Outputs exceeding this length are considered garbled and the corresponding response will be removed. If set to -1, all model outputs will be retained."
        },
    )


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return


def main():
    parser = HfArgumentParser(MyArgs)
    args = parser.parse_args_into_dataclasses()
    args: MyArgs = args[0]

    # Set random seeds
    _set_seed(args.seed)

    # Load data
    data: Dataset = load_dataset(args.dataset_name_or_path, split="train")
    one_num_share = int(data.num_rows / args.my_world_size)
    data = data.select(np.arange(args.local_index * one_num_share, (args.local_index + 1) * one_num_share))

    # Check for already processed files to avoid duplicate processing
    # output_file format: ".../Numina_iter_1_0.jsonl"
    output_file = args.output_dir + "_" + str(args.local_index) + ".jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_prompts = set()
    qs_key = args.dataset_key

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf8") as f:
            for line in f:
                processed_data = json.loads(line)
                processed_prompts.add(processed_data[qs_key])

    # Filter out already processed samples
    data = data.filter(lambda x: x[qs_key] not in processed_prompts)
    print(f"Remaining samples to process: {data.num_rows}")
    if data.num_rows == 0:
        print(f"No new samples to process for index {args.local_index}.")
        return

    # Load model and tokenizer
    model_path = args.model_name_or_path
    model_dir = os.getenv("MODEL_PATH")
    print("Model path:", model_path, "Model dir:", model_dir)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tokenizer=model_path,
        gpu_memory_utilization=args.gpu_utils,
        download_dir=model_dir,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        load_format="auto",
        seed=args.seed,
        task="generate",
        enforce_eager=ENFORCE_EAGER,
    )
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=model_dir,
        use_fast=True,
    )

    # Apply chat template to prompts
    # Example format:
    # <|im_start|>system\nPlease reason step by step, and put your final answer within
    # \boxed{}.<|im_end|>\n<|im_start|>user\n{Question} Let's think step by step and
    # output the final answer within \boxed{}<|im_end|>\n<|im_start|>assistant
    data = data.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": x[qs_key] + " Let's think step by step and output the final answer within \\boxed{}",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        }
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.8,
        max_tokens=args.max_new_tokens,
        n=args.K,
        stop_token_ids=[tokenizer.eos_token_id] + args.eos_ids,
    )

    # Print start time
    start_time = datetime.now()
    print(f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    batch_size = args.batch_size
    for i in tqdm(range(0, data.num_rows, batch_size), desc="Processing Batches"):
        batch: dict[str, list] = data[i : i + batch_size]
        if not batch or not batch["prompt"]:
            continue

        prompts: list[str] = batch["prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        # Collect results
        with open(output_file, "a", encoding="utf8") as f:
            # Variable `output` contains the prompt and list of answers for a single question
            for j, output in enumerate(outputs):
                # Filter out responses where the number of token ids exceeds truncate_length
                valid_responses: list[str] = []
                for out in output.outputs:
                    completion_length = len(output.prompt_token_ids) + len(out.token_ids)
                    if args.truncate_length < 0 or completion_length <= args.truncate_length:
                        valid_responses.append(out.text)

                if valid_responses:
                    json.dump(
                        {
                            qs_key: batch[qs_key][j],
                            "prompt": batch["prompt"][j],
                            "gt": batch["gt"][j],
                            "responses": valid_responses,
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
                else:
                    print(f"Skipping sample {i + j} as all responses were removed due to length.")

    print(f"Index {args.local_index} of {args.my_world_size} finished processing.")
    print(f"Results saved to {output_file}")

    # Print end time
    end_time = datetime.now()
    print(f"Processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {end_time - start_time}")


if __name__ == "__main__":
    main()
