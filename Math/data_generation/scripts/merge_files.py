"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""

import json
import random
from dataclasses import dataclass, field

from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    base_path: str = field(metadata={"help": "the location of the dataset name or path"})
    output_file: str = field(metadata={"help": "the location of the output file"})
    num_datasets: int = field(metadata={"help": "the location of the output file"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    assert script_args.num_datasets > 0, "num_datasets should be greater than 0"

    data_files: list[str] = [script_args.base_path + "_" + str(i) + ".jsonl" for i in range(script_args.num_datasets)]

    gathered_data = []
    for data_file in data_files:
        # 从 JSONL 文件加载数据
        with open(data_file, "r", encoding="utf8") as f:
            data = [json.loads(line) for line in f]

        print("Load ", len(data), "samples from ", data_file)
        gathered_data.extend(data)

    random.shuffle(gathered_data)

    print("I collect ", len(gathered_data), "samples")

    with open(script_args.output_file, "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write("\n")
