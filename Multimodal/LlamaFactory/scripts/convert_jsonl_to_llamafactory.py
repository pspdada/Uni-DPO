import json
import re
import sys

input_path = "your/path/to/Train_Qwen_2_5_math_7B.jsonl"
output_prefix = "llama_factory"


def extract_instruction(prompt):
    # 使用正则表达式匹配 user\n之后，<|im 之前的内容
    match = re.search(r"user\n(.*?)<\|im", prompt, re.DOTALL)
    if not match:
        raise ValueError("Failed to extract instruction from prompt.")
    instruction = match.group(1).strip()
    return instruction


def convert_jsonl_to_llamafactory():
    data = []

    with open(input_path, encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                sys.exit(1)

            try:
                instruction = extract_instruction(entry["prompt"])
            except ValueError as e:
                print(f"Error extracting instruction on line {line_number}: {e}")
                sys.exit(1)

            new_entry = {
                "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                "instruction": instruction,
                "chosen": entry["chosen"],
                "rejected": entry["rejected"],
                "score_chosen": entry["score_chosen"],
                "score_rejected": entry["score_rejected"],
            }

            data.append(new_entry)

    # 写入输出文件
    output_path = input_path.replace(".jsonl", f"_{output_prefix}.json")
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

    print(f"Successfully converted {len(data)} entries to LLaMA Factory format and saved to {output_path}")


if __name__ == "__main__":
    convert_jsonl_to_llamafactory()
