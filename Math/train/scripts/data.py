import sys
from pathlib import Path

from datasets import Dataset, load_dataset

sys.path.append(str(Path(__file__).resolve().parents[3]))
from Math.train.scripts.utils import rank0_print

NUM_PROC = 64


def prepare_data(
    data_path: str,
    sanity_check: bool,
    eot_token: str = "",
) -> Dataset:
    """Prepare the dataset for Uni-DPO training."""
    data: Dataset = load_dataset("json", data_files=data_path, split="train")

    if sanity_check:
        selected_range = range(min(len(data), 100))
        rank0_print(f"Sanity check is enabled: using the first {len(selected_range)} samples.")
        data = data.select(selected_range)
    else:
        rank0_print(f"Using the full dataset with {len(data)} samples.")

    if eot_token:
        data = data.map(
            lambda x: {
                "chosen": x["chosen"] if x["chosen"].endswith(eot_token) else x["chosen"] + eot_token,
                "rejected": x["rejected"] if x["rejected"].endswith(eot_token) else x["rejected"] + eot_token,
            },
            num_proc=NUM_PROC,
            desc="Appending EOT token to chosen and rejected responses",
        )

    return data
