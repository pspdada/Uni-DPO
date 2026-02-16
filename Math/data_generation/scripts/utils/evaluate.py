import re
import sys
import warnings
from concurrent.futures import TimeoutError
from pathlib import Path

from pebble import ProcessPool
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[4]))
from Math.data_generation.scripts.utils.parser import run_execute
from Math.data_generation.scripts.utils.python_executor import PythonExecutor
from Math.evaluation.scripts.eval_utils.grader import math_equal

# Set maximum number of workers for parallel processing
MAX_WORKERS = 32


def _math_answer_cleaning(answer, dataset_name="math"):
    """
    remove irrelevant strings and unify the answer format before checking whether the answers are equal
    """

    def _is_completely_wrapped_by_text(input_string):
        pattern = r"^\\text{(.*)}$"
        match = re.match(pattern, input_string)
        if match:
            ## input_string is completely wrapped by \text{}
            extracted_content = match.group(1)
            extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
            return extracted_content
        else:
            return None

    ## remove irrelevant \\text and space
    extracted_content = _is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    ## e.g., convert 5,\!460 into 5460; convert 14{,}916 into 14916 convert \$4 into 4
    answer = answer.replace(",\!", "").replace("{,}", "").replace("\$", "")
    ## e.g., convert \dfrac{3}{2} into frac{3}{2}
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    ## e.g., convert 121^\circ into 121
    answer = answer.replace("^\circ", "")
    answer = answer.replace("^{\circ}", "")
    ## remove \quad
    answer = answer.replace("\quad", "")
    ## remove space
    answer = answer.replace(" ", "")
    ## remove \n
    answer = answer.replace("\n", "").replace("\\n", "")
    ## e.g., convert 5th into 5
    answer = answer.replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
    ## e.g., convert 3.54\times10^{10} into 3.54e10
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}", r"\1e\2", answer)
    ## e.g., convert 3.54\times10^10 into 3.54e10
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)", r"\1e\2", answer)
    ## e.g., convert 558\,\text{nm} into 558
    answer = re.sub(r"\\,\\text\{.*?\}", "", answer)
    ## e.g., convert 558\text{nm} into 558
    answer = re.sub(r"\\text\{.*?\}", "", answer)
    ## e.g., convert 2^{10} into 2^10
    answer = re.sub(r"(\d+)\^{(\d+)}", r"\1^\2", answer)
    ## lowercase
    answer = answer.lower()

    if dataset_name == "collegemath":
        ## convert 558\mathrm{ft} into 558
        answer = re.sub(r"\\mathrm\{.*?\}", "", answer)
        ## clean noisy answer
        answer = re.sub(r"\$\([^)]*\)", "", answer)
        if answer.endswith("-"):
            answer = answer[:-1]
        if answer.endswith("."):
            answer = answer[:-1]
        if answer.endswith("hours"):
            answer = answer[: -len("hours")]
        ## extract final answer after '=' or ':'
        if "=" in answer:
            answer = answer.split("=", 1)[1]
        if ":" in answer:
            answer = answer.split(":", 1)[1]
        ## \emptyset and \oslash both reprsent empty set in latex
        answer = answer.replace("\\emptyset", "\\oslash")
    if dataset_name == "gsm8k":
        # Example: 5,600 -> 5600
        answer = answer.replace(",", "")
    if dataset_name == "gaokao2023en":
        unit_strings = [
            "students",
            "dollars",
            "boxes",
            "feet",
            "kilometers",
            "meters",
            "degreesontheBreadusscale",
            "$",
            "a.m.",
            "am",
            "minutes",
        ]
        for unit in unit_strings:
            answer = answer.replace(unit, "")

    return answer


def _math_equal_process(param: tuple):
    return math_equal(_math_answer_cleaning(param[-2]), _math_answer_cleaning(param[-1]))


def get_batch_scores(response_lists: list[list[str]], ground_truths: list[str]) -> list[dict]:
    """
    Get batch scores and predictions for each sample.
    Args:
        response_lists (list[list[str]]): A list of lists, where each inner list contains code responses.
        ground_truths (list[str]): A list of ground truth answers.
    Returns:
        A list of dictionaries, where each dictionary contains:
        - "correct": A list of booleans indicating whether each response is correct.
        - "pred": A list of predictions extracted from each response.
    """
    samples: list[dict] = []
    assert len(response_lists) == len(ground_truths)
    for i in range(len(response_lists)):
        samples.append({"gt": ground_truths[i], "code": response_lists[i]})
    if "idx" in samples[0]:
        samples = {sample["idx"]: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x["idx"])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
    prompt_type = "cot"

    # Execute responses to get predictions
    if "pred" not in samples[0]:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        for sample in tqdm(samples, desc="Execute"):
            sample["pred"] = []
            sample["report"] = []
            gt = f"\\boxed{{{sample['gt']}}}"
            sample["_gt"] = run_execute(executor, gt, prompt_type, execute=False)[0]

            for code in sample["code"]:
                pred, report = run_execute(executor, code, prompt_type, execute=True)
                sample["pred"].append(pred)
                sample["report"].append(report)

    params: list[tuple] = [(idx, pred, sample["_gt"]) for idx, sample in enumerate(samples) for pred in sample["pred"]]

    scores: list[bool] = []
    timeout_cnt = 0

    with ProcessPool(max_workers=MAX_WORKERS) as pool, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        future = pool.map(_math_equal_process, params, timeout=10)
        iterator = future.result()
        with tqdm(total=len(params), desc="Evaluate") as progress_bar:
            for i, param in enumerate(params):
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError:
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    exit()
                finally:
                    progress_bar.update(1)
        print(f"Total timeouts: {timeout_cnt}")

    _idx = 0
    results: list[dict] = []
    for sample in samples:
        correct = scores[_idx : _idx + len(sample["pred"])]
        assert len(correct) == len(sample["pred"])
        results.append({"correct": correct, "pred": sample["pred"]})
        _idx += len(sample["pred"])
    del _idx

    return results


if __name__ == "__main__":
    pass
