import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from Math.data_generation.scripts.utils.python_executor import PythonExecutor


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def _strip_string(s: str) -> str:
    s = str(s).strip()
    # linebreaks
    s = s.replace("\n", "")

    # right "."
    s = s.rstrip(".")

    # remove inverse spaces
    s = s.replace("\\!", "")
    s = s.replace("\\ ", "")

    # replace \\ with \
    s = s.replace("\\\\", "\\")
    s = s.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")

    # remove \left and \right
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", s).strip()
    if _string != "" and _string != s:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        s = _string

    # Remove circ (degrees)
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")

    # remove dollar signs
    s = s.replace("\\$", "")
    s = s.replace("$", "")

    s = s.replace("\\text", "")
    s = s.replace("x\\in", "")

    # remove percentage
    s = s.replace("\\%", "")
    s = s.replace("\%", "")
    s = s.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    s = s.replace(" .", " 0.")
    s = s.replace("{.", "{0.")

    # cdot
    s = s.replace("\\cdot", "")

    # inf
    s = s.replace("infinity", "\\infty")
    if "\\infty" not in s:
        s = s.replace("inf", "\\infty")
    s = s.replace("+\\inity", "\\infty")

    # and
    s = s.replace("and", "")
    s = s.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    s = re.sub(r"\\mbox{.*?}", "", s)

    # quote
    s.replace("'", "")
    s.replace('"', "")

    # i, j
    if "j" in s and "i" not in s:
        s = s.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    s = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", s)
    s = re.sub(r"(\d+)\.0+$", r"\1", s)

    # if empty, return empty string
    if len(s) == 0:
        return s
    if s[0] == ".":
        s = "0" + s

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(s.split("=")) == 2:
        if len(s.split("=")[0]) <= 2:
            s = s.split("=")[1]

    s = _fix_sqrt(s)
    s = s.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    s = _fix_fracs(s)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    s = _fix_a_slash_b(s)

    return s


def _extract_answer(pred_str: str) -> str:
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif _extract_program_output(pred_str) != "":
        # fall back to program
        pred = _extract_program_output(pred_str)
    else:  # use the last number
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = _strip_string(pred)
    return pred


def _extract_program(result: str, last_only=False):
    """
    extract the program after "```python", and before "```"
    """
    all_program = []
    start = False
    program = ""
    for line in result.split("\n"):
        # if line.startswith("```python"):
        if "```python" in line:
            program = ""
            # if last_only:
            #    program = "" # only extract the last program
            # else:
            # program += "\n# ========\n"

            start = True
        elif line.startswith("```"):
            start = False
            all_program.append(program)
            program = ""
        elif start:
            program += line + "\n"
    return all_program


def _extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if "```output" in pred_str:
        pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    output = pred_str.strip()
    return output


def run_execute(executor: PythonExecutor, result: str, prompt_type, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = _extract_program_output(result)
    elif prompt_type in ["pot", "pal"] and execute:
        code = _extract_program(result)
        prediction, report = executor.apply(code)
    else:
        prediction = _extract_answer(result)

    prediction = _strip_string(prediction)
    return prediction, report
