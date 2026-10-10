"""Countdown correctness reward, with no partial credit for formatting.

Scoring follows kanishkg/cognitive-behaviors:
https://github.com/kanishkg/cognitive-behaviors/blob/main/verl/utils/reward_score/countdown.py
"""

import re


def extract_solution(response):
    """Accept local boxed answers and reference answer tags in a Slime response.

    Slime supplies the response alone, so no assistant prefix is required and
    extraction is not restricted to the last line. The last opening delimiter
    selects the final answer; an incomplete final answer receives zero.
    """
    if not isinstance(response, str):
        return None
    boxed_start = response.rfind("\\boxed{")
    answer_start = response.rfind("<answer>")
    if boxed_start < 0 and answer_start < 0:
        return None
    if boxed_start > answer_start:
        start, closing = boxed_start + 7, "}"
    else:
        start, closing = answer_start + 8, "</answer>"
    end = response.find(closing, start)
    return response[start:end].strip() if end >= 0 else None


def validate_equation(equation_str, available_numbers):
    """Match the reference's integer-token multiset check, including duplicates."""
    try:
        return sorted(int(n) for n in re.findall(r"\d+", equation_str)) == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str):
    """Use the reference's character whitelist and Python arithmetic semantics."""
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_countdown_reward(response, label):
    """Require the target and exactly the supplied multiset of integer operands."""
    if not isinstance(label, dict):
        raise ValueError("Countdown label must contain 'target' and 'numbers'.")
    numbers = label.get("numbers")
    target = label.get("target")
    if (not isinstance(numbers, list) or not numbers
            or any(type(n) is not int or n < 0 for n in numbers)
            or type(target) is not int):
        raise ValueError("Countdown requires nonnegative integer numbers and an integer target.")

    expression = extract_solution(response)

    reward = {
        "score": 0.0,
        "point": 0.0,
        "acc": False,
        "extracted_pred": expression or "",
        "extracted_gt": label,
        "scored_by": "countdown",
    }
    if not expression or not validate_equation(expression, numbers):
        return reward

    try:
        result = evaluate_equation(expression)
        correct = result is not None and abs(result - target) < 1e-5
    except Exception:
        return reward
    reward.update(score=float(correct), point=float(correct), acc=correct)
    return reward
