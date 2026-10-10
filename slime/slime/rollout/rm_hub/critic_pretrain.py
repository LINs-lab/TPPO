import re


VALUE_TAG_PATTERN = re.compile(r"<value>\s*([0-9]{1,3})\s*</value>", flags=re.IGNORECASE)
ESCAPED_VALUE_TAG_PATTERN = re.compile(
    r"&lt;\s*value\s*&gt;\s*([0-9]{1,3})\s*&lt;\s*/\s*value\s*&gt;",
    flags=re.IGNORECASE,
)
LOOSE_VALUE_PATTERN = re.compile(r"\bvalue\b\s*[:=]\s*([0-9]{1,3})\b", flags=re.IGNORECASE)

NO_PARSE_REWARD = 0.0


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _parse_binary_label(label) -> int | None:
    raw = label
    if isinstance(label, list):
        if not label:
            return None
        raw = label[0]

    text = str(raw).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    return None


def _parse_soft_label_to_prob(label) -> float | None:
    raw = label
    if isinstance(label, list):
        if not label:
            return None
        raw = label[0]

    text = str(raw).strip().lower()

    # 1) binary
    b = _parse_binary_label(text)
    if b is not None:
        return float(b)

    # 2) 0..100 score
    try:
        v = float(text)
    except ValueError:
        return None

    if 0.0 <= v <= 1.0:
        return float(v)
    if 0.0 <= v <= 100.0:
        return float(v) / 100.0
    return None


def extract_critic_pretrain_value(response: str) -> int | None:
    """
    Extract `<value>INTEGER_0_TO_100</value>` from model output.
    Returns None if format is invalid or value is out of range.
    """
    text = _strip_chain_of_thought(response or "").strip()
    matches = VALUE_TAG_PATTERN.findall(text)
    if not matches:
        matches = ESCAPED_VALUE_TAG_PATTERN.findall(text)
    if not matches:
        matches = LOOSE_VALUE_PATTERN.findall(text)
    if not matches:
        return None

    for candidate in reversed(matches):
        try:
            value = int(candidate)
        except ValueError:
            continue
        if 0 <= value <= 100:
            return value
    return None


def compute_critic_pretrain_reward(response: str, label, error_power: float = 1.0) -> float:
    pred = extract_critic_pretrain_value(response)
    target_p = _parse_soft_label_to_prob(label)
    if pred is None or target_p is None:
        return NO_PARSE_REWARD

    try:
        power = float(error_power)
    except (TypeError, ValueError):
        power = 1.0
    if power <= 0.0:
        power = 1.0

    prob = pred / 100.0
    diff = abs(prob - float(target_p))
    reward = 1.0 - (diff**power)
    if reward < 0.0:
        return 0.0
    if reward > 1.0:
        return 1.0
    return reward
