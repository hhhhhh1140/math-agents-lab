import re
from fractions import Fraction

from math_verify import LatexExtractionConfig, parse, verify


def boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return text.strip()
    start += 7
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return text.strip()


def numeric(text):
    value = boxed(text)
    # Unambiguous decoration only: currency/percent signs, LaTeX spacing and a
    # trailing \text{unit}. Free-form sentences are still rejected below.
    value = re.sub(r"\\(?:text|mbox|mathrm)\{[^{}]*\}\s*$", "", value)
    value = re.sub(r"\\[$%!,;]|[$%]|\{,\}|\\ ", lambda m: "," if m.group() == "{,}" else "", value).strip()
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", value):
        raise ValueError("Not an unambiguous numeric answer")
    return Fraction(value)


def grade(prediction, gold, dataset):
    if not prediction.strip():
        return {"correct": False, "grading_status": "empty"}
    if dataset in ("gsm8k", "demo"):
        try:
            return {"correct": numeric(prediction) == numeric(gold), "grading_status": "numeric"}
        except (ValueError, ZeroDivisionError):
            return {"correct": False, "grading_status": "unparsed"}
    try:
        config = (LatexExtractionConfig(),)
        target = parse("$" + boxed(gold).strip("$") + "$", extraction_config=config, parsing_timeout=3)
        candidate = parse("$" + boxed(prediction).strip("$") + "$", extraction_config=config, parsing_timeout=3)
        if not target or not candidate:
            return {"correct": False, "grading_status": "unparsed"}
        return {"correct": bool(verify(target, candidate, timeout_seconds=3)), "grading_status": "math_verify"}
    except Exception as exc:
        return {"correct": False, "grading_status": "error", "grading_error": type(exc).__name__}
