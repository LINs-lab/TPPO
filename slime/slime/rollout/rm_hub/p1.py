# # Copyright 2025 Garena Online Private Limited
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Provides a math answer grading function with high recall.
# Based on HF math_verify, verl, open reasoner zero, etc.
# """

# import os
# import re
# import signal
# import math
# import time
# import traceback
# from openai import OpenAI
# from functools import wraps, partial
# from itertools import islice, zip_longest
# from typing import Optional, Union
# from pylatexenc import latex2text
# from decimal import Decimal, localcontext
# import sympy
# from sympy import N, Pow, Mul
# from sympy.parsing import sympy_parser
# from math_verify import (ExprExtractionConfig, LatexExtractionConfig, parse, verify)


# def timeout(timeout_seconds: int = 10):
#     if os.name == "posix":
#         import signal
#         def decorator(func):
#             def handler(signum, frame):
#                 raise TimeoutError("verify timed out!")
#             def wrapper(*args, **kwargs):
#                 old_handler = signal.getsignal(signal.SIGALRM)
#                 signal.signal(signal.SIGALRM, handler)
#                 signal.alarm(timeout_seconds)
#                 try:
#                     return func(*args, **kwargs)
#                 finally:
#                     signal.alarm(0)
#                     signal.signal(signal.SIGALRM, old_handler)
#             return wrapper
#         return decorator


# # units mainly from MathQA
# unit_texts = [
#     "east",
#     "degree",
#     "mph",
#     "kmph",
#     "ft",
#     "m sqaure",
#     " m east",
#     "sq m",
#     "deg",
#     "mile",
#     "q .",
#     "monkey",
#     "prime",
#     "ratio",
#     "profit of rs",
#     "rd",
#     "o",
#     "p . m",
#     "lb",
#     "tile",
#     "per",
#     "lt",
#     "gain",
#     "ab",
#     "way",
#     "west",
#     "no change",
#     "men",
#     "soldier",
#     "pie",
#     "bc",
#     "excess",
#     "st",
#     "inches",
#     "noon",
#     "percent",
#     "by",
#     "gal",
#     "kmh",
#     "acre",
#     "rise",
#     "a . m",
#     "th",
#     "π r 2",
#     "sq",
#     "mark",
#     "toy",
#     "coin",
#     "sq . m",
#     "gallon",
#     "° f",
#     "profit",
#     "minw",
#     "yr",
#     "women",
#     "feet",
#     "am",
#     "pm",
#     "hr",
#     "cu cm",
#     "square",
#     "v â € ™",
#     "are",
#     "rupee",
#     "rounds",
#     "cubic",
#     "cc",
#     "mtr",
#     "ohm",
#     "number",
#     "kmph",
#     "day",
#     "hour",
#     "minute",
#     "min",
#     "second",
#     "man",
#     "woman",
#     "sec",
#     "cube",
#     "mt",
#     "sq inch",
#     "mp",
#     "∏ cm ³",
#     "hectare",
#     "more",
#     "sec",
#     "unit",
#     "cu . m",
#     "cm 2",
#     "rs .",
#     "rs",
#     "kg",
#     "month",
#     "cm",
#     "mm",
#     "apple",
#     "liter",
#     "loss",
#     "yard",
#     "pure",
#     "year",
#     "increase",
#     "decrease",
#     "less",
#     "Surface",
#     "litre",
#     "pi sq m",
#     "s .",
#     "metre",
#     "meter",
#     "inch",
#     "kilogram",
#     "second",
#     "ampere",
#     "A",
#     "K",
#     "mol",
#     "cd",
#     "N",
#     "J",
#     "W",
#     "Pa",
#     "Hz",
#     "C",
#     "V",
#     "Ω",
#     "F",
#     "T",
#     "H",
#     "eV",
#     "kW·h",
#     "atm",
#     "bar",
#     "°C"
# ]
# unit_texts.extend([t + "s" for t in unit_texts])

# def _strip_string(string):
#     def _fix_fracs(string):
#         substrs = string.split("\\frac")
#         new_str = substrs[0]
#         if len(substrs) > 1:
#             substrs = substrs[1:]
#             for substr in substrs:
#                 new_str += "\\frac"
#                 if substr[0] == "{":
#                     new_str += substr
#                 else:
#                     try:
#                         assert len(substr) >= 2
#                     except:
#                         return string
#                     a = substr[0]
#                     b = substr[1]
#                     if b != "{":
#                         if len(substr) > 2:
#                             post_substr = substr[2:]
#                             new_str += "{" + a + "}{" + b + "}" + post_substr
#                         else:
#                             new_str += "{" + a + "}{" + b + "}"
#                     else:
#                         if len(substr) > 2:
#                             post_substr = substr[2:]
#                             new_str += "{" + a + "}" + b + post_substr
#                         else:
#                             new_str += "{" + a + "}" + b
#         string = new_str
#         return string

#     def _fix_a_slash_b(string):
#         if len(string.split("/")) != 2:
#             return string
#         a = string.split("/")[0]
#         b = string.split("/")[1]
#         try:
#             a = int(a)
#             b = int(b)
#             assert string == "{}/{}".format(a, b)
#             new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
#             return new_string
#         except:
#             return string

#     def _remove_right_units(string):
#         # "\\text{ " only ever occurs (at least in the val set) when describing units
#         if "\\text{ " in string:
#             splits = string.split("\\text{ ")
#             assert len(splits) == 2
#             return splits[0]
#         else:
#             return string

#     def _fix_sqrt(string):
#         if "\\sqrt" not in string:
#             return string
#         splits = string.split("\\sqrt")
#         new_string = splits[0]
#         for split in splits[1:]:
#             if split[0] != "{":
#                 a = split[0]
#                 new_substr = "\\sqrt{" + a + "}" + split[1:]
#             else:
#                 new_substr = "\\sqrt" + split
#             new_string += new_substr
#         return new_string

#     # linebreaks
#     string = string.replace("\n", "")
#     # print(string)

#     # remove inverse spaces
#     string = string.replace("\\!", "")
#     # print(string)

#     # replace \\ with \
#     string = string.replace("\\\\", "\\")
#     # print(string)

#     # matrix
#     string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
#     string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
#     string = string.replace("bmatrix", "pmatrix")

#     # replace tfrac and dfrac with frac
#     string = string.replace("tfrac", "frac")
#     string = string.replace("dfrac", "frac")
#     string = (
#         string.replace("\\neq", "\\ne")
#         .replace("\\leq", "\\le")
#         .replace("\\geq", "\\ge")
#     )
#     # print(string)

#     # remove \left and \right
#     string = string.replace("\\left", "")
#     string = string.replace("\\right", "")
#     # print(string)

#     # Remove unit: miles, dollars if after is not none
#     _string = re.sub(r"\\text{.*?}$", "", string).strip()
#     if _string != "" and _string != string:
#         # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
#         string = _string

#     # Remove unit: texts
#     for _ in range(2):
#         for unit_text in unit_texts:
#             # use regex, the prefix should be either the start of the string or a non-alphanumeric character
#             # the suffix should be either the end of the string or a non-alphanumeric character
#             _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
#             if _string != "":
#                 string = _string

#     # Remove circ (degrees)
#     string = string.replace("^{\\circ}", "")
#     string = string.replace("^\\circ", "")

#     # remove dollar signs
#     string = string.replace("\\$", "")

#     # remove units (on the right)
#     string = _remove_right_units(string)

#     # remove percentage
#     string = string.replace("\\%", "")
#     string = string.replace("\%", "")

#     # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
#     string = string.replace(" .", " 0.")
#     string = string.replace("{.", "{0.")
#     # if empty, return empty string
#     if len(string) == 0:
#         return string
#     if string[0] == ".":
#         string = "0" + string

#     # to consider: get rid of e.g. "k = " or "q = " at beginning
#     if len(string.split("=")) == 2:
#         if len(string.split("=")[0]) <= 2:
#             string = string.split("=")[1]

#     # fix sqrt3 --> sqrt{3}
#     string = _fix_sqrt(string)

#     # remove spaces
#     string = string.replace(" ", "")

#     # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
#     string = _fix_fracs(string)

#     # manually change 0.5 --> \frac{1}{2}
#     if string == "0.5":
#         string = "\\frac{1}{2}"

#     # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
#     string = _fix_a_slash_b(string)

#     return string


# def _strip_properly_formatted_commas(expr: str):
#     # We want to be careful because we don't want to strip tuple commas
#     p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
#     while True:
#         next_expr = p1.sub("\\1\\3\\4", expr)
#         if next_expr == expr:
#             break
#         expr = next_expr
#     return next_expr

# def _is_float(num: str) -> bool:
#     try:
#         float(num)
#         return True
#     except ValueError:
#         return False

# def _is_int(x: float) -> bool:
#     try:
#         return abs(x - int(round(x))) <= 1e-7
#     except:
#         return False

# def _is_frac(expr: str) -> bool:
#     return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

# def _str_is_int(x: str) -> bool:
#     try:
#         x = _strip_properly_formatted_commas(x)
#         x = float(x)
#         return abs(x - int(round(x))) <= 1e-7
#     except:
#         return False

# def _str_to_int(x: str) -> bool:
#     x = x.replace(",", "")
#     x = float(x)
#     return int(x)

# def _inject_implicit_mixed_number(step: str):
#     """
#     Automatically make a mixed number evalable
#     e.g. 7 3/4 => 7+3/4
#     """
#     p1 = re.compile("([0-9]) +([0-9])")
#     step = p1.sub("\\1+\\2", step)  ## implicit mults
#     return step

# def _parse_latex(expr: str) -> str:
#     """Attempts to parse latex to an expression sympy can read."""
#     expr = expr.replace("\\tfrac", "\\frac")
#     expr = expr.replace("\\dfrac", "\\frac")
#     expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
#     expr = latex2text.LatexNodes2Text().latex_to_text(expr)

#     # Replace the specific characters that this parser uses.
#     expr = expr.replace("√", "sqrt")
#     expr = expr.replace("π", "pi")
#     expr = expr.replace("∞", "inf")
#     expr = expr.replace("∪", "U")
#     expr = expr.replace("·", "*")
#     expr = expr.replace("×", "*")

#     return expr.strip()


# # Dan Hendrycks' code
# def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
#     if answer is None:
#         return None
#     answer = answer.strip()
#     try:
#         # Remove enclosing `\text{}`.
#         m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
#         if m is not None:
#             answer = m.group("text").strip()
#         return _strip_string(answer)
#     except:
#         return answer

# def sympy_normalize_answer(expr: str) -> str:
#     """Normalize answer expressions."""
#     if expr is None:
#         return None

#     # Remove enclosing `\text{}`.
#     m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
#     if m is not None:
#         expr = m.group("text")

#     expr = expr.replace("\\%", "%")
#     expr = expr.replace("\\$", "$")
#     expr = expr.replace("$", "")
#     expr = expr.replace("%", "")
#     expr = expr.replace(" or ", " , ")
#     expr = expr.replace(" and ", " , ")

#     expr = expr.replace("million", "*10^6")
#     expr = expr.replace("billion", "*10^9")
#     expr = expr.replace("trillion", "*10^12")

#     for _ in range(2):
#         for unit_text in unit_texts:
#             # use regex, the prefix should be either the start of the string or a non-alphanumeric character
#             # the suffix should be either the end of the string or a non-alphanumeric character
#             _expr = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", expr)
#             if _expr != "":
#                 expr = _expr

#     expr = re.sub(f"\^ *\\\\circ", "", expr)

#     if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
#         expr = expr[1:-1]

#     expr = re.sub(",\\\\! *", "", expr)
#     if _is_float(expr) and _is_int(float(expr)):
#         expr = str(int(round(float(expr))))
#     if "\\" in expr:
#         try:
#             expr = _parse_latex(expr)
#         except:
#             pass

#     # edge case with mixed numbers and negative signs
#     expr = re.sub("- *", "-", expr)

#     expr = _inject_implicit_mixed_number(expr)
#     expr = expr.replace(" ", "")

#     # if we somehow still have latex braces here, just drop them
#     expr = expr.replace("{", "")
#     expr = expr.replace("}", "")

#     # don't be case sensitive for text answers
#     expr = expr.lower()

#     if _str_is_int(expr):
#         expr = str(_str_to_int(expr))

#     return expr


# def judge_MC(pred, gold):
#     common_answer = [chr(i) for i in range(65, 91)] # 'A'~'Z'
#     if pred == gold:
#         return True
#     else:
#         if pred.startswith("[") and pred.endswith("]"):
#             pred = pred.strip("[]")
#         if not pred:
#             return False
#         if pred[0] in common_answer and (len(pred) > 1 and (pred[1] == "." or pred[1] == ":")):
#             return pred[0] == gold
#         if f"'{gold}'" in pred:
#             return True
#         else:
#             return False

# def judge_TF(pred, gold):
#     def contains_chinese(d):
#         def is_chinese_char(ch):
#             return '\u4e00' <= ch <= '\u9fff'

#         def check(value):
#             if isinstance(value, str):
#                 return any(is_chinese_char(ch) for ch in value)
#             elif isinstance(value, dict):
#                 return any(check(v) for v in value.values())
#             elif isinstance(value, list):
#                 return any(check(item) for item in value)
#             return False

#         return check(d)

#     if contains_chinese(pred):
#             pred = "TRUE"
#             pred = "FALSE"
#     else:
#         pred = pred.upper()
#     answers = ["TRUE", "FALSE", "T", "F", "YES", "NO", "Y", "N"]
#     gold = gold.upper()
#     if gold not in answers or pred not in answers:
#         return False
#     if gold in ["TRUE", "YES", "T", "Y"]:
#         gold = "TRUE"
#     if gold in ["FALSE", "NO", "F", "N"]:
#         gold = "FALSE"
#     if pred in ["TRUE", "YES", "T", "Y"]:
#         pred = "TRUE"
#     if pred in ["FALSE", "NO", "F", "N"]:
#         pred = "FALSE"
#     return pred == gold


# def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
#     ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
#     given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
#     # be at least as lenient as mathd
#     if ground_truth_normalized_mathd == given_answer_normalized_mathd:
#         return True, given_answer_normalized_mathd, ground_truth_normalized_mathd
#     return False, given_answer_normalized_mathd, ground_truth_normalized_mathd


# # sympy might hang -- we don't care about trying to be lenient in these cases
# BAD_SUBSTRINGS = ["^{", "^("]
# BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
# TUPLE_CHARS = "()[]"
# def split_tuple(expr: str):
#     """
#     Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
#     """
#     expr = _strip_properly_formatted_commas(expr)
#     if len(expr) == 0:
#         return []
#     if (
#         len(expr) > 2
#         and expr[0] in TUPLE_CHARS
#         and expr[-1] in TUPLE_CHARS
#         and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
#     ):
#         elems = [elem.strip() for elem in expr[1:-1].split(",")]
#     else:
#         elems = [expr]
#     return elems


# def _sympy_parse(expr: str):
#     """Parses an expression with sympy."""
#     py_expr = expr.replace("^", "**")
#     return sympy_parser.parse_expr(
#         py_expr,
#         transformations=(
#             sympy_parser.standard_transformations
#             + (sympy_parser.implicit_multiplication_application,)
#         ),
#         evaluate=False,
#     )

# def should_allow_eval(expr: str):
#     def count_unknown_letters_in_expr(expr: str):
#         expr = expr.replace("sqrt", "")
#         expr = expr.replace("frac", "")
#         letters_in_expr = set([x for x in expr if x.isalpha()])
#         return len(letters_in_expr)
#     # we don't want to try parsing unknown text or functions of more than two variables
#     if count_unknown_letters_in_expr(expr) > 2:
#         return False

#     for bad_string in BAD_SUBSTRINGS:
#         if bad_string in expr:
#             return False

#     for bad_regex in BAD_REGEXES:
#         if re.search(bad_regex, expr) is not None:
#             return False

#     return True

# def handle_pi(string, pi):
#     if isinstance(string, str) and "pi" in string:
#         # Find the first occurrence of "\pi"
#         idx = string.find("pi")

#         # Iterate over the string and find all occurrences of "\pi" with a valid previous character
#         while idx != -1:

#             if idx > 0 and string[idx - 1].isdigit():
#                 # Replace "\pi" with "*math.pi" if the previous character is a digit
#                 string = string[:idx] + f"*{pi}" + string[idx + 2:]
#             else:
#                 # Replace "\pi" with "1*math.pi" if the previous character is not a digit
#                 string = string[:idx] + f"1*{pi}" + string[idx + 2:]

#             # Find the next occurrence of "\pi"
#             idx = string.find("pi", idx + 1)

#         # Evaluate the expression using eval() function
#         try:
#             string = eval(string)
#         except:
#             pass

#     return string

# @timeout(timeout_seconds=30)
# def are_equal_under_sympy(gold: str, pred: str, precision: float = 2e-3):
#     def is_scientific_notation(expr):
#         return (
#             isinstance(expr, Mul)
#             and isinstance(expr.args[1], Pow)
#             and expr.args[1].args[0] == 10
#         )

#     def to_scientific_notation_sympy(num):
#         num_sci = f"{num:.2e}"  # e.g., "1.23e-5"
#         base, exponent = num_sci.split("e")
#         return f"{base}*10**{int(exponent)}"

#     def count_decimal_places(x, tol=1e-6):
#         """
#         """
#         with localcontext() as ctx:
#             d = Decimal(str(x)).normalize()
#             if "." not in s:
#                 return 0
#             integer_part, decimal_part = s.split(".")
#             clean_decimal = ""
#             for i, ch in enumerate(decimal_part):
#                 clean_decimal += ch
#                 if abs(x - round(x, i+1)) <= tol:
#                     break

#             return len(clean_decimal)

#     try:
#         if pred == gold:
#             return True

#         pred_value = float(pred)
#         gold_value = float(gold)
#         min_decimal_places = min(count_decimal_places(gold_value), count_decimal_places(pred_value))

#         pred_value = round(pred_value, min_decimal_places)
#         gold_value = round(gold_value, min_decimal_places)

#         # if abs((pred_value - gold_value) / gold_value) <= precision * 1.01:
#         #     return True

#         spred = _sympy_parse(to_scientific_notation_sympy(float(pred)))
#         sgold = _sympy_parse(to_scientific_notation_sympy(float(gold)))
#         if is_scientific_notation(spred) and is_scientific_notation(sgold):
#             base_pred, exponent_pred = N(spred.args[0]), N(spred.args[1].args[1])
#             base_gold, exponent_gold = N(sgold.args[0]), N(sgold.args[1].args[1])
#             min_decimal_places = min(count_decimal_places(base_gold), count_decimal_places(base_pred))
#             base_pred = round(base_pred, min_decimal_places)
#             base_gold = round(base_gold, min_decimal_places)
#             if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
#                 return True
#     except Exception:
#         pass

#     try:
#         if should_allow_eval(gold) and should_allow_eval(pred):
#             exp_gold = _sympy_parse(gold)
#             exp_pred = _sympy_parse(pred)

#             expr = (exp_gold - exp_pred) / (exp_gold if exp_gold != 0 else 1)
#             simplified = sympy.simplify(expr)
#             if abs(N(simplified)) <= precision * 1.01:
#                 return True
#             if is_scientific_notation(exp_pred) != is_scientific_notation(exp_gold):
#                 if is_scientific_notation(exp_pred):
#                     gold = to_scientific_notation_sympy(float(gold))
#                     exp_gold = _sympy_parse(gold)
#                 else:
#                     pred = to_scientific_notation_sympy(float(pred))
#                     exp_pred = _sympy_parse(pred)

#             if is_scientific_notation(exp_pred) and is_scientific_notation(exp_gold):
#                 base_pred, exponent_pred = N(exp_pred.args[0]), N(exp_pred.args[1].args[1])
#                 base_gold, exponent_gold = N(exp_gold.args[0]), N(exp_gold.args[1].args[1])
#                 min_decimal_places = min(count_decimal_places(base_gold), count_decimal_places(base_pred))
#                 base_pred = round(base_pred, min_decimal_places)
#                 base_gold = round(base_gold, min_decimal_places)

#                 if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
#                     return True
#             else:
#                 if N(exp_pred) == N(exp_gold):
#                     return True
#     except Exception:
#         pass

#     return False

# def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
#     ground_truth_normalized = sympy_normalize_answer(ground_truth)
#     given_normalized = sympy_normalize_answer(given_answer)
#     if ground_truth_normalized is None:
#         return False, given_normalized, ground_truth_normalized

#     if ground_truth_normalized == given_normalized:
#         return True, given_normalized, ground_truth_normalized

#     if len(given_normalized) == 0:
#         return False, given_normalized, ground_truth_normalized

#     ground_truth_elems = split_tuple(ground_truth_normalized)
#     given_elems = split_tuple(given_normalized)

#     if len(ground_truth_elems) > 1 and (
#         ground_truth_normalized[0] != given_normalized[0]
#         or ground_truth_normalized[-1] != given_normalized[-1]
#     ):
#         is_correct = False
#     elif len(ground_truth_elems) != len(given_elems):
#         is_correct = False
#     else:
#         for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
#             if _is_frac(ground_truth_elem) and _is_frac(given_elem):
#                 # if fractions aren't reduced, then shouldn't be marked as correct
#                 # so, we don't want to allow sympy.simplify in this case
#                 is_correct = ground_truth_elem == given_elem
#             # elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
#             #     # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
#             #     is_correct = False
#             else:
#                 is_correct = judge_MC(given_elem, ground_truth_elem) or judge_TF(given_elem, ground_truth_elem)
#                 if not is_correct:
#                     if "pi" in given_elem or "pi" in ground_truth_elem:
#                         equivs = []
#                         for pi in [math.pi, 3.14, 180]:
#                             given_elem_pi = handle_pi(given_elem, pi)
#                             ground_truth_elem_pi = handle_pi(ground_truth_elem, pi)
#                             try:
#                                 equivs.append(are_equal_under_sympy(ground_truth_elem_pi, given_elem_pi))
#                             except TimeoutError:
#                                 equivs.append(False)
#                         is_correct = any(equivs)
#                     else:
#                         try:
#                             is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
#                         except TimeoutError:
#                             is_correct = False
#             if not is_correct:
#                 break

#     return is_correct, given_normalized, ground_truth_normalized


# def repeatness(s: str):
#     def ranks(l):
#         index = {v: i for i, v in enumerate(sorted(set(l)))}
#         return [index[v] for v in l]

#     def suffixArray(s):
#         line = ranks(s)
#         n, k, ans, sa = len(s), 1, line, [0] * len(s)
#         while k < n - 1:
#             line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
#             ans, k = line, k << 1
#         for i, k in enumerate(ans):
#             sa[k] = i
#         return ans, sa

#     def lcp(arr, suffixArr, inv_suff):
#         n, ans, k = len(arr), [0] * len(arr), 0

#         for i in range(n):
#             if inv_suff[i] == n - 1:
#                 k = 0
#                 continue

#             j = suffixArr[inv_suff[i] + 1]
#             while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
#                 k += 1

#             ans[inv_suff[i]] = k
#             if k > 0:
#                 k -= 1

#         return ans

#     arr = [ord(i) for i in s]
#     n = len(arr)
#     if n <= 1:
#         return 0
#     c, sa = suffixArray(arr)
#     cnt = sum(lcp(arr, sa, c))

#     return (cnt * 2 / (n * (n + 1))) > 0.2

# class timeout:
#     def __init__(self, seconds=1, error_message="Timeout"):
#         self.seconds = seconds
#         self.error_message = error_message

#     def handle_timeout(self, signum, frame):
#         raise TimeoutError(self.error_message)

#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)

#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)

# def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
#     try:
#         with timeout(5):
#             try:
#                 if (len(given_answer) > 128 and repeatness(given_answer)) or (
#                     len(ground_truth) > 128 and repeatness(ground_truth)
#                 ):
#                     return False, given_answer, ground_truth

#                 # Next call math verify.
#                 given_answer.replace("\n", "")
#                 ground_truth.replace("\n", "")
#                 if not "$" in given_answer:
#                     given_answer = f"${given_answer}$"
#                 if not "$" in ground_truth:
#                     ground_truth = f"${ground_truth}$"
#                 given_answer = parse(
#                     given_answer,
#                     extraction_config=(
#                         LatexExtractionConfig(boxed_match_priority=0),
#                         ExprExtractionConfig(),
#                     ),
#                     fallback_mode="no_fallback",
#                     extraction_mode="first_match",
#                     parsing_timeout=1,
#                 )
#                 ground_truth = parse(
#                     ground_truth,
#                     extraction_config=(
#                         LatexExtractionConfig(boxed_match_priority=0),
#                         ExprExtractionConfig(),
#                     ),
#                     fallback_mode="no_fallback",
#                     extraction_mode="first_match",
#                     parsing_timeout=1,
#                 )
#                 result = verify(
#                     ground_truth,
#                     given_answer,
#                     numeric_precision=3,
#                     timeout_seconds=1,
#                 )
#                 return result, given_answer, ground_truth
#                 # or symbolic_equal(ground_truth, given_answer)
#             except Exception as e:
#                 print(f"Math verify error: {e}")
#                 return False, given_answer, ground_truth
#     except TimeoutError:
#         print("Math verify timeout")
#         return False, given_answer, ground_truth
#     except Exception as e:
#         print(f"Unexpected error in math_verify: {e}")
#         return False, given_answer, ground_truth


# def attach_wrapper(obj, func=None):
#     if func is None:
#         return partial(attach_wrapper, obj)
#     setattr(obj, func.__name__, func)
#     return func

# def retry(max_attempts:int=5, delay:Union[int, float]=300, print_trace_back=False, return_error_info=False):

#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             attempts = 0
#             while attempts < max_attempts:
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception:
#                     if print_trace_back:
#                         e = traceback.format_exc()
#                         print(error_info)
#                     time.sleep(delay)
#                     attempts += 1
#             if return_error_info:
#                 return error_info
#             else:
#                 return None

#         @attach_wrapper(wrapper)
#         def set_max_attempts(new_max_attempts):
#             nonlocal max_attempts
#             max_attempts = new_max_attempts

#         @attach_wrapper(wrapper)
#         def set_delay(new_delay):
#             nonlocal delay
#             delay = new_delay

#         wrapper.get_attempts = lambda: max_attempts
#         wrapper.get_delay = lambda: delay
#         return wrapper
#     return decorator

# class Model_args:
#     use_model: bool = False
#     model_name = 'gpt-oss-120b'  # Model name
#     api_key = 'None' # API key used to access the model via API, if not available, set to None
#     base_url = 'http://127.0.0.1:34882/v1'  # Anonymized model path or URL

#     def __init__(self, model_port=34882):
#         self.base_url = f'http://127.0.0.1:{model_port}/v1'

# def grade_answer_xverify(given_answer: str, ground_truth: str, problem: str, model_args: Model_args) -> tuple[bool, str]:
#     """

#     Returns:
#         tuple: (is_correct: bool, llm_response: str)
#     """
#     # print(f"[DEBUG] Entering xverify with problem={problem}, pred={given_answer}, gt={ground_truth}")
#     def call_api(prompt:str,
#                 system_prompt:Optional[str]=None,
#                 client=None,
#                 base_url:Optional[str]=None,
#                 model:str="gpt-3.5-turbo",
#                 api_key:Union[None,str]=None,
#                 max_tokens:int=None,
#                 logprobs:bool=False,
#                 top_logprobs:int=1,
#                 **kwargs) -> str:
#         if not client:
#             assert api_key is not None,'Please input your api key'
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#                 )
#         messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
#         if prompt:
#             messages.append({"role": "user", "content": prompt})

#         # Only include top_logprobs when logprobs is True
#         api_kwargs = {
#             "model": model,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "logprobs": logprobs,
#             **kwargs
#         }
#         if logprobs:
#             api_kwargs["top_logprobs"] = top_logprobs

#         response = client.chat.completions.create(**api_kwargs)
#         return response.choices[0].message.content

#     client = OpenAI(api_key=model_args.api_key, base_url=model_args.base_url)
#     prompt = """Determine if the Model Answer is mathematically equivalent to the Golden Answer. Only evaluate the final result, not reasoning steps.

# Equivalence rules:
# - Algebraic: n(n+1)/2 = n^2/2 + n/2
# - Numerical: 1/2 = 0.5; sqrt(2)/2 = 1/sqrt(2)
# - Sets: order doesn't matter unless specified
# - No partial credit

# Output format:
# <think>
# 1. Golden Answer: [state]
# 2. Model Answer: [extract]
# 3. Equivalence: [brief analysis]
# 4. Conclusion: Correct/Incorrect
# </think>
# \\boxed{{Correct}} or \\boxed{{Incorrect}}

# Problem: {Problem_Statement}
# Model Answer: {Model_Answer}
# Golden Answer: {Golden_Answer}"""
#     prompt = prompt.format(Problem_Statement=problem, Model_Answer=given_answer, Golden_Answer=ground_truth)
#     llm_response = call_api(prompt=prompt,
#                        client=client,
#                        max_tokens=model_args.max_tokens,
#                        model=model_args.model_name,
#                        temperature=model_args.temperature)
#     print(f"[DEBUG] Result from xverify: {llm_response}")
#     extracted_answer = extract_boxed_answer(llm_response)
#     is_correct = extracted_answer == "Correct"
#     return is_correct, llm_response


# def grade_marking_total_xverify(model_output: str, marking_items: list, problem: str, model_args: Model_args) -> tuple[float, str]:
#     """

#     Args:

#     Returns:
#         tuple: (total_score: float, llm_response: str)
#     """
#     def call_api(prompt:str,
#                 system_prompt:Optional[str]=None,
#                 client=None,
#                 base_url:Optional[str]=None,
#                 model:str="gpt-3.5-turbo",
#                 api_key:Union[None,str]=None,
#                 max_tokens:int=None,
#                 logprobs:bool=False,
#                 top_logprobs:int=1,
#                 **kwargs) -> str:
#         if not client:
#             assert api_key is not None,'Please input your api key'
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#                 )
#         messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
#         if prompt:
#             messages.append({"role": "user", "content": prompt})

#         # Only include top_logprobs when logprobs is True
#         api_kwargs = {
#             "model": model,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "logprobs": logprobs,
#             **kwargs
#         }
#         if logprobs:
#             api_kwargs["top_logprobs"] = top_logprobs

#         response = client.chat.completions.create(**api_kwargs)
#         return response.choices[0].message.content

#     client = OpenAI(api_key=model_args.api_key, base_url=model_args.base_url)

#     marking_list_text = ""
#     total_possible_points = 0.0
#     for idx, marking_item in enumerate(marking_items, 1):
#         desc = marking_item.get("desc", "")
#         pt = marking_item.get("pt", 0.0)
#         expressions_info = ""
#         if "expressions" in marking_item and marking_item["expressions"]:
#             expressions_info = f"\n  Expected expressions or patterns: {marking_item['expressions']}"
#         marking_list_text += f"\n{idx}. {desc}\n   Points: {pt}{expressions_info}\n"
#         total_possible_points += pt

#     prompt = """Evaluate the student's complete solution against ALL marking criteria and determine the total score.

# Guidelines:
# - Examine entire solution (reasoning, calculations, final answer)
# - Explicit statements or equivalent formulations count
# - Each criterion must be substantially met
# - Evaluate independently

# Marking Criteria:
# {Marking_Criteria_List}
# Total possible: {Total_Points}

# Output:
# <think>
# 1. Solution overview: [brief summary]
# 2. Criterion evaluation: [for each criterion, state satisfied/not]
# 3. Score calculation: [sum points]
# 4. Conclusion: [total score]
# </think>
# \\boxed{{<score>}}

# Problem: {Problem_Statement}
# Solution: {Model_Solution}"""
#     prompt = prompt.format(
#         Problem_Statement=problem,
#         Model_Solution=model_output,
#         Marking_Criteria_List=marking_list_text,
#         Total_Points=total_possible_points
#     )

#     llm_response = call_api(
#         prompt=prompt,
#         client=client,
#         max_tokens=model_args.max_tokens,
#         model=model_args.model_name,
#         temperature=model_args.temperature
#     )
#     # Log LLM judge response for total marking evaluation
#     print(f"[RM LLM Judge] Total Marking Evaluation Response:")
#     print(f"[RM LLM Judge] Problem: {problem[:100]}..." if len(problem) > 100 else f"[RM LLM Judge] Problem: {problem}")
#     print(f"[RM LLM Judge] Total Possible Points: {total_possible_points}")
#     print(f"[RM LLM Judge] Response: {llm_response}")
#     print(f"[DEBUG] Result from total marking xverify: {llm_response}")
#     score_str = extract_boxed_answer(llm_response)
#     try:
#         total_score = float(score_str)
#         total_score = max(0.0, min(total_score, total_possible_points))
#         return total_score, llm_response
#     except (ValueError, TypeError):
#         print(f"[DEBUG] Failed to parse score from: {score_str}")
#         return 0.0, llm_response


# def grade_marking_item_xverify(model_output: str, marking_item: dict, problem: str, model_args: Model_args) -> tuple[bool, str]:
#     """

#     Args:

#     Returns:
#         tuple: (is_satisfied: bool, llm_response: str)
#     """
#     def call_api(prompt:str,
#                 system_prompt:Optional[str]=None,
#                 client=None,
#                 base_url:Optional[str]=None,
#                 model:str="gpt-3.5-turbo",
#                 api_key:Union[None,str]=None,
#                 max_tokens:int=None,
#                 logprobs:bool=False,
#                 top_logprobs:int=1,
#                 **kwargs) -> str:
#         if not client:
#             assert api_key is not None,'Please input your api key'
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#                 )
#         messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
#         if prompt:
#             messages.append({"role": "user", "content": prompt})

#         # Only include top_logprobs when logprobs is True
#         api_kwargs = {
#             "model": model,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "logprobs": logprobs,
#             **kwargs
#         }
#         if logprobs:
#             api_kwargs["top_logprobs"] = top_logprobs

#         response = client.chat.completions.create(**api_kwargs)
#         return response.choices[0].message.content

#     client = OpenAI(api_key=model_args.api_key, base_url=model_args.base_url)

#     marking_desc = marking_item.get("desc", "")
#     marking_pt = marking_item.get("pt", 0.0)

#     expressions_info = ""
#     if "expressions" in marking_item and marking_item["expressions"]:
#         expressions_info = f"\n\nExpected expressions or patterns (if applicable):\n{marking_item['expressions']}"

#     prompt = """Determine if the marking criterion is satisfied by the student's complete solution.

# Guidelines:
# - Examine entire solution (reasoning, calculations, final answer)
# - Explicit statements or equivalent formulations count
# - Criterion must be substantially met

# Criterion: {Marking_Criterion}
# Points: {Points}

# Output:
# <think>
# 1. Criterion: [restate requirement]
# 2. Solution examination: [relevant parts]
# 3. Evaluation: [satisfied/not]
# 4. Conclusion: Satisfied/Not Satisfied
# </think>
# \\boxed{{Satisfied}} or \\boxed{{Not Satisfied}}

# Problem: {Problem_Statement}
# Solution: {Model_Solution}
# {Expressions_Info}"""
#     prompt = prompt.format(
#         Problem_Statement=problem,
#         Model_Solution=model_output,
#         Marking_Criterion=marking_desc,
#         Points=marking_pt,
#         Expressions_Info=expressions_info
#     )

#     llm_response = call_api(
#         prompt=prompt,
#         client=client,
#         max_tokens=model_args.max_tokens,
#         model=model_args.model_name,
#         temperature=model_args.temperature
#     )
#     # Log LLM judge response for individual marking item evaluation
#     print(f"[RM LLM Judge] Individual Marking Item Evaluation:")
#     print(f"[RM LLM Judge] Criterion: {marking_desc[:150]}..." if len(marking_desc) > 150 else f"[RM LLM Judge] Criterion: {marking_desc}")
#     print(f"[RM LLM Judge] Points: {marking_pt}")
#     print(f"[RM LLM Judge] Response: {llm_response}")
#     print(f"[DEBUG] Result from marking xverify: {llm_response}")
#     extracted_answer = extract_boxed_answer(llm_response)
#     is_satisfied = extracted_answer == "Satisfied"
#     return is_satisfied, llm_response


# def last_boxed_only_string(string):
#     idx = string.rfind("\\boxed")
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#         if string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break
#         i += 1

#     if right_brace_idx == None:
#         retval = None
#     else:
#         retval = string[idx : right_brace_idx + 1]

#     return retval

# def remove_boxed(s):
#     left = "\\boxed{"
#     try:
#         assert s[: len(left)] == left
#         assert s[-1] == "}"
#         return s[len(left) : -1]
#     except:
#         return None

# def extract_boxed_answer(solution: str) -> str:
#     """Extract the answer from inside a LaTeX \\boxed{} command"""
#     solution = last_boxed_only_string(solution)
#     solution = remove_boxed(solution)
#     return solution

# def grade(model_answer: str, gt_answer: str, is_matched: bool, problem=None, use_xverify=False, model_args=None, max_point=1.0, skip_xverify_threshold=0.8):
#     """

#     Args:

#     Returns:
#         tuple: (correct: bool, score_by: str, extracted_pred: str, extracted_gt: str, llm_judge_responses: list)
#     """
#     if model_answer is None:
#         model_answer = ""
#     if gt_answer is None:
#         gt_answer = ""

#     if "\\boxed" in gt_answer:
#         extracted = extract_boxed_answer(gt_answer)
#         if extracted is not None:
#             gt_answer = extracted

#     score_by = "not_scored"
#     correct, pred, gold = grade_answer_mathd(model_answer, gt_answer)
#     extracted_pred = pred
#     extracted_gt = gold

#     split_answer = model_answer.split("=")[-1]
#     split_gt = gt_answer.split("=")[-1]
#     enable_split = (split_answer != extracted_pred) or (split_gt != gt_answer)

#     if not correct:
#         correct, pred, gold = grade_answer_sympy(model_answer, gt_answer)
#         extracted_pred = pred
#         extracted_gt = gold
#         if (not correct) and enable_split:
#             correct, extracted_pred, extracted_gt = grade_answer_sympy(split_answer, split_gt)
#     else:
#         score_by = "mathd"

#     if not correct:
#         correct, pred, gold = grade_answer_math_verify(model_answer, gt_answer)
#         extracted_pred = pred
#         extracted_gt = gold
#         try:
#             if (isinstance(extracted_pred[0], sympy.core.numbers.Integer) or isinstance(extracted_pred[0], sympy.core.numbers.Float) or (isinstance(extracted_pred[0], sympy.core.numbers.Rational))) and isinstance(extracted_gt[0], sympy.sets.sets.Interval):
#                 correct = 1.0 if extracted_gt[0].contains(extracted_pred[0]) == True else 0.0
#         except:
#             pass
#         if (not correct) and enable_split:
#             correct, extracted_pred, extracted_gt = grade_answer_math_verify(split_answer, split_gt)
#     elif score_by == "not_scored":
#         score_by = "sympy_verify"

#     rule_based_score = 1.0 if correct else 0.0
#     rule_based_point = rule_based_score * max_point

#     should_skip_xverify = rule_based_point >= (skip_xverify_threshold * max_point)

#     llm_judge_responses = []
#     if not correct and not should_skip_xverify:
#         if use_xverify:
#             if model_args is None:
#                 model_args = Model_args()
#             correct, llm_response = grade_answer_xverify(model_answer, gt_answer, problem, model_args)
#             llm_judge_responses.append({
#                 "type": "answer_xverify",
#                 "response": llm_response,
#                 "model_answer": model_answer,
#                 "ground_truth": gt_answer
#             })
#             extracted_pred = model_answer
#             extracted_gt = gt_answer
#             if (not correct) and enable_split:
#                 correct, llm_response_split = grade_answer_xverify(split_answer, split_gt, problem, model_args)
#                 llm_judge_responses.append({
#                     "type": "answer_xverify_split",
#                     "response": llm_response_split,
#                     "model_answer": split_answer,
#                     "ground_truth": split_gt
#                 })
#             if correct:
#                 score_by = "xverify"
#     elif score_by == "not_scored":
#         score_by = "math_verify"

#     return correct, score_by, extracted_pred, extracted_gt, llm_judge_responses

# def last_n_boxed_strings(string, n):
#     boxed_list = []

#     work_str = string[:]
#     while work_str and len(boxed_list) < n:
#         idx = work_str.rfind("\\boxed")
#         if idx < 0:
#             idx = work_str.rfind("\\fbox")

#         if idx < 0:
#             break

#         i = idx
#         right_brace_idx = None
#         num_left_braces_open = 0
#         while i < len(work_str):
#             if work_str[i] == "{":
#                 num_left_braces_open += 1
#             elif work_str[i] == "}":
#                 num_left_braces_open -= 1
#                 if num_left_braces_open == 0:
#                     right_brace_idx = i
#                     break
#             i += 1

#         if right_brace_idx is not None:
#             boxed_expr = work_str[idx: right_brace_idx + 1]
#             boxed_list.append(boxed_expr)
#             work_str = work_str[:idx]
#         else:
#             work_str = work_str[:idx]

#     boxed_list.reverse()
#     return boxed_list

# def get_answer_str(s: str, return_origin=False, num_answers=1):
#     boxed_list = last_n_boxed_strings(s, num_answers)
#     answer_list = [remove_boxed(b) if b else "" for b in boxed_list]

#     missing = num_answers - len(answer_list)
#     fill_str = s if return_origin else ""
#     answer_list = [fill_str] * missing + answer_list

#     return answer_list

# def solution2answer(solution: str, math_mode="eval_peeking", return_origin=False, num_answers=1) -> tuple[bool, list | str]:
#     answer = solution
#     if math_mode == "eval_peeking":
#         answer = get_answer_str(solution, return_origin, num_answers)
#     else:
#         raise ValueError(f"Invalid math_mode: {math_mode}")
#     return answer

# def answer_tag_reward_fn_for_r1(model_output: str, ground_truths, problem=None, points=None, use_xverify=False, model_args=None, marking=None, marking_mode="total_score", skip_xverify_threshold=0.8):
#     """

#     Args:

#     Returns:
#         tuple: (total_score, point, extracted_preds, extracted_gts, scored_by_list, score_noxverify, point_noxverify, llm_judge_responses)
#     """
#     extracted_pred = model_output
#     is_matched = False

#     num_questions_to_answer = len(ground_truths)

#     extracted_answers = solution2answer(str(model_output), num_answers=num_questions_to_answer)
#     ground_truths = [solution2answer(str(gt), return_origin=True)[0] for gt in ground_truths]

#     if not any(extracted_answers):
#         return 0.0, 0.0, extracted_answers, ground_truths, ["not_scored"] * num_questions_to_answer, 0.0, 0.0, []
#     is_matched = True

#     if points is None or len(points) == 0:
#         points = [1.0] * num_questions_to_answer
#     elif len(points) != num_questions_to_answer:
#         points = [1.0] * num_questions_to_answer

#     total_score = 0.0
#     extracted_preds, extracted_gts, scored_by_list = [], [], []
#     score_list = []
#     all_llm_judge_responses = []
#     for extracted_pred, ground_truth, max_point in zip(extracted_answers, ground_truths, points):
#         score, score_by, extracted_pred, extracted_gt, llm_responses = grade(
#             extracted_pred, ground_truth, is_matched, problem,
#             use_xverify=use_xverify, model_args=model_args,
#             max_point=max_point, skip_xverify_threshold=skip_xverify_threshold
#         )
#         score_list.append(score)
#         scored_by_list.append(score_by)
#         extracted_preds.append(extracted_pred)
#         extracted_gts.append(extracted_gt)
#         all_llm_judge_responses.extend(llm_responses)

#     total_score = sum(score_list) / num_questions_to_answer

#     if len(points) == num_questions_to_answer:
#         try:
#             point = sum([s * p for s, p in zip(score_list, points)])
#         except:
#             point = 0
#     else:
#         point = score

#     rule_based_point = point
#     total_max_point = sum(points)

#     should_skip_marking_xverify = rule_based_point >= (skip_xverify_threshold * total_max_point)

#     if should_skip_marking_xverify:
#         print(f"[DEBUG] Skipping marking xverify: rule_based_point={rule_based_point:.2f} >= {skip_xverify_threshold}*{total_max_point:.2f}={skip_xverify_threshold * total_max_point:.2f}")

#     marking_point = 0.0
#     if use_xverify and marking is not None and len(marking) > 0 and not should_skip_marking_xverify:

#         def normalize_marking_item(item):
#             if isinstance(item, dict):
#                 desc = item.get("desc") or item.get("description", "")
#                 pt = item.get("pt")
#                 if pt is None:
#                     pt = item.get("point", 0.0)
#                 if pt == 0.0 or pt is None:
#                     import re
#                     m = re.search(r"Award\s*([\d.]+)\s*pt", desc, re.IGNORECASE)
#                     if m:
#                         pt = float(m.group(1))
#                 return {"desc": str(desc), "pt": float(pt) if pt is not None else 0.0}
#             elif isinstance(item, str):
#                 import re
#                 m = re.search(r"Award\s*([\d.]+)\s*pt", item, re.IGNORECASE)
#                 pt = float(m.group(1)) if m else 0.0
#                 return {"desc": item, "pt": pt}
#             else:
#                 return {"desc": str(item), "pt": 0.0}

#         def flatten_marking(marking_data):
#             all_marking_items = []

#             if not isinstance(marking_data, list):
#                 marking_data = [marking_data]

#             for item in marking_data:
#                 if isinstance(item, dict):
#                     all_marking_items.append(normalize_marking_item(item))
#                 elif isinstance(item, list):
#                     if len(item) > 0:
#                         if isinstance(item[0], list):
#                             for sub_list in item:
#                                 if isinstance(sub_list, list):
#                                     for sub_item in sub_list:
#                                         all_marking_items.append(normalize_marking_item(sub_item))
#                                 else:
#                                     all_marking_items.append(normalize_marking_item(sub_list))
#                         else:
#                             for sub_item in item:
#                                 all_marking_items.append(normalize_marking_item(sub_item))
#                 else:
#                     all_marking_items.append(normalize_marking_item(item))

#             return all_marking_items

#         all_marking_items = flatten_marking(marking)

#         if marking_mode == "total_score":
#             try:
#                 marking_point, llm_response = grade_marking_total_xverify(
#                     model_output=str(model_output),
#                     marking_items=all_marking_items,
#                     problem=problem if problem else "",
#                     model_args=model_args
#                 )
#                 all_llm_judge_responses.append({
#                     "type": "marking_total_xverify",
#                     "response": llm_response,
#                     "marking_items": all_marking_items,
#                     "score": marking_point
#                 })
#                 print(f"[DEBUG] Total marking score (mode=total_score): {marking_point}")
#             except Exception as e:
#                 print(f"[DEBUG] Error evaluating total marking score: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 marking_point = 0.0
#         else:
#             for marking_item in all_marking_items:
#                 if isinstance(marking_item, dict) and marking_item.get("pt", 0.0) > 0:
#                     try:
#                         is_satisfied, llm_response = grade_marking_item_xverify(
#                             model_output=str(model_output),
#                             marking_item=marking_item,
#                             problem=problem if problem else "",
#                             model_args=model_args
#                         )
#                         all_llm_judge_responses.append({
#                             "type": "marking_item_xverify",
#                             "response": llm_response,
#                             "marking_item": marking_item,
#                             "is_satisfied": is_satisfied
#                         })
#                         if is_satisfied:
#                             marking_point += marking_item.get("pt", 0.0)
#                             print(f"[DEBUG] Marking item satisfied: {marking_item.get('desc', '')[:50]}... (+{marking_item.get('pt', 0.0)} pts)")
#                         else:
#                             print(f"[DEBUG] Marking item not satisfied: {marking_item.get('desc', '')[:50]}... (0 pts)")
#                     except Exception as e:
#                         print(f"[DEBUG] Error evaluating marking item: {e}")
#                         import traceback
#                         traceback.print_exc()
#                         pass

#     original_point = point
#     point = max(point, marking_point)
#     if marking_point > 0:
#         print(f"[DEBUG] Original point: {original_point}, Marking point: {marking_point}, Final point: {point}")

#     if use_xverify:
#         score_noxverify = sum([int(s_by != "xverify") * s for s_by, s in zip(scored_by_list, score_list)])
#         point_noxverify = sum([int(s_by != "xverify") * p * s for s_by, p, s in zip(scored_by_list, points, score_list)])
#         if marking_point > 0:
#             point_noxverify = max(point_noxverify, marking_point)
#     else:
#         score_noxverify = total_score
#         point_noxverify = point
#     return total_score, point, extracted_preds, extracted_gts, scored_by_list, score_noxverify, point_noxverify, all_llm_judge_responses


# def compute_score_p1(model_output, label, points=None, question="", use_xverify=False, model_port=34882, marking=None, marking_mode="total_score", skip_xverify_threshold=0.8):
#     """

#     Args:

#     Returns:
#     """
#     # todo: add use_xverify as a config
#     ground_truths = label
#     if isinstance(ground_truths, str):
#         ground_truths = [ground_truths]
#     print(f"[DEBUG] points: {points}, ground_truths: {ground_truths}")
#     if marking is not None:
#         print(f"[DEBUG] marking: {marking}, marking_mode: {marking_mode}")
#     print(f"[DEBUG] skip_xverify_threshold: {skip_xverify_threshold}")

#     model_args = Model_args(model_port=model_port)

#     score, point, extracted_pred, extracted_gt, scored_by, score_noxverify, point_noxverify, llm_judge_responses = answer_tag_reward_fn_for_r1(
#         model_output, ground_truths, question, points, use_xverify, model_args,
#         marking=marking, marking_mode=marking_mode, skip_xverify_threshold=skip_xverify_threshold
#     )
#     print(f"[DEBUG] Result score: {score}, point: {point}, score_by: {scored_by}, score_noxverify: {score_noxverify}, point_noxverify: {point_noxverify}")

#     return {
#         "score": score,
#         "point": point,
#         "acc": abs(score - 1.0) < 1e-5,
#         "extracted_gt": str(extracted_gt),
#         "extracted_pred": str(extracted_pred),
#         "scored_by": str(scored_by),
#         "score_noxverify": score_noxverify,
#         "point_noxverify": point_noxverify,
#         "llm_judge_responses": llm_judge_responses,
#     }


# if __name__ == "__main__":
#     # Example usage
#     model_output = r"The answer is <answer>$\boxed{\dfrac{5}{3}}$</answer>"
#     ground_truth = ["$[1.64,1.70]$"]
#     question = "What is the answer to the ultimate question of life, the universe, and everything?"

#     result = compute_score_p1(model_output, ground_truth, points=[1.0], question=question, use_xverify=True)
#     print(result)  # Should print a dictionary with score and correctness information.
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import os
import re
import signal
import math
import multiprocessing
import queue as queue_module
import time
import traceback
import threading
from openai import OpenAI
from functools import wraps, partial
from itertools import islice, zip_longest
from typing import Optional, Union
from pylatexenc import latex2text
from decimal import Decimal, localcontext
import sympy
from sympy import N, Pow, Mul
from sympy.parsing import sympy_parser
from math_verify import (ExprExtractionConfig,
                         LatexExtractionConfig, parse, verify)
# from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

_MATH_VERIFY_TIMEOUT_SECONDS = float(
    os.getenv("P1_MATH_VERIFY_TIMEOUT_SECONDS", "5"))
_MATH_VERIFY_PARSE_TIMEOUT_SECONDS = float(
    os.getenv("P1_MATH_VERIFY_PARSE_TIMEOUT_SECONDS", "1"))
_MATH_VERIFY_MAX_CHARS = int(os.getenv("P1_MATH_VERIFY_MAX_CHARS", "2048"))
_MATH_VERIFY_MAX_TOTAL_CHARS = int(
    os.getenv("P1_MATH_VERIFY_MAX_TOTAL_CHARS", "4096"))
_MATH_VERIFY_MAX_LATEX_COMMANDS = int(
    os.getenv("P1_MATH_VERIFY_MAX_LATEX_COMMANDS", "128"))
_MATH_VERIFY_MAX_BRACES = int(os.getenv("P1_MATH_VERIFY_MAX_BRACES", "256"))
_MATH_VERIFY_MAX_BRACE_DEPTH = int(
    os.getenv("P1_MATH_VERIFY_MAX_BRACE_DEPTH", "32"))
_HEAVY_PARSER_MAX_FACTORIAL = int(
    os.getenv("P1_HEAVY_PARSER_MAX_FACTORIAL", "256"))
_MATH_VERIFY_MAX_PROCS = int(os.getenv("P1_MATH_VERIFY_MAX_PROCS", "16"))
_MATH_VERIFY_DEFAULT_MP_CONTEXT = (
    "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
)
_MATH_VERIFY_MP_CONTEXT = os.getenv(
    "P1_MATH_VERIFY_MP_CONTEXT",
    _MATH_VERIFY_DEFAULT_MP_CONTEXT,
)
_MATH_VERIFY_PROCESS_SEMAPHORE = threading.BoundedSemaphore(
    max(1, _MATH_VERIFY_MAX_PROCS))


def timeout_handler(timeout_seconds=10, default_return=None):
    """
    Thread-safe timeout decorator for functions that might hang.
    Works in both single-threaded and multi-threaded environments.
    Returns default_return instead of raising TimeoutError when timeout occurs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            completed = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    completed.set()

            # Start the function in a separate thread
            thread = threading.Thread(target=target)
            thread.daemon = True  # Dies when main thread dies
            thread.start()

            # Wait for completion with timeout
            if completed.wait(timeout_seconds):
                # Function completed within timeout
                if exception[0]:
                    raise exception[0]
                return result[0]
            else:
                # Function timed out - return default value instead of raising error
                print("[WARNING] Function {} timed out after {} seconds, returning default value".format(
                    func.__name__, timeout_seconds))
                return default_return

        return wrapper
    return decorator


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "p . m",
    "lb",
    "tile",
    "per",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "month",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
    "kilogram",
    "second",
    "ampere",
    "A",
    "K",
    "mol",
    "cd",
    "N",
    "J",
    "W",
    "Pa",
    "Hz",
    "C",
    "V",
    "Ω",
    "F",
    "T",
    "H",
    "eV",
    "kW·h",
    "atm",
    "bar",
    "°C"
]
unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
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
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text +
                             r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  # implicit mults
    return step


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def sympy_normalize_answer(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _expr = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", expr)
            if _expr != "":
                expr = _expr

    expr = re.sub(rf"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def judge_MC(pred, gold):
    common_answer = [chr(i) for i in range(65, 91)]  # 'A'~'Z'
    if pred == gold:
        return True
    else:
        if pred.startswith("[") and pred.endswith("]"):
            pred = pred.strip("[]")
        if not pred:
            return False
        if pred[0] in common_answer and (len(pred) > 1 and (pred[1] == "." or pred[1] == ":")):
            return pred[0] == gold
        if "'{}'".format(gold) in pred:
            return True
        else:
            return False


def judge_TF(pred, gold):
    def contains_chinese(d):
        def is_chinese_char(ch):
            return '\u4e00' <= ch <= '\u9fff'

        def check(value):
            if isinstance(value, str):
                return any(is_chinese_char(ch) for ch in value)
            elif isinstance(value, dict):
                return any(check(v) for v in value.values())
            elif isinstance(value, list):
                return any(check(item) for item in value)
            return False

        return check(d)

    if contains_chinese(pred):
        if pred in ["是", "对", "正确", "能"]:
            pred = "TRUE"
        elif pred in ["否", "错", "错误", "不能"]:
            pred = "FALSE"
    else:
        pred = pred.upper()
    answers = ["TRUE", "FALSE", "T", "F", "YES", "NO", "Y", "N"]
    gold = gold.upper()
    if gold not in answers or pred not in answers:
        return False
    if gold in ["TRUE", "YES", "T", "Y"]:
        gold = "TRUE"
    if gold in ["FALSE", "NO", "F", "N"]:
        gold = "FALSE"
    if pred in ["TRUE", "YES", "T", "Y"]:
        pred = "TRUE"
    if pred in ["FALSE", "NO", "F", "N"]:
        pred = "FALSE"
    return pred == gold


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True, given_answer_normalized_mathd, ground_truth_normalized_mathd
    return False, given_answer_normalized_mathd, ground_truth_normalized_mathd


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
        evaluate=False,
    )


def should_allow_eval(expr: str):
    def count_unknown_letters_in_expr(expr: str):
        expr = expr.replace("sqrt", "")
        expr = expr.replace("frac", "")
        letters_in_expr = set([x for x in expr if x.isalpha()])
        return len(letters_in_expr)
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def handle_pi(string, pi):
    if isinstance(string, str) and "pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:

            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + "*{}".format(pi) + string[idx + 2:]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + "1*{}".format(pi) + string[idx + 2:]

            # Find the next occurrence of "\pi"
            idx = string.find("pi", idx + 1)

        # Evaluate the expression using eval() function
        try:
            string = eval(string)
        except:
            pass

    return string

# @timeout_handler(timeout_seconds=5)  # Sympy can hang on complex expressions


def are_equal_under_sympy(gold: str, pred: str, precision: float = 2e-3):
    def is_scientific_notation(expr):
        return (
            isinstance(expr, Mul)
            and isinstance(expr.args[1], Pow)
            and expr.args[1].args[0] == 10
        )

    def to_scientific_notation_sympy(num):
        num_sci = "{:.2e}".format(num)  # e.g., "1.23e-5"
        base, exponent = num_sci.split("e")
        return "{}*10**{}".format(base, int(exponent))

    def count_decimal_places(x, tol=1e-6):
        with localcontext() as ctx:
            ctx.prec = 20  
            d = Decimal(str(x)).normalize()
            s = format(d, "f")  
            if "." not in s:
                return 0
            integer_part, decimal_part = s.split(".")
            clean_decimal = ""
            for i, ch in enumerate(decimal_part):
                clean_decimal += ch
                if abs(x - round(x, i+1)) <= tol:
                    break

            return len(clean_decimal)

    try:
        if pred == gold:
            return True

        pred_value = float(pred)
        gold_value = float(gold)
        min_decimal_places = min(count_decimal_places(
            gold_value), count_decimal_places(pred_value))

        pred_value = round(pred_value, min_decimal_places)
        gold_value = round(gold_value, min_decimal_places)
        if abs((pred_value - gold_value) / gold_value) <= precision * 1.01:
            return True

        spred = _sympy_parse(to_scientific_notation_sympy(float(pred)))
        sgold = _sympy_parse(to_scientific_notation_sympy(float(gold)))
        if is_scientific_notation(spred) and is_scientific_notation(sgold):
            base_pred, exponent_pred = N(
                spred.args[0]), N(spred.args[1].args[1])
            base_gold, exponent_gold = N(
                sgold.args[0]), N(sgold.args[1].args[1])
            min_decimal_places = min(count_decimal_places(
                base_gold), count_decimal_places(base_pred))
            base_pred = round(base_pred, min_decimal_places)
            base_gold = round(base_gold, min_decimal_places)
            if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
                return True
    except Exception:
        pass

    try:
        if should_allow_eval(gold) and should_allow_eval(pred):
            exp_gold = _sympy_parse(gold)
            exp_pred = _sympy_parse(pred)

            expr = (exp_gold - exp_pred) / (exp_gold if exp_gold != 0 else 1)
            simplified = sympy.simplify(expr)
            if abs(N(simplified)) <= precision * 1.01:
                return True
            if is_scientific_notation(exp_pred) != is_scientific_notation(exp_gold):
                if is_scientific_notation(exp_pred):
                    gold = to_scientific_notation_sympy(float(gold))
                    exp_gold = _sympy_parse(gold)
                else:
                    pred = to_scientific_notation_sympy(float(pred))
                    exp_pred = _sympy_parse(pred)

            if is_scientific_notation(exp_pred) and is_scientific_notation(exp_gold):
                base_pred, exponent_pred = N(
                    exp_pred.args[0]), N(exp_pred.args[1].args[1])
                base_gold, exponent_gold = N(
                    exp_gold.args[0]), N(exp_gold.args[1].args[1])
                min_decimal_places = min(count_decimal_places(
                    base_gold), count_decimal_places(base_pred))
                base_pred = round(base_pred, min_decimal_places)
                base_gold = round(base_gold, min_decimal_places)

                if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
                    return True
            else:
                if N(exp_pred) == N(exp_gold):
                    return True
    except Exception:
        pass

    return False

# @timeout_handler(timeout_seconds=20)  # Sympy grading can take time


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = sympy_normalize_answer(ground_truth)
    given_normalized = sympy_normalize_answer(given_answer)
    if ground_truth_normalized is None:
        return False, given_normalized, ground_truth_normalized

    if ground_truth_normalized == given_normalized:
        return True, given_normalized, ground_truth_normalized

    if len(given_normalized) == 0:
        return False, given_normalized, ground_truth_normalized

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            # elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            #     # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
            #     is_correct = False
            else:
                is_correct = judge_MC(given_elem, ground_truth_elem) or judge_TF(
                    given_elem, ground_truth_elem)
                if not is_correct:
                    if "pi" in given_elem or "pi" in ground_truth_elem:
                        equivs = []
                        for pi in [math.pi, 3.14, 180]:
                            given_elem_pi = handle_pi(given_elem, pi)
                            ground_truth_elem_pi = handle_pi(
                                ground_truth_elem, pi)
                            try:
                                equivs.append(are_equal_under_sympy(
                                    ground_truth_elem_pi, given_elem_pi))
                            except TimeoutError:
                                equivs.append(False)
                        is_correct = any(equivs)
                    else:
                        try:
                            is_correct = are_equal_under_sympy(
                                ground_truth_elem, given_elem)
                        except TimeoutError:
                            is_correct = False
            if not is_correct:
                break

    return is_correct, given_normalized, ground_truth_normalized


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(
                list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


def _latex_braces_invalid_or_too_deep(text: str, max_depth: int) -> bool:
    depth = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
            if depth > max_depth:
                return True
        elif char == "}":
            if depth == 0:
                return True
            depth -= 1
    return depth != 0


def _has_unbalanced_math_delimiters(text: str) -> bool:
    dollar_count = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "$":
            dollar_count += 1
    return dollar_count % 2 == 1


def _has_expensive_integer_factorial(text: str) -> bool:
    for match in re.finditer(r"(?<![A-Za-z0-9_])(\d+)\s*!", text):
        try:
            if int(match.group(1)) > _HEAVY_PARSER_MAX_FACTORIAL:
                return True
        except ValueError:
            continue
    return False


def _should_skip_math_verify(given_answer: str, ground_truth: str) -> bool:
    texts = (given_answer or "", ground_truth or "")
    if sum(len(text) for text in texts) > _MATH_VERIFY_MAX_TOTAL_CHARS:
        return True
    for text in texts:
        if len(text) > _MATH_VERIFY_MAX_CHARS:
            return True
        if len(text) > 128 and repeatness(text):
            return True
        if text.count("\\") > _MATH_VERIFY_MAX_LATEX_COMMANDS:
            return True
        if text.count("{") + text.count("}") > _MATH_VERIFY_MAX_BRACES:
            return True
        if _latex_braces_invalid_or_too_deep(text, _MATH_VERIFY_MAX_BRACE_DEPTH):
            return True
        if _has_unbalanced_math_delimiters(text):
            return True
        # Evaluating expressions such as 1/1994! in Sympy can outlive the
        # asyncio timeout because the grader runs in a non-cancellable thread.
        # Exact/string graders run before this check, so only the risky
        # symbolic-equivalence fallback is skipped.
        if _has_expensive_integer_factorial(text):
            return True
    return False


def _math_verify_worker(given_answer: str, ground_truth: str, result_queue):
    try:
        given_answer = given_answer.replace("\n", "")
        ground_truth = ground_truth.replace("\n", "")
        if "$" not in given_answer:
            given_answer = f"${given_answer}$"
        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"

        parsed_answer = parse(
            given_answer,
            extraction_config=(
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ),
            fallback_mode="no_fallback",
            extraction_mode=["first_match"],
            parsing_timeout=_MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
        )
        parsed_ground_truth = parse(
            ground_truth,
            extraction_config=(
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ),
            fallback_mode="no_fallback",
            extraction_mode=["first_match"],
            parsing_timeout=_MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
        )
        correct = verify(
            parsed_ground_truth,
            parsed_answer,
            numeric_precision=3,
            timeout_seconds=_MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
        )
        result_queue.put(
            ("ok", bool(correct), parsed_answer, parsed_ground_truth))
    except Exception as e:
        result_queue.put(("error", str(e), given_answer, ground_truth))


def _run_math_verify_process(given_answer: str, ground_truth: str, timeout_seconds: float):
    ctx = multiprocessing.get_context(_MATH_VERIFY_MP_CONTEXT)
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_math_verify_worker,
        args=(given_answer, ground_truth, result_queue),
    )
    try:
        process.daemon = True
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            print("[WARNING] math_verify subprocess timed out after {} seconds".format(
                timeout_seconds))
            return False, given_answer, ground_truth
        try:
            status, first, second, third = result_queue.get_nowait()
        except queue_module.Empty:
            return False, given_answer, ground_truth
        if status == "ok":
            return first, second, third
        print("[DEBUG] Error in grade_answer_math_verify: {}".format(first))
        return False, second, third
    except Exception as e:
        print("[DEBUG] Failed to run math_verify subprocess: {}".format(e))
        return False, given_answer, ground_truth
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass


def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None or ground_truth is None:
        return False, given_answer, ground_truth
    given_answer = str(given_answer)
    ground_truth = str(ground_truth)
    if _should_skip_math_verify(given_answer, ground_truth):
        return False, given_answer, ground_truth
    acquired = _MATH_VERIFY_PROCESS_SEMAPHORE.acquire(
        timeout=_MATH_VERIFY_TIMEOUT_SECONDS)
    if not acquired:
        print("[WARNING] Skipping math_verify because subprocess slots are exhausted")
        return False, given_answer, ground_truth
    try:
        return _run_math_verify_process(
            given_answer,
            ground_truth,
            _MATH_VERIFY_TIMEOUT_SECONDS,
        )
    finally:
        _MATH_VERIFY_PROCESS_SEMAPHORE.release()


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def retry(max_attempts: int = 3, delay: int = 1, print_trace_back=False, return_error_info=False):
    assert isinstance(max_attempts, int) and isinstance(delay, int), 'Parameters must be integers'

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if print_trace_back:
                        e = traceback.format_exc()
                        error_info = ">>> Function {} failed on attempt {}: {}".format(
                            func.__name__, attempts + 1, e)
                        print(error_info)
                    time.sleep(delay)
                    attempts += 1
            if return_error_info:
                return error_info
            else:
                return None

        @attach_wrapper(wrapper)
        def set_max_attempts(new_max_attempts):
            nonlocal max_attempts
            max_attempts = new_max_attempts

        @attach_wrapper(wrapper)
        def set_delay(new_delay):
            nonlocal delay
            delay = new_delay

        wrapper.get_attempts = lambda: max_attempts
        wrapper.get_delay = lambda: delay
        return wrapper
    return decorator


class Model_args:
    use_model: bool = False
    model_name = os.environ.get("XVERIFY_MODEL_NAME", "Judge")  # Model name
    # API key used to access the model via API, if not available, set to None
    api_key = os.environ.get("XVERIFY_API_KEY", "None")
    # Anonymized model path or URL
    base_url = os.environ.get("XVERIFY_BASE_URL", "http://127.0.0.1:34881/v1")
    max_tokens = int(os.environ.get("XVERIFY_MAX_TOKENS", "4096"))
    temperature = float(os.environ.get("XVERIFY_TEMPERATURE", "0.1"))

# @timeout_handler(timeout_seconds=60)  # Longer timeout for API calls


def _parse_xverify_is_correct(response: str) -> bool:
    if response is None:
        return False

    text = str(response).strip()
    bracket_tags = re.findall(
        r"\[\s*(Correct|Incorrect)\s*\]", text, flags=re.IGNORECASE)
    if bracket_tags:
        return bracket_tags[-1].lower() == "correct"

    boxed_tags = re.findall(
        r"\\boxed\{\s*(Correct|Incorrect)\s*\}", text, flags=re.IGNORECASE)
    if boxed_tags:
        return boxed_tags[-1].lower() == "correct"

    final_line = text.splitlines(
    )[-1].strip().strip(".").lower() if text else ""
    return final_line == "correct"


def grade_answer_xverify(given_answer: str, ground_truth: str, problem: str, model_args: Model_args) -> bool:
    @retry(max_attempts=2, delay=1, print_trace_back=True, return_error_info=True)
    def call_api(prompt: str,
                 system_prompt: Optional[str] = None,
                 client=None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 api_key: Union[None, str] = None,
                 max_tokens: int = None,
                 temperature: float = 0.7,
                 logprobs: bool = False,
                 top_logprobs: int = 1,
                 **kwargs) -> str:
        if not client:
            assert api_key is not None, 'Please input your api key'
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        if not logprobs:
            top_logprobs = None

        messages = [{"role": "system", "content": system_prompt}
                    ] if system_prompt is not None else []
        if prompt:
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **kwargs
        )
        return response.choices[0].message.content
    # print("[DEBUG] Enter xverify.")
    client = OpenAI(api_key=model_args.api_key, base_url=model_args.base_url)
    prompt = f'''
You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will
receive a question, an output sentence, and the correct answer. Your task is to determine if the output
sentence accurately answers the question based on the provided correct answer. Respond with either
[Correct] or [Incorrect].
-
Special considerations:
1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers
modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the
final answer is unclear or incorrect, respond with [Incorrect].
2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent such as 256/55=4.65,
respond with [Correct].
3. **Phycis Problems**: If the values match such as 3=3 \\, \\text{{GHz}} return [Correct].
4. **Explicit Options**: If the question provides explicit candidate answers, the output will be
considered correct if it clearly indicates the correct option's code or the correct option's content.
5. **No Explicit Options**: If the question does not provide explicit options, the output must align
with the correct answer in content and meaning to be considered [Correct].
-
Question: """{problem}"""
Output sentence: """{given_answer}"""
Correct answer: {ground_truth}
Judgement:
'''
    correct = call_api(prompt=prompt,
                       client=client,
                       max_tokens=model_args.max_tokens,
                       model=model_args.model_name,
                       temperature=model_args.temperature)
    is_correct = _parse_xverify_is_correct(correct)
    print("[DEBUG] Prediction: {}, Answer: {}, Parsed xverify: {}, Result from xverify: {}".format(
        given_answer, ground_truth, is_correct, correct))
    return is_correct


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx: right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

# @timeout_handler(timeout_seconds=10)


def grade_with_timeout(model_answer: str, gt_answer: str, is_matched: bool, problem=None, use_xverify=False, base_url=None):
    """Internal grade function with timeout protection"""
    if model_answer is None or gt_answer is None:
        return False, "error", str(model_answer), str(gt_answer)
    if "\\boxed" in gt_answer:
        try:
            gt_answer = extract_boxed_answer(gt_answer)
        except:
            pass  # Keep original answer if extraction fails
    score_by = "not_scored"
    correct = False
    # extracted_pred = str(model_answer)
    # extracted_gt = str(gt_answer)
    correct, pred, gold = grade_answer_mathd(model_answer, gt_answer)
    extracted_pred = pred
    extracted_gt = gold
    # Try mathd grading first
    if "=" in model_answer:
        split_answer = model_answer.split("=")[-1]
    else:
        split_answer = model_answer
    if "=" in gt_answer:
        split_gt = gt_answer.split("=")[-1]
    else:
        split_gt = gt_answer
    enable_split = (split_answer != extracted_pred) or (split_gt != gt_answer)
    skip_heavy_parsers = _should_skip_math_verify(model_answer, gt_answer)
    skip_split_heavy_parsers = _should_skip_math_verify(split_answer, split_gt)

    # Try sympy grading if mathd failed
    if correct:
        score_by = "mathd"
    else:
        if not skip_heavy_parsers:
            correct, pred, gold = grade_answer_sympy(model_answer, gt_answer)
            extracted_pred = pred
            extracted_gt = gold
        if (not correct) and enable_split and not skip_split_heavy_parsers:
            correct, extracted_pred, extracted_gt = grade_answer_sympy(
                split_answer, split_gt)
        if correct:
            score_by = "sympy_verify"

    if not correct:
        if not skip_heavy_parsers:
            correct, pred, gold = grade_answer_math_verify(
                model_answer, gt_answer)
            extracted_pred = pred
            extracted_gt = gold
            try:
                if (isinstance(extracted_pred[0], sympy.core.numbers.Integer) or isinstance(extracted_pred[0], sympy.core.numbers.Float) or (isinstance(extracted_pred[0], sympy.core.numbers.Rational))) and isinstance(extracted_gt[0], sympy.sets.sets.Interval):
                    correct = 1.0 if extracted_gt[0].contains(
                        extracted_pred[0]) == True else 0.0
            except:
                pass
        if (not correct) and enable_split and not skip_split_heavy_parsers:
            correct, extracted_pred, extracted_gt = grade_answer_math_verify(
                split_answer, split_gt)
        if correct:
            score_by = "math_verify"

    if not correct and use_xverify:
        model_args = Model_args()
        model_args.base_url = base_url
        try:
            correct = grade_answer_xverify(
                model_answer, gt_answer, problem, model_args)
            extracted_pred = model_answer
            extracted_gt = gt_answer
            if (not correct) and enable_split:
                try:
                    correct = grade_answer_xverify(
                        split_answer, split_gt, problem, model_args)
                except (TimeoutError, Exception):
                    pass
            if correct:
                score_by = "xverify"
        except (TimeoutError, Exception) as e:
            print("[WARNING] xverify grading failed: {}".format(e))
            pass

    return correct, score_by, extracted_pred, extracted_gt


def grade(model_answer: str, gt_answer: str, is_matched: bool, problem=None, use_xverify=False, base_url=None):
    """
    Grade function with fault tolerance and timeout handling.
    Returns default values for failed cases.

    **THREAD-SAFE**: This function can be safely called from multiple threads simultaneously.
    Uses thread-based timeout mechanism instead of signals for compatibility.

    Args:
        model_answer: The model's answer
        gt_answer: The ground truth answer
        is_matched: Whether the answer format is matched
        problem: Optional problem text for xverify
        use_xverify: Whether to use xverify model

    Returns:
        tuple: (correct, score_by, extracted_pred, extracted_gt)
               For failed cases, returns (False, "error"/"timeout", model_answer, gt_answer)

    Thread Safety:
        - Safe to call from multiple threads
        - Each call gets independent timeout handling
        - No shared state between calls
        - Graceful degradation on errors/timeouts
    """
    default_return = (False, "error", str(model_answer), str(gt_answer))

    try:
        # Attempt to grade with timeout protection
        result = grade_with_timeout(
            model_answer, gt_answer, is_matched, problem, use_xverify, base_url)
        return result

    except TimeoutError as e:
        print("[WARNING] Grade function timed out: {}".format(e))
        return (False, "timeout", str(model_answer), str(gt_answer))

    except Exception as e:
        print("[WARNING] Grade function failed with error: {}: {}".format(
            type(e).__name__, e))
        # Log the full traceback for debugging
        print("[DEBUG] Full traceback: {}".format(traceback.format_exc()))
        return default_return


def last_n_boxed_strings(string, n):
    boxed_list = []

    work_str = string[:]
    while work_str and len(boxed_list) < n:
        idx = work_str.rfind("\\boxed")
        if idx < 0:
            idx = work_str.rfind("\\fbox")

        if idx < 0:
            break

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(work_str):
            if work_str[i] == "{":
                num_left_braces_open += 1
            elif work_str[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is not None:
            boxed_expr = work_str[idx: right_brace_idx + 1]
            boxed_list.append(boxed_expr)
            work_str = work_str[:idx]
        else:
            work_str = work_str[:idx]

    boxed_list.reverse()
    return boxed_list


def get_answer_str(s: str, return_origin=False, num_answers=1):
    boxed_list = last_n_boxed_strings(s, num_answers)
    answer_list = [remove_boxed(b) if b else "" for b in boxed_list]

    missing = num_answers - len(answer_list)
    fill_str = s if return_origin else ""
    answer_list = [fill_str] * missing + answer_list

    return answer_list


def solution2answer(solution: str, math_mode="eval_peeking", return_origin=False, num_answers=1) -> tuple[bool, list | str]:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution, return_origin, num_answers)
    else:
        raise ValueError("Invalid math_mode: {}".format(math_mode))
    return answer


def answer_tag_reward_fn_for_r1(model_output: str, ground_truths, problem=None, points=None, use_xverify=False, base_url=None):
    extracted_pred = model_output
    is_matched = False

    num_questions_to_answer = len(ground_truths)

    extracted_answers = solution2answer(
        str(model_output), num_answers=num_questions_to_answer)
    ground_truths = [solution2answer(str(gt), return_origin=True)[
        0] for gt in ground_truths]

    if not any(extracted_answers):
        return 0.0, 0.0, extracted_answers, ground_truths, ["not_scored"] * num_questions_to_answer
    is_matched = True

    total_score = 0.0
    extracted_preds, extracted_gts, scored_by_list = [], [], []
    score_list = []
    for extracted_pred, ground_truth in zip(extracted_answers, ground_truths):
        score, score_by, extracted_pred, extracted_gt = grade(
            extracted_pred, ground_truth, is_matched, problem, use_xverify=use_xverify, base_url=base_url)
        score_list.append(score)
        scored_by_list.append(score_by)
        extracted_preds.append(extracted_pred)
        extracted_gts.append(extracted_gt)

    total_score = sum(score_list) / num_questions_to_answer

    if points is None or len(points) == 0:
        points = [1.0] * num_questions_to_answer
    if len(points) == num_questions_to_answer:
        try:
            point = sum([s * p for s, p in zip(score_list, points)])
        except:
            point = 0
    else:
        point = score

    return total_score, point, extracted_preds, extracted_gts, scored_by_list


# def compute_score_p1(model_output, label, points=None, question="", use_xverify=False, base_url=None):
#     # todo: add use_xverify as a config
#     ground_truths = label
#     if isinstance(ground_truths, str):
#         ground_truths = [ground_truths]
#     # print(f"[DEBUG] points: {points}, ground_truths: {ground_truths}")
#     score, point, extracted_pred, extracted_gt, scored_by = answer_tag_reward_fn_for_r1(model_output, ground_truths, question, points, use_xverify, base_url)
#     #print(f"[DEBUG] Result score: {score}, point: {point}, score_by: {scored_by}")
#     # return

#     return {
#         "score": score,
#         "point": point,
#         "acc": abs(score - 1.0) < 1e-5,
#         "extracted_gt": str(extracted_gt),
#         "extracted_pred": str(extracted_pred),
#         "scored_by": str(scored_by)
#     }

def compute_score_p1(model_output, label, points=None, question="", use_xverify=False, base_url=None):
    if "</think>" in model_output:
        model_solution = model_output.split("</think>")[1]
    else:
        model_solution = model_output

    num_questions_to_answer = len(label)

    extracted_answers = solution2answer(
        str(model_solution), num_answers=num_questions_to_answer)
    ground_truths = [solution2answer(str(gt), return_origin=True)[
        0] for gt in label]
    score_list = []
    scored_by_list = []
    model_args = Model_args()
    model_args.base_url = base_url
    for extracted_pred, ground_truth in zip(extracted_answers, ground_truths):
        if extracted_pred is None:
            extracted_pred = ""
        if ground_truth is None:
            ground_truth = ""
        split_pred = extracted_pred.split(
            "=")[-1] if "=" in extracted_pred else extracted_pred
        split_gt = ground_truth.split(
            "=")[-1] if "=" in ground_truth else ground_truth

        scored_by = "wrong"
        is_correct = grade_answer_mathd(extracted_pred, ground_truth)[
            0] or grade_answer_mathd(split_pred, split_gt)[0]
        if not is_correct:
            if not _should_skip_math_verify(extracted_pred, ground_truth):
                is_correct = grade_answer_sympy(
                    extracted_pred, ground_truth)[0]
            if not is_correct and not _should_skip_math_verify(split_pred, split_gt):
                is_correct = grade_answer_sympy(split_pred, split_gt)[0]
        if is_correct:
            scored_by = "rule"

        # if not is_correct:
        #     is_correct, math_pred, math_gt = grade_answer_math_verify(extracted_pred, ground_truth)

        #     if not is_correct:
        #         if isinstance(math_pred, list) and isinstance(math_gt, list) and len(math_pred) > 0 and len(math_gt) > 0:
        #             if (isinstance(math_pred[0], sympy.core.numbers.Integer) or isinstance(math_pred[0], sympy.core.numbers.Float) or (isinstance(math_pred[0], sympy.core.numbers.Rational))) and isinstance(math_gt[0], sympy.sets.sets.Interval):
        #                 is_correct = 1.0 if math_gt[0].contains(math_pred[0]) == True else 0.0

        #         is_correct = is_correct or (grade_answer_math_verify(split_pred, split_gt)[0])

        if is_correct:
            scored_by = "math_verify"

        if not is_correct and use_xverify:
            is_correct = grade_answer_xverify(
                extracted_pred, ground_truth, question, model_args)
            if (not is_correct) and ((split_pred, split_gt) != (extracted_pred, ground_truth)):
                is_correct = grade_answer_xverify(
                    split_pred, split_gt, question, model_args)
            if is_correct:
                scored_by = "xverify"
        scored_by_list.append(scored_by)
        score_list.append(is_correct)

    total_score = sum(score_list) / num_questions_to_answer
    # if points is None or len(points) == 0:
    #     points = [1.0] * num_questions_to_answer
    if points is not None and len(points) > 0 and len(points) == num_questions_to_answer:
        point = sum([s * p for s, p in zip(score_list, points)])
    else:
        point = total_score

    return {
        "score": total_score,
        "point": point,
        "acc": abs(total_score - 1.0) < 1e-5,
        "extracted_gt": ground_truths,
        "extracted_pred": extracted_answers,
        "scored_by": scored_by_list
    }


def test_grade_fault_tolerance():
    """Test function to demonstrate fault tolerance in grade function"""
    print("Testing grade function fault tolerance...")

    # Test normal case
    try:
        result = grade("2", "2", True)
        print("Normal case result: {}".format(result))
        assert result[0] == True  # Should be correct
        assert result[1] in ["mathd", "sympy_verify",
                             "math_verify"]  # Should have a valid score_by
    except Exception as e:
        print("Normal case failed: {}".format(e))

    # Test with potentially problematic inputs
    test_cases = [
        ("", "", True),  # Empty strings
        ("invalid_math_expr^{^{", "2", True),  # Invalid LaTeX
        ("2", "invalid_math_expr^{^{", True),  # Invalid ground truth
        (None, "2", True),  # None input (will be converted to string)
        ("2", None, True),  # None ground truth
    ]

    for i, (model_ans, gt_ans, matched) in enumerate(test_cases):
        try:
            result = grade(model_ans, gt_ans, matched)
            print("Test case {}: model='{}', gt='{}' -> {}".format(i +
                  1, model_ans, gt_ans, result))
            # Should always return a 4-tuple even on error
            assert len(result) == 4
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
        except Exception as e:
            print("Test case {} failed: {}".format(i+1, e))


def test_grade_threading():
    """Test function to demonstrate thread-safety of grade function"""
    import concurrent.futures
    import random

    print("\nTesting grade function in multiple threads...")

    # Test cases for threading
    test_cases = [
        ("2", "2", True),
        ("3.14", "π", True),
        ("1/2", "0.5", True),
        ("x^2", "invalid", True),  # Should fail gracefully
        ("", "", True),  # Edge case
    ]

    def worker_func(thread_id, test_case):
        """Worker function for thread testing"""
        model_ans, gt_ans, matched = test_case
        try:
            result = grade(model_ans, gt_ans, matched)
            return "Thread {}: {} -> Success: {}".format(thread_id, test_case, result[0])
        except Exception as e:
            return "Thread {}: {} -> Error: {}".format(thread_id, test_case, str(e))

    # Run tests in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10):  # Run 10 parallel tasks
            test_case = random.choice(test_cases)
            future = executor.submit(worker_func, i, test_case)
            futures.append(future)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                # 10 second timeout per thread
                result = future.result(timeout=10)
                print(result)
            except Exception as e:
                print("Thread failed: {}".format(e))

    print("Threading test completed.")


if __name__ == "__main__":
    # Test fault tolerance first
    # test_grade_fault_tolerance()
    # print("\n" + "="*30 + "\n")

    # # Test threading
    # test_grade_threading()
    # print("\n" + "="*30 + "\n")

    # Example usage
    model_output = r"The answer is <answer>$\boxed{\dfrac{5}{3}}$</answer>"
    ground_truth = ["$[1.64,1.70]$"]
    question = "What is the answer to the ultimate question of life, the universe, and everything?"

    result = compute_score_p1(
        model_output, ground_truth, question=question, points=[1.0], use_xverify=False)
    # Should print a dictionary with score and correctness information.
    print(result)
