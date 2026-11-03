"""Deterministic answer checker (no LLM judge).

verify(text, task, customer_names) finds the last `FINAL ANSWER:` line in the
agent's output and scores it against task["answer"] by task["answer_type"]:
  number   correct within +-0.01; currency symbols, thousands separators and
           dates in the answer are ignored; number words zero..twenty accepted
  string   customer name: full name matches, or the surname alone when no
           other customer has it; naming any other customer as well is wrong
  order_id normalized exact match (ORD-10234 / ord 10234 / 10234)
  yes_no   yes/no plus common variants; both or neither is wrong
"""
import re
import sqlite3
import unicodedata

FINAL_RE = re.compile(r"^[\s*_#>`-]*final\s+answer\s*[*_]*\s*[:：]\s*(.*)$", re.I | re.M)

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
MONTHS = ("january february march april may june july august september october november "
          "december jan feb mar apr jun jul aug sep sept oct nov dec")
DATE_RES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?({MONTHS.replace(' ', '|')})\b\.?(\s+\d{{4}})?", re.I),
    re.compile(rf"\b({MONTHS.replace(' ', '|')})\.?\s+\d{{1,2}}(st|nd|rd|th)?\b(,?\s+\d{{4}})?", re.I),
    re.compile(rf"\b({MONTHS.replace(' ', '|')})\s+\d{{4}}\b", re.I),
    re.compile(r"\b20\d{2}\b"),
]
YES = {"yes", "y", "yeah", "yep", "true", "correct", "affirmative"}
NO = {"no", "n", "nope", "false", "negative", "none"}


def extract_final(text):
    """Return the text after the last FINAL ANSWER: marker, or None."""
    matches = FINAL_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def words(s):
    """Lowercase, strip accents, punctuation and currency; return a list of word tokens."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[€$£]|\beur(os?)?\b", " ", s)
    return re.findall(r"[a-z0-9]+", s)


def parse_number(ans):
    s = ans
    for r in DATE_RES:
        s = r.sub(" ", s)
    m = re.search(r"-?\d[\d.,\s]*\d|-?\d", s)
    if m:
        tok = re.sub(r"\s", "", m.group())
        if "," in tok and "." in tok:  # last separator is the decimal one
            dec = "." if tok.rfind(".") > tok.rfind(",") else ","
            tok = tok.replace("," if dec == "." else ".", "").replace(dec, ".")
        elif "," in tok:
            tok = tok.replace(",", ".") if re.fullmatch(r"-?\d+,\d{1,2}", tok) else tok.replace(",", "")
        try:
            return float(tok)
        except ValueError:
            pass
    for w in words(s):
        if w in NUMBER_WORDS:
            return float(NUMBER_WORDS[w])
    return None


def _contains(tokens, phrase):
    n = len(phrase)
    return any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


def check_name(ans, expected, customer_names):
    tokens = words(ans)
    exp = words(expected)
    others = [words(n) for n in customer_names if words(n) != exp]
    if any(_contains(tokens, o) for o in others):
        return False
    if _contains(tokens, exp):
        return True
    surname = exp[1:]
    surname_unique = not any(o[1:] == surname for o in others)
    other_surnames = {tuple(o[1:]) for o in others if tuple(o[1:]) != tuple(surname)}
    if any(_contains(tokens, list(s)) for s in other_surnames):
        return False
    return surname_unique and _contains(tokens, surname)


def check_order_id(ans, expected):
    exp = re.sub(r"\D", "", expected)
    ids = set(re.findall(r"ord\W*(\d{5})", ans, re.I)) or set(re.findall(r"\b(\d{5})\b", ans))
    return ids == {exp}


def check_yes_no(ans, expected):
    tokens = words(ans)
    if tokens and tokens[0] in YES | NO:
        got = "yes" if tokens[0] in YES else "no"
    else:
        has_yes, has_no = bool(set(tokens) & YES), bool(set(tokens) & NO)
        if has_yes == has_no:
            return False
        got = "yes" if has_yes else "no"
    return got == expected


def verify(text, task, customer_names=()):
    """Score agent output. Returns {"success", "final_answer", "reason"}."""
    ans = extract_final(text)
    if ans is None:
        return {"success": False, "final_answer": None, "reason": "no FINAL ANSWER line"}
    if not ans:
        return {"success": False, "final_answer": ans, "reason": "empty FINAL ANSWER"}
    kind, expected = task["answer_type"], task["answer"]
    if kind == "number":
        got = parse_number(ans)
        ok = got is not None and abs(got - float(expected)) <= 0.01 + 1e-9
    elif kind == "string":
        ok = check_name(ans, expected, customer_names)
    elif kind == "order_id":
        ok = check_order_id(ans, expected)
    elif kind == "yes_no":
        ok = check_yes_no(ans, expected)
    else:
        raise ValueError(f"unknown answer_type {kind!r}")
    return {"success": ok, "final_answer": ans, "reason": "correct" if ok else "wrong answer"}


def load_customer_names(db_path):
    con = sqlite3.connect(db_path)
    names = [r[0] for r in con.execute("SELECT full_name FROM customers")]
    con.close()
    return names
