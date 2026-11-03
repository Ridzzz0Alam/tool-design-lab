"""Unit tests for agent/verifier.py."""
import pytest

from agent.verifier import extract_final, parse_number, verify

NAMES = ["Ana Jensen", "Hugo Jensen", "Laila Wagner", "Olga Fischer", "Mei Fisher",
         "Samuel Kovac", "Tariq Kovac", "Priya Ahmed", "Grace Nowak", "Leo Novak"]


def task(answer, answer_type):
    return {"answer": answer, "answer_type": answer_type}


def ok(text, answer, answer_type):
    return verify(text, task(answer, answer_type), NAMES)["success"]


# --- extraction ----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I checked.\nFINAL ANSWER: 5", "5"),
    ("FINAL ANSWER: 3\nwait\nFINAL ANSWER: 5", "5"),
    ("**FINAL ANSWER:** ORD-10426", "** ORD-10426"),
    ("final answer: yes", "yes"),
    ("  - Final Answer : no", "no"),
    ("The answer is 5.", None),
    ("", None),
    (None, None),
])
def test_extract_final(text, expected):
    assert extract_final(text) == expected


def test_no_final_line_fails():
    r = verify("The order was ORD-10426.", task("ORD-10426", "order_id"), NAMES)
    assert r == {"success": False, "final_answer": None, "reason": "no FINAL ANSWER line"}


def test_empty_final_fails():
    assert not ok("FINAL ANSWER:", "5", "number")


# --- numbers -------------------------------------------------------------------

@pytest.mark.parametrize("ans", [
    "577.53", "€577.53", "577.53 EUR", "577.53 euros", "EUR 577.53", "577,53 €",
    "€ 577.53", "Total: 577.53", "577.525", "577.54", "**577.53**",
])
def test_number_correct(ans):
    assert ok(f"FINAL ANSWER: {ans}", 577.53, "number")


@pytest.mark.parametrize("ans", ["577.55", "577", "57.75", "unknown", "about six hundred"])
def test_number_wrong(ans):
    assert not ok(f"FINAL ANSWER: {ans}", 577.53, "number")


@pytest.mark.parametrize("ans,val", [
    ("1,234.56", 1234.56), ("1.234,56", 1234.56), ("1,234", 1234), ("12,5", 12.5),
    ("five", 5), ("5 customers", 5), ("Five customers were affected.", 5),
    ("5 customers on 2 May 2026", 5), ("On 2 May 2026, 5 customers", 5),
    ("On May 2, 2026 there were 5", 5), ("2026-05-02: 5 customers", 5),
    ("In March 2026 they spent €347.52", 347.52),
])
def test_parse_number(ans, val):
    assert parse_number(ans) == pytest.approx(val)


def test_count_answers():
    assert ok("FINAL ANSWER: 5", 5, "number")
    assert ok("FINAL ANSWER: 5 different customers", 5, "number")
    assert not ok("FINAL ANSWER: 11", 5, "number")  # counted log lines, not customers


# --- names ---------------------------------------------------------------------

@pytest.mark.parametrize("ans", [
    "Ana Jensen", "ana jensen", "ANA JENSEN.", "The customer was Ana Jensen.", "**Ana Jensen**",
    "Ana  Jensen",
])
def test_name_full_match(ans):
    assert ok(f"FINAL ANSWER: {ans}", "Ana Jensen", "string")


def test_name_reversed_order_not_accepted():
    # "Jensen, Ana" has no "ana jensen" sequence and Jensen is shared -> rejected.
    assert not ok("FINAL ANSWER: Jensen, Ana", "Ana Jensen", "string")


def test_surname_only_when_unique():
    assert ok("FINAL ANSWER: Wagner", "Laila Wagner", "string")
    assert ok("FINAL ANSWER: Ms Wagner", "Laila Wagner", "string")
    assert not ok("FINAL ANSWER: Jensen", "Ana Jensen", "string")  # Hugo Jensen too
    assert not ok("FINAL ANSWER: Kovac", "Samuel Kovac", "string")


def test_lookalike_surnames_are_distinct():
    assert ok("FINAL ANSWER: Fischer", "Olga Fischer", "string")
    assert not ok("FINAL ANSWER: Fisher", "Olga Fischer", "string")
    assert not ok("FINAL ANSWER: Mei Fisher", "Olga Fischer", "string")


def test_naming_another_customer_too_is_wrong():
    assert not ok("FINAL ANSWER: Ana Jensen or Hugo Jensen", "Ana Jensen", "string")
    assert not ok("FINAL ANSWER: Wagner or Fischer", "Laila Wagner", "string")


def test_wrong_name():
    assert not ok("FINAL ANSWER: Hugo Jensen", "Ana Jensen", "string")
    assert not ok("FINAL ANSWER: I could not find the customer", "Ana Jensen", "string")
    assert not ok("FINAL ANSWER: user 3f2a9c1e-7b8d-4e2f-9a1b-2c3d4e5f6a7b", "Ana Jensen", "string")


def test_name_with_accents():
    assert ok("FINAL ANSWER: Samuel Kováč", "Samuel Kovac", "string")


# --- order ids -----------------------------------------------------------------

@pytest.mark.parametrize("ans", [
    "ORD-10426", "ord-10426", "ORD 10426", "ORD10426", "Order ORD-10426.", "10426", "**ORD-10426**",
    "The duplicate charge was on order ORD-10426 (charged twice).",
])
def test_order_id_correct(ans):
    assert ok(f"FINAL ANSWER: {ans}", "ORD-10426", "order_id")


@pytest.mark.parametrize("ans", [
    "ORD-10427", "ORD-1042", "ORD-10426 and ORD-10109", "none", "TXN-500123",
])
def test_order_id_wrong(ans):
    assert not ok(f"FINAL ANSWER: {ans}", "ORD-10426", "order_id")


# --- yes/no --------------------------------------------------------------------

@pytest.mark.parametrize("ans,expected", [
    ("yes", "yes"), ("Yes.", "yes"), ("YES", "yes"), ("Yes, one refund is pending.", "yes"),
    ("yep", "yes"), ("true", "yes"), ("no", "no"), ("No.", "no"), ("Nope", "no"),
    ("No, all refunds are completed.", "no"), ("**No**", "no"),
    ("There is no pending refund", "no"),
])
def test_yes_no_correct(ans, expected):
    assert ok(f"FINAL ANSWER: {ans}", expected, "yes_no")


@pytest.mark.parametrize("ans,expected", [
    ("yes", "no"), ("no", "yes"), ("maybe", "yes"), ("I am not sure", "no"),
    ("unclear: yes or no", "yes"),
])
def test_yes_no_wrong(ans, expected):
    assert not ok(f"FINAL ANSWER: {ans}", expected, "yes_no")


def test_unknown_answer_type():
    with pytest.raises(ValueError):
        verify("FINAL ANSWER: x", task("x", "date"), NAMES)
