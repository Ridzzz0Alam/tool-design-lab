"""Tests for tasks/make_tasks.py and tasks/tasks.jsonl.

Ground truth is recomputed here with independent SQL over env/shop.sqlite
(not via planted.json), so a bug in the manifest would show up as a mismatch.
"""
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime

import pytest

from agent.verifier import load_customer_names, verify
from tasks.make_tasks import DB, OUT, build

TYPES = ["duplicate_charge", "spend_lookup", "log_to_customer", "incident_impact", "refund_status"]
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June"], start=1)}


@pytest.fixture(scope="module")
def tasks():
    return [json.loads(line) for line in OUT.read_text().splitlines()]


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(DB)
    yield c
    c.close()


def plain_to_iso(text):
    m = re.search(r"(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", text)
    return f"{m.group(3)}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"


def name_in(prompt, con):
    names = [r[0] for r in con.execute("SELECT full_name FROM customers")]
    found = [n for n in names if n in prompt]
    assert len(found) == 1, (prompt, found)
    return found[0]


# --- structure -----------------------------------------------------------------

def test_file_matches_generator(tasks):
    assert build() == tasks


def test_structure(tasks):
    assert len(tasks) == 20
    assert len({t["id"] for t in tasks}) == 20
    assert Counter(t["type"] for t in tasks) == {k: 4 for k in TYPES}
    for t in tasks:
        assert set(t) == {"id", "type", "type_no", "split", "prompt", "answer", "answer_type",
                          "expected_tools"}
        assert t["answer_type"] in {"string", "number", "yes_no", "order_id"}
        assert set(t["expected_tools"]) == {"A", "B", "C", "D"}


def test_split(tasks):
    dev = [t for t in tasks if t["split"] == "dev"]
    test = [t for t in tasks if t["split"] == "test"]
    assert len(dev) == 8 and len(test) == 12
    assert Counter(t["type"] for t in dev) == {k: 2 for k in TYPES[:4]}
    assert all(t["split"] == "test" for t in tasks if t["type"] == "refund_status")


def test_prompts_use_plain_language(tasks, con):
    uuids = [r[0] for r in con.execute("SELECT uuid FROM customers")]
    for t in tasks:
        p = t["prompt"]
        assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", p), p      # no UUIDs
        assert not re.search(r"\bORD-|\bTXN-|\bTCK-", p), p          # no internal record ids
        assert not re.search(r"\d{4}-\d{2}-\d{2}", p), p             # no ISO dates
        assert not re.search(r"\b(id|uuid|customer_id)\b", p, re.I), p
        assert not any(u in p for u in uuids)
        assert not re.search(r"\b(he|she|his|her)\b", p, re.I), p    # no gendered pronouns


def test_refund_answers_balanced(tasks):
    assert Counter(t["answer"] for t in tasks if t["type"] == "refund_status") == {"yes": 2, "no": 2}


# --- independent ground truth ----------------------------------------------------

def truth(t, con):
    p = t["prompt"]
    if t["type"] == "duplicate_charge":
        rows = con.execute(
            "SELECT o.order_number FROM transactions x JOIN orders o USING(order_number) "
            "JOIN customers c ON c.id=o.customer_id WHERE c.full_name=? AND x.type='charge' "
            "AND x.status='succeeded' GROUP BY o.order_number HAVING COUNT(*)>1",
            (name_in(p, con),)).fetchall()
        assert len(rows) == 1
        return rows[0][0]
    if t["type"] == "spend_lookup":
        m = re.search(r"in (\w+) (\d{4})", p)
        ym = f"{m.group(2)}-{MONTHS[m.group(1)]:02d}"
        rows = con.execute(
            "SELECT o.total_eur, o.status FROM orders o JOIN customers c ON c.id=o.customer_id "
            "WHERE c.full_name=? AND substr(o.order_date,1,7)=?", (name_in(p, con), ym)).fetchall()
        assert len(rows) >= 2 and all(s == "delivered" for _, s in rows)
        return round(sum(r[0] for r in rows), 2)
    if t["type"] == "log_to_customer":
        rows = con.execute(
            "SELECT DISTINCT c.full_name FROM logs l JOIN customers c ON c.uuid=l.user_uuid "
            "WHERE l.message LIKE '%E-PAY-502%' AND substr(l.ts,1,10)=?",
            (plain_to_iso(p),)).fetchall()
        assert len(rows) == 1
        return rows[0][0]
    if t["type"] == "incident_impact":
        return con.execute(
            "SELECT COUNT(DISTINCT user_uuid) FROM logs WHERE service='payment-gateway' "
            "AND lower(message) LIKE '%timeout%' AND substr(ts,1,10)=?",
            (plain_to_iso(p),)).fetchone()[0]
    if t["type"] == "refund_status":
        n = con.execute(
            "SELECT COUNT(*) FROM transactions x JOIN orders o USING(order_number) "
            "JOIN customers c ON c.id=o.customer_id WHERE c.full_name=? AND x.type='refund' "
            "AND x.status='pending'", (name_in(p, con),)).fetchone()[0]
        return "yes" if n else "no"
    raise AssertionError(t["type"])


def test_answers_match_independent_sql(tasks, con):
    for t in tasks:
        assert truth(t, con) == t["answer"], t["id"]


def test_spend_month_filter_matters(tasks, con):
    """The customer also ordered in other months, so summing all orders is wrong."""
    for t in (t for t in tasks if t["type"] == "spend_lookup"):
        total_all = con.execute(
            "SELECT ROUND(SUM(total_eur),2) FROM orders o JOIN customers c ON c.id=o.customer_id "
            "WHERE c.full_name=?", (name_in(t["prompt"], con),)).fetchone()[0]
        assert abs(total_all - t["answer"]) > 1


def test_refund_no_answers_have_completed_refunds(tasks, con):
    """'No' tasks are the trap: the customer has a refund, just not a pending one."""
    for t in (t for t in tasks if t["type"] == "refund_status" and t["answer"] == "no"):
        n = con.execute(
            "SELECT COUNT(*) FROM transactions x JOIN orders o USING(order_number) "
            "JOIN customers c ON c.id=o.customer_id WHERE c.full_name=? AND x.type='refund'",
            (name_in(t["prompt"], con),)).fetchone()[0]
        assert n >= 1


def test_impact_dates_have_more_lines_than_customers(tasks, con):
    for t in (t for t in tasks if t["type"] == "incident_impact"):
        lines = con.execute(
            "SELECT COUNT(*) FROM logs WHERE lower(message) LIKE '%timeout%' AND substr(ts,1,10)=?",
            (plain_to_iso(t["prompt"]),)).fetchone()[0]
        assert lines > t["answer"]


def test_plain_dates_are_real(tasks):
    for t in tasks:
        if t["type"] in {"log_to_customer", "incident_impact"}:
            datetime.fromisoformat(plain_to_iso(t["prompt"]))


# --- verifier against tasks ------------------------------------------------------

def test_verifier_accepts_own_answers(tasks):
    names = load_customer_names(DB)
    for t in tasks:
        assert verify(f"Done.\nFINAL ANSWER: {t['answer']}", t, names)["success"], t["id"]


def test_verifier_rejects_other_tasks_answers(tasks):
    names = load_customer_names(DB)
    for t in tasks:
        for other in tasks:
            if other["type"] == t["type"] and other["answer"] != t["answer"]:
                r = verify(f"FINAL ANSWER: {other['answer']}", t, names)
                assert not r["success"], (t["id"], other["answer"])
