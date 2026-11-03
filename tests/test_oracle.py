"""Oracle test: every task is solvable in every condition.

A scripted solution per task type per condition uses only that condition's
dispatch() and parses only what the tool returns (plus the task prompt), then
the answer is scored by the real verifier.
"""
import json
import re

import pytest

from agent.verifier import load_customer_names, verify
from tasks.make_tasks import DB, OUT
from tools import CONDITIONS, load

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June"], start=1)}
TASKS = [json.loads(line) for line in OUT.read_text().splitlines()]
NAMES = load_customer_names(DB)


class Tools:
    def __init__(self, cond):
        self.mod, self.calls = load(cond), []

    def __call__(self, name, **args):
        r = self.mod.dispatch(name, args)
        assert r["error"] is None, (name, args, r)
        self.calls.append(name)
        return json.loads(r["content"])


def prompt_name(prompt):
    found = [n for n in NAMES if n in prompt]
    assert len(found) == 1
    return found[0]


def prompt_date(prompt):
    m = re.search(r"(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", prompt)
    return f"{m.group(3)}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"


def prompt_month(prompt):
    m = re.search(r"in (\w+) (\d{4})", prompt)
    return f"{m.group(2)}-{MONTHS[m.group(1)]:02d}"


# --- A and B: raw API wrappers ----------------------------------------------------

def oracle_ab(task, t, p):
    """p maps the role of a parameter to its name in this condition."""
    kind, prompt = task["type"], task["prompt"]
    customers = t("list_customers")
    if kind in {"log_to_customer", "incident_impact"}:
        logs = t("read_logs", date=prompt_date(prompt))
        if kind == "log_to_customer":
            uuids = {x["user_uuid"] for x in logs if "E-PAY-502" in x["message"]}
            assert len(uuids) == 1
            return next(c["full_name"] for c in customers if c["uuid"] in uuids)
        return len({x["user_uuid"] for x in logs
                    if x["service"] == "payment-gateway" and "timeout" in x["message"].lower()})
    cid = next(c["id"] for c in customers if c["full_name"] == prompt_name(prompt))
    orders = t("list_orders", **{p["customer"]: cid})
    if kind == "spend_lookup":
        ym = prompt_month(prompt)
        return round(sum(o["total_eur"] for o in orders
                         if o["order_date"].startswith(ym) and o["status"] != "cancelled"), 2)
    txns = {o["order_number"]: t("list_transactions", **{p["order"]: o["order_number"]})
            for o in orders}
    if kind == "duplicate_charge":
        dup = [n for n, xs in txns.items()
               if sum(x["type"] == "charge" and x["status"] == "succeeded" for x in xs) > 1]
        assert len(dup) == 1
        return dup[0]
    if kind == "refund_status":
        pending = any(x["type"] == "refund" and x["status"] == "pending"
                      for xs in txns.values() for x in xs)
        return "yes" if pending else "no"
    raise AssertionError(kind)


# --- C: same tools, better responses ----------------------------------------------

def oracle_c(task, t):
    kind, prompt = task["type"], task["prompt"]
    if kind == "log_to_customer":
        r = t("read_logs", date=prompt_date(prompt), contains="E-PAY-502")
        who = {x["customer"] for x in r["lines"]}
        assert len(who) == 1 and "note" not in r
        return who.pop().rsplit(" (id", 1)[0]
    if kind == "incident_impact":
        r = t("read_logs", date=prompt_date(prompt), contains="timeout", service="payment-gateway")
        assert "note" not in r
        return len({x["customer"] for x in r["lines"]})
    found = t("list_customers", name_contains=prompt_name(prompt))["customers"]
    assert len(found) == 1
    cid = found[0]["id"]
    if kind == "spend_lookup":
        r = t("list_orders", customer_id=cid, month=prompt_month(prompt))
        assert "note" not in r
        return round(sum(o["total_eur"] for o in r["orders"] if o["status"] != "cancelled"), 2)
    r = t("list_orders", customer_id=cid)
    assert "note" not in r
    txns = {o["order_number"]: t("list_transactions", order_number=o["order_number"])["transactions"]
            for o in r["orders"]}
    if kind == "duplicate_charge":
        dup = [n for n, xs in txns.items()
               if sum(x["type"] == "charge" and x["status"] == "succeeded" for x in xs) > 1]
        assert len(dup) == 1
        return dup[0]
    if kind == "refund_status":
        return "yes" if any(x["type"] == "refund" and x["status"] == "pending"
                            for xs in txns.values() for x in xs) else "no"
    raise AssertionError(kind)


# --- D: consolidated tools ----------------------------------------------------------

def oracle_d(task, t):
    kind, prompt = task["type"], task["prompt"]
    if kind in {"log_to_customer", "incident_impact"}:
        query = "E-PAY-502" if kind == "log_to_customer" else "payment-gateway timeout"
        r = t("logs_search", query=query, date=prompt_date(prompt))
        assert "note" not in r
        who = {re.search(r"\[(.+?)\]", x).group(1) for x in r["lines"] if x.startswith(">>")}
        if kind == "log_to_customer":
            assert len(who) == 1
            return who.pop()
        return len(who)
    if kind == "spend_lookup":
        r = t("crm_get_customer_context", customer_name=prompt_name(prompt), month=prompt_month(prompt))
        return round(sum(o["total_eur"] for o in r["orders"] if o["status"] != "cancelled"), 2)
    r = t("crm_get_customer_context", customer_name=prompt_name(prompt))
    if kind == "duplicate_charge":
        dup = [o["order_number"] for o in r["orders"]
               if int(re.match(r"(\d+) succeeded", o["payments"]).group(1)) > 1]
        assert len(dup) == 1
        return dup[0]
    if kind == "refund_status":
        return "yes" if any(x["status"] == "pending" for x in r["refunds"]) else "no"
    raise AssertionError(kind)


def solve(cond, task):
    t = Tools(cond)
    if cond == "A":
        ans = oracle_ab(task, t, {"customer": "user", "order": "id"})
    elif cond == "B":
        ans = oracle_ab(task, t, {"customer": "customer_id", "order": "order_number"})
    elif cond == "C":
        ans = oracle_c(task, t)
    else:
        ans = oracle_d(task, t)
    return ans, t.calls


@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_task_solvable(cond, task):
    ans, calls = solve(cond, task)
    r = verify(f"FINAL ANSWER: {ans}", task, NAMES)
    assert r["success"], (cond, task["id"], ans, task["answer"])
    # The oracle's path uses the expected tools (diagnostic field sanity check).
    assert set(task["expected_tools"][cond]) <= set(calls), (cond, task["id"], calls)


def test_d_needs_fewer_calls_than_a():
    for task in TASKS:
        _, calls_a = solve("A", task)
        _, calls_d = solve("D", task)
        assert len(calls_d) < len(calls_a) or len(calls_a) == 1, task["id"]
