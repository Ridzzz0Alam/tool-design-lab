"""Build the 20 evaluation tasks from the generated shop data (deterministic).

Reads env/shop.sqlite and env/planted.json, writes tasks/tasks.jsonl.
Prompts name people and dates in plain language, never internal IDs.

Usage: .venv/bin/python tasks/make_tasks.py
"""
import json
import random
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "env" / "shop.sqlite"
MANIFEST = ROOT / "env" / "planted.json"
OUT = ROOT / "tasks" / "tasks.jsonl"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Diagnostic only, never used for scoring. B uses A's tools; C uses them too unless listed
# (C's read_logs already shows customer names, so no list_customers lookup is needed).
EXPECTED_TOOLS = {
    "duplicate_charge": {"A": ["list_customers", "list_orders", "list_transactions"],
                         "D": ["crm_get_customer_context"]},
    "spend_lookup": {"A": ["list_customers", "list_orders"], "D": ["crm_get_customer_context"]},
    "log_to_customer": {"A": ["read_logs", "list_customers"], "C": ["read_logs"],
                        "D": ["logs_search"]},
    "incident_impact": {"A": ["read_logs"], "D": ["logs_search"]},
    "refund_status": {"A": ["list_customers", "list_orders", "list_transactions"],
                      "D": ["crm_get_customer_context"]},
}


def plain_date(iso):
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def expected(task_type):
    e = EXPECTED_TOOLS[task_type]
    return {"A": e["A"], "B": e["A"], "C": e.get("C", e["A"]), "D": e["D"]}


def spend_candidates(con, manifest):
    """Customer-months whose spend is unambiguous: >=2 orders, all delivered, no refunds,
    and the customer also ordered in other months (so the month filter matters)."""
    excluded = {p["customer_id"] for p in manifest["duplicate_charges"] + manifest["refunds"]}
    orders = con.execute("SELECT customer_id, substr(order_date,1,7) ym, status, total_eur "
                         "FROM orders ORDER BY order_number").fetchall()
    months = defaultdict(lambda: defaultdict(list))
    for cid, ym, status, total in orders:
        months[cid][ym].append((status, total))
    out = []
    for cid in sorted(months):
        if cid in excluded or len(months[cid]) < 2:
            continue
        for ym in sorted(months[cid]):
            rows = months[cid][ym]
            if len(rows) >= 2 and all(s == "delivered" for s, _ in rows):
                out.append((cid, ym, round(sum(t for _, t in rows), 2), len(rows)))
    return out


def build(db=DB, manifest_path=MANIFEST, seed=42):
    rng = random.Random(seed)
    m = json.loads(Path(manifest_path).read_text())
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, full_name FROM customers"))
    by_type = {}

    by_type["duplicate_charge"] = [{
        "prompt": f"{p['customer_name']} says they were charged more than once for a single "
                  "order. Which order number was it?",
        "answer": p["order_number"], "answer_type": "order_id",
    } for p in m["duplicate_charges"]]

    cands = spend_candidates(con, m)
    rng.shuffle(cands)
    picked, used_c, used_m = [], set(), set()
    for cid, ym, total, n in cands:  # distinct customers and months
        if cid not in used_c and ym not in used_m:
            picked.append((cid, ym, total, n))
            used_c.add(cid)
            used_m.add(ym)
        if len(picked) == 4:
            break
    by_type["spend_lookup"] = [{
        "prompt": f"How much did {names[cid]} spend in total in "
                  f"{MONTHS[int(ym[5:]) - 1]} {ym[:4]}, in euros?",
        "answer": total, "answer_type": "number",
    } for cid, ym, total, n in sorted(picked, key=lambda x: x[1])]

    by_type["log_to_customer"] = [{
        "prompt": f"Someone hit checkout error E-PAY-502 on {plain_date(p['date'])}. "
                  "Which customer was it?",
        "answer": p["customer_name"], "answer_type": "string",
    } for p in m["epay502_errors"]]

    by_type["incident_impact"] = [{
        "prompt": "How many different customers were affected by the payment-gateway timeouts "
                  f"on {plain_date(p['date'])}?",
        "answer": p["n_distinct_customers"], "answer_type": "number",
    } for p in m["gateway_timeouts"]]

    # 2 yes (always including the customer with both completed and pending), 2 no.
    refunds = m["refunds"]
    both = [p for p in refunds if len({r["status"] for r in p["refunds"]}) == 2]
    yes_pool = [p for p in refunds if p["has_pending"] and p not in both]
    no_pool = [p for p in refunds if not p["has_pending"]]
    chosen = both[:1] + rng.sample(yes_pool, 1) + rng.sample(no_pool, 2)
    by_type["refund_status"] = [{
        "prompt": f"Does {p['customer_name']} have a refund that is still pending? "
                  "Answer yes or no.",
        "answer": "yes" if p["has_pending"] else "no", "answer_type": "yes_no",
    } for p in sorted(chosen, key=lambda p: p["customer_name"])]

    tasks = []
    for type_no, (task_type, items) in enumerate(by_type.items(), start=1):
        assert len(items) == 4, (task_type, len(items))
        dev = set() if task_type == "refund_status" else set(rng.sample(range(4), 2))
        for i, t in enumerate(items):
            tasks.append({
                "id": f"{task_type}_{i + 1}", "type": task_type, "type_no": type_no,
                "split": "dev" if i in dev else "test", **t,
                "expected_tools": expected(task_type),
            })
    con.close()
    return tasks


def main():
    tasks = build()
    OUT.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"Wrote {len(tasks)} tasks to {OUT}")
    for t in tasks:
        print(f"  [{t['split']:4}] {t['id']:18} {t['prompt']}  ->  {t['answer']}")


if __name__ == "__main__":
    main()
