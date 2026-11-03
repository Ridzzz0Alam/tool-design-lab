"""Generate the mock online shop's support backend (deterministic, seed 42).

Writes env/shop.sqlite and env/planted.json. The manifest lists every planted
incident, so task generation and tests share one source of truth.

Usage: .venv/bin/python env/generate.py [--seed 42] [--db PATH] [--manifest PATH]
"""
import argparse
import json
import random
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = date(2026, 1, 1), date(2026, 6, 30)

N_CUSTOMERS = 60
N_ORDERS = 300
N_LOGS = 2000
N_TICKETS = 40

FIRST_NAMES = [
    "Priya", "Tomas", "Lena", "Amara", "Jonas", "Sofia", "Mateo", "Elif", "Noah", "Ingrid",
    "Rahul", "Chloe", "Lukas", "Aisha", "Marco", "Freya", "Daniel", "Yuki", "Olga", "Samuel",
    "Nadia", "Pieter", "Grace", "Emil", "Zara", "Hugo", "Leila", "Oskar", "Maya", "Felix",
    "Ana", "Viktor", "Hannah", "Karim", "Julia", "Mikael", "Esther", "Diego", "Ines", "Arjun",
    "Clara", "Bruno", "Sara", "Henrik", "Lucia", "Tariq", "Eva", "Niels", "Mei", "Pablo",
    "Alice", "Stefan", "Laila", "Theo", "Rosa", "Adam", "Nora", "Ivan", "Fatima", "Leo",
]
# Includes near-duplicates (Shah/Sharma, Novak/Nowak, Fischer/Fisher) so name
# search and "did you mean" errors have something real to disambiguate.
LAST_NAMES = [
    "Shah", "Sharma", "Novak", "Nowak", "Fischer", "Fisher", "Okafor", "Berg", "Garcia", "Kaya",
    "Martin", "Lindqvist", "Patel", "Dubois", "Weber", "Rossi", "Jensen", "Tanaka", "Petrova",
    "Cohen", "Haddad", "de Vries", "Mensah", "Larsen", "Ali", "Moreau", "Rahman", "Andersen",
    "Silva", "Ivanova", "Becker", "Costa", "Nguyen", "Bauer", "Kowalski", "Moretti", "Eriksson",
    "Santos", "Ahmed", "Wagner", "Horvat", "Laurent", "Kim", "Romano", "Hansen", "Yilmaz",
    "Fernandez", "Schulz", "Murphy", "Kovac", "Brennan", "Varga",
]
CITIES = ["Berlin", "Amsterdam", "Lisbon", "Vienna", "Prague", "Dublin", "Madrid", "Milan",
          "Copenhagen", "Warsaw", "Brussels", "Helsinki", "Lyon", "Munich", "Rotterdam"]
EMAIL_DOMAINS = ["example.com", "example.org", "example.net"]

# Planted incidents. Dates are fixed so tasks can name them in plain language.
DUPLICATE_CHARGE_COUNTS = [2, 3, 2, 2]  # succeeded charges on the one affected order
EPAY502_DATES = ["2026-02-09", "2026-03-23", "2026-04-14", "2026-06-03"]
TIMEOUT_DATES = {"2026-01-27": 3, "2026-03-11": 4, "2026-05-02": 5, "2026-06-18": 6}
REFUND_PLAN = ["pending"] * 2 + ["both"] * 1 + ["completed"] * 3

EPAY502_MSG = "Checkout failed code=E-PAY-502: payment service returned bad gateway"
TIMEOUT_MSG = "Gateway timeout after 30000ms provider=paylink"

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, uuid TEXT UNIQUE NOT NULL, full_name TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL, city TEXT NOT NULL, signup_date TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL, marketing_opt_in INTEGER NOT NULL, crm_source TEXT NOT NULL
);
CREATE TABLE orders (
    order_number TEXT PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL, created_at TEXT NOT NULL, total_eur REAL NOT NULL,
    status TEXT NOT NULL, items_count INTEGER NOT NULL, shipping_method TEXT NOT NULL,
    currency TEXT NOT NULL, warehouse_code TEXT NOT NULL
);
CREATE TABLE transactions (
    txn_id TEXT PRIMARY KEY, order_number TEXT NOT NULL REFERENCES orders(order_number),
    created_at TEXT NOT NULL, type TEXT NOT NULL, amount_eur REAL NOT NULL,
    status TEXT NOT NULL, card_last4 TEXT NOT NULL, gateway_ref TEXT NOT NULL
);
CREATE TABLE logs (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, service TEXT NOT NULL, level TEXT NOT NULL,
    user_uuid TEXT REFERENCES customers(uuid), message TEXT NOT NULL
);
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    created_date TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL, channel TEXT NOT NULL
);
"""

GENERIC_TICKET_REASONS = [
    "My package arrived damaged, the box was crushed.",
    "I ordered the wrong size and would like to exchange it.",
    "How do I change the delivery address on my account?",
    "My discount code is not being accepted at checkout.",
    "Delivery is a week late and tracking has not updated.",
    "I want to delete my account and personal data.",
    "One item was missing from my parcel.",
    "Can I get an invoice with my company VAT number?",
    "The product colour is different from the photos.",
    "I can't log in, the password reset email never arrives.",
]


def _ts(d, t):
    return datetime.combine(d, t).isoformat(timespec="seconds")


def _rand_date(rng, start=START, end=END):
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def _rand_time(rng, h0=0, h1=23):
    return time(rng.randint(h0, h1), rng.randrange(60), rng.randrange(60))


def _uuid(rng):
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def make_customers(rng):
    firsts = FIRST_NAMES[:]
    rng.shuffle(firsts)
    customers = []
    for i, first in enumerate(firsts[:N_CUSTOMERS], start=1):
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        local = f"{first}.{last}".lower().replace(" ", "")
        customers.append({
            "id": i, "uuid": _uuid(rng), "full_name": name,
            "email": f"{local}@{rng.choice(EMAIL_DOMAINS)}",
            "city": rng.choice(CITIES),
            "signup_date": _rand_date(rng, date(2024, 1, 1), date(2025, 12, 31)).isoformat(),
            "loyalty_tier": rng.choice(["none", "bronze", "silver", "gold"]),
            "marketing_opt_in": rng.randint(0, 1),
            "crm_source": rng.choice(["web", "app", "import-2024", "partner"]),
        })
    return customers


def make_orders(rng, customers):
    weights = [rng.uniform(0.3, 2.0) for _ in customers]
    owners = rng.choices(customers, weights=weights, k=N_ORDERS)
    stamps = sorted((_rand_date(rng), _rand_time(rng, 7, 22), c["id"]) for c in owners)
    orders, num = [], 10001
    for d, t, cid in stamps:
        if rng.random() < 0.05:
            status = "cancelled"
        elif d >= date(2026, 6, 22):
            status = rng.choice(["processing", "shipped"])
        else:
            status = "delivered"
        orders.append({
            "order_number": f"ORD-{num}", "customer_id": cid, "order_date": d.isoformat(),
            "created_at": _ts(d, t), "total_eur": round(rng.uniform(8, 260), 2), "status": status,
            "items_count": rng.randint(1, 6),
            "shipping_method": rng.choice(["standard", "express", "pickup"]),
            "currency": "EUR", "warehouse_code": rng.choice(["WH-AMS-1", "WH-BER-2", "WH-MAD-1"]),
        })
        num += rng.randint(1, 3)
    return orders


def make_charges(rng, orders):
    """One succeeded charge per non-cancelled order, preceded by some failed attempts."""
    txns = []
    for o in orders:
        start = datetime.fromisoformat(o["created_at"])
        card = f"{rng.randrange(10000):04d}"
        r = rng.random()
        n_failed = 2 if r < 0.15 else 1 if r < 0.55 else 0
        if o["status"] == "cancelled":
            n_failed = max(n_failed, 1)
        at = start
        for _ in range(n_failed):
            at += timedelta(seconds=rng.randint(20, 300))
            txns.append(_txn(rng, o, at, "charge", o["total_eur"], "failed", card))
        if o["status"] != "cancelled":
            at += timedelta(seconds=rng.randint(20, 300))
            txns.append(_txn(rng, o, at, "charge", o["total_eur"], "succeeded", card))
    return txns


def _txn(rng, order, at, typ, amount, status, card):
    return {"order_number": order["order_number"], "created_at": at.isoformat(timespec="seconds"),
            "type": typ, "amount_eur": amount, "status": status, "card_last4": card,
            "gateway_ref": _uuid(rng)}


def plant_duplicates(rng, customers, orders, txns):
    by_cust = {}
    for o in orders:
        if o["status"] != "cancelled":
            by_cust.setdefault(o["customer_id"], []).append(o)
    eligible = sorted(cid for cid, os_ in by_cust.items() if len(os_) >= 2)
    chosen = rng.sample(eligible, len(DUPLICATE_CHARGE_COUNTS))
    planted = []
    for cid, n_charges in zip(chosen, DUPLICATE_CHARGE_COUNTS):
        order = rng.choice(by_cust[cid])
        first = next(t for t in txns
                     if t["order_number"] == order["order_number"] and t["status"] == "succeeded")
        at = datetime.fromisoformat(first["created_at"])
        for _ in range(n_charges - 1):
            at += timedelta(minutes=rng.randint(1, 90))
            txns.append(_txn(rng, order, at, "charge", order["total_eur"], "succeeded",
                             first["card_last4"]))
        planted.append({"customer_id": cid, "customer_name": customers[cid - 1]["full_name"],
                        "order_number": order["order_number"], "order_date": order["order_date"],
                        "succeeded_charges": n_charges, "amount_eur": order["total_eur"]})
    return planted


def plant_refunds(rng, customers, orders, txns, exclude):
    by_cust = {}
    for o in orders:
        if o["status"] == "delivered" and o["order_date"] <= "2026-06-01":
            by_cust.setdefault(o["customer_id"], []).append(o)
    eligible = sorted(cid for cid, os_ in by_cust.items() if len(os_) >= 2 and cid not in exclude)
    chosen = rng.sample(eligible, len(REFUND_PLAN))
    planted = []
    for cid, kind in zip(chosen, REFUND_PLAN):
        candidates = sorted(by_cust[cid], key=lambda o: o["order_date"])
        if kind == "both":
            statuses = ["completed", "pending"]
        elif kind == "completed":
            statuses = ["completed"] * rng.randint(1, 2)
        else:
            statuses = ["pending"]
        refunds = []
        for order, status in zip(rng.sample(candidates, len(statuses)), statuses):
            at = datetime.fromisoformat(order["created_at"]) + timedelta(days=rng.randint(3, 20),
                                                                         hours=rng.randint(0, 8))
            amount = order["total_eur"] if rng.random() < 0.6 else round(order["total_eur"] * 0.5, 2)
            card = next(t["card_last4"] for t in txns if t["order_number"] == order["order_number"])
            txns.append(_txn(rng, order, at, "refund", amount, status, card))
            refunds.append({"order_number": order["order_number"], "status": status,
                            "amount_eur": amount, "created_at": at.isoformat(timespec="seconds")})
        planted.append({"customer_id": cid, "customer_name": customers[cid - 1]["full_name"],
                        "has_pending": "pending" in statuses, "refunds": refunds})
    return planted


def make_logs(rng, customers, orders, txns):
    uuid_of = {c["id"]: c["uuid"] for c in customers}
    order_owner = {o["order_number"]: o["customer_id"] for o in orders}
    logs = []

    def add(ts, service, level, cid, message):
        logs.append({"ts": ts, "service": service, "level": level,
                     "user_uuid": uuid_of[cid] if cid else None, "message": message})

    # Activity derived from orders and transactions.
    for o in orders:
        add(o["created_at"], "checkout", "INFO", o["customer_id"],
            f"Order {o['order_number']} placed total={o['total_eur']:.2f} EUR")
    for t in txns:
        cid = order_owner[t["order_number"]]
        if t["type"] == "refund":
            add(t["created_at"], "payments", "INFO", cid,
                f"Refund requested order={t['order_number']} amount={t['amount_eur']:.2f} EUR")
        elif t["status"] == "succeeded":
            add(t["created_at"], "payment-gateway", "INFO", cid,
                f"Charge succeeded order={t['order_number']} amount={t['amount_eur']:.2f} EUR")
        else:
            add(t["created_at"], "payment-gateway", "WARN", cid,
                f"Card declined code=E-PAY-402 order={t['order_number']}")

    # Planted: E-PAY-502 checkout errors, one customer per date.
    epay = []
    for ds, c in zip(EPAY502_DATES, rng.sample(customers, len(EPAY502_DATES))):
        d = date.fromisoformat(ds)
        t0 = datetime.combine(d, _rand_time(rng, 9, 20))
        n = rng.randint(1, 2)
        for k in range(n):
            add((t0 + timedelta(seconds=40 * k)).isoformat(timespec="seconds"),
                "checkout", "ERROR", c["id"], EPAY502_MSG)
        epay.append({"date": ds, "customer_id": c["id"], "customer_name": c["full_name"],
                     "n_lines": n})

    # Planted: payment-gateway timeout bursts. Some customers hit it repeatedly,
    # so "how many different customers" differs from the number of lines.
    timeouts = []
    for ds, n_cust in TIMEOUT_DATES.items():
        d = date.fromisoformat(ds)
        picked = rng.sample(customers, n_cust + 2)
        affected, bystanders = picked[:n_cust], picked[n_cust:]
        entries = []
        for i, c in enumerate(affected):
            n = 2 if i == 0 else rng.randint(1, 3)
            for _ in range(n):
                add(_ts(d, _rand_time(rng, 14, 15)), "payment-gateway", "ERROR", c["id"], TIMEOUT_MSG)
            entries.append({"customer_id": c["id"], "customer_name": c["full_name"], "n_lines": n})
        for c in bystanders:
            add(_ts(d, _rand_time(rng, 13, 16)), "payment-gateway", "WARN", c["id"],
                f"Gateway latency {rng.randint(2000, 4000)}ms provider=paylink")
        add(_ts(d, time(13, 58, 0)), "payment-gateway", "WARN", None,
            "Provider paylink health check degraded")
        timeouts.append({"date": ds, "n_distinct_customers": n_cust,
                         "n_lines": sum(e["n_lines"] for e in entries), "customers": entries})

    # Background noise up to N_LOGS lines.
    templates = [
        ("web", "INFO", True, lambda: f"GET /products/{rng.randint(100, 999)} 200"),
        ("web", "INFO", True, lambda: f"GET /search?q={rng.choice(['shoes', 'lamp', 'mug', 'jacket'])} 200"),
        ("auth", "INFO", True, lambda: "User login succeeded"),
        ("auth", "WARN", True, lambda: "Login failed: wrong password"),
        ("inventory", "INFO", False, lambda: f"Stock reserved sku=SKU-{rng.randint(1000, 9999)}"),
        ("email", "INFO", True, lambda: "Sent newsletter"),
        ("checkout", "INFO", True, lambda: f"Cart updated items={rng.randint(1, 6)}"),
        ("checkout", "ERROR", True, lambda: "Checkout failed code=E-CART-409: cart version conflict"),
        ("web", "ERROR", False, lambda: "Upstream cache miss storm, serving stale content"),
    ]
    weights = [30, 15, 20, 5, 10, 8, 10, 2, 1]
    while len(logs) < N_LOGS:
        service, level, has_user, msg = rng.choices(templates, weights=weights)[0]
        cid = rng.choice(customers)["id"] if has_user else None
        add(_ts(_rand_date(rng), _rand_time(rng)), service, level, cid, msg())

    logs.sort(key=lambda r: r["ts"])  # stable: equal timestamps keep insertion order
    for i, r in enumerate(logs, start=1):
        r["id"] = i
    return logs, epay, timeouts


def make_tickets(rng, customers, duplicates, refunds):
    tickets = []

    def add(cid, d, reason, status):
        tickets.append({"customer_id": cid, "created_date": d, "reason": reason, "status": status,
                        "channel": rng.choice(["email", "chat", "phone"])})

    for p in duplicates[:2]:
        d = (date.fromisoformat(p["order_date"]) + timedelta(days=rng.randint(1, 5))).isoformat()
        add(p["customer_id"], d, "I think I was charged more than once for one order. Please check.",
            "open")
    for p in refunds[:3]:
        d = p["refunds"][-1]["created_at"][:10]
        add(p["customer_id"], d, "I returned an item, when will my refund arrive?", "pending")
    while len(tickets) < N_TICKETS:
        add(rng.choice(customers)["id"], _rand_date(rng).isoformat(),
            rng.choice(GENERIC_TICKET_REASONS), rng.choice(["open", "pending", "closed", "closed"]))
    tickets.sort(key=lambda t: (t["created_date"], t["customer_id"]))
    for i, t in enumerate(tickets, start=1001):
        t["ticket_id"] = f"TCK-{i}"
    return tickets


def generate(db_path=HERE / "shop.sqlite", manifest_path=HERE / "planted.json", seed=42):
    rng = random.Random(seed)
    customers = make_customers(rng)
    orders = make_orders(rng, customers)
    txns = make_charges(rng, orders)
    duplicates = plant_duplicates(rng, customers, orders, txns)
    refunds = plant_refunds(rng, customers, orders, txns,
                            exclude={p["customer_id"] for p in duplicates})
    txns.sort(key=lambda t: t["created_at"])
    for i, t in enumerate(txns, start=500001):
        t["txn_id"] = f"TXN-{i}"
    logs, epay, timeouts = make_logs(rng, customers, orders, txns)
    tickets = make_tickets(rng, customers, duplicates, refunds)

    db_path, manifest_path = Path(db_path), Path(manifest_path)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for table, rows in [("customers", customers), ("orders", orders), ("transactions", txns),
                        ("logs", logs), ("tickets", tickets)]:
        cols = list(rows[0])
        con.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [tuple(r[c] for c in cols) for r in rows])
    con.commit()
    con.close()

    manifest = {
        "seed": seed,
        "counts": {"customers": len(customers), "orders": len(orders), "transactions": len(txns),
                   "logs": len(logs), "tickets": len(tickets)},
        "duplicate_charges": duplicates,
        "epay502_errors": epay,
        "gateway_timeouts": timeouts,
        "refunds": refunds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default=HERE / "shop.sqlite")
    ap.add_argument("--manifest", default=HERE / "planted.json")
    args = ap.parse_args()
    m = generate(args.db, args.manifest, args.seed)
    print(f"Wrote {args.db} and {args.manifest}")
    print("Counts:", m["counts"])


if __name__ == "__main__":
    main()
