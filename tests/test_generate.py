"""Tests for env/generate.py: counts, determinism, integrity, and planted incidents."""
import json
import re
import sqlite3
from collections import Counter, defaultdict

import pytest

from env.generate import (EPAY502_DATES, EPAY502_MSG, TIMEOUT_DATES, TIMEOUT_MSG,
                          generate)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    manifest = generate(d / "shop.sqlite", d / "planted.json", seed=42)
    con = sqlite3.connect(d / "shop.sqlite")
    con.row_factory = sqlite3.Row
    yield con, manifest, d
    con.close()


def q(con, sql, *args):
    return [dict(r) for r in con.execute(sql, args)]


# --- counts, determinism, integrity -------------------------------------------

def test_counts(built):
    con, m, _ = built
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["customers", "orders", "transactions", "logs", "tickets"]}
    assert counts == m["counts"]
    assert counts["customers"] == 60
    assert 280 <= counts["orders"] <= 320
    assert 450 <= counts["transactions"] <= 550
    assert 1900 <= counts["logs"] <= 2100
    assert 35 <= counts["tickets"] <= 45


def test_deterministic(built, tmp_path):
    con, m, d = built
    m2 = generate(tmp_path / "shop.sqlite", tmp_path / "planted.json", seed=42)
    con2 = sqlite3.connect(tmp_path / "shop.sqlite")
    assert list(con.iterdump()) == list(con2.iterdump())
    assert (d / "planted.json").read_text() == (tmp_path / "planted.json").read_text()
    assert m == m2
    con2.close()


def test_different_seed_differs(built, tmp_path):
    _, m, _ = built
    assert generate(tmp_path / "s.sqlite", tmp_path / "p.json", seed=7) != m


def test_customers(built):
    con, _, _ = built
    rows = q(con, "SELECT * FROM customers")
    assert len({r["full_name"] for r in rows}) == 60
    assert len({r["uuid"] for r in rows}) == 60
    assert [r["id"] for r in rows] == list(range(1, 61))
    surnames = Counter(r["full_name"].split(" ", 1)[1] for r in rows)
    assert any(n > 1 for n in surnames.values()), "need shared surnames"
    assert any(n == 1 for n in surnames.values()), "need unique surnames"
    for r in rows:
        assert re.fullmatch(r"[0-9a-f-]{36}", r["uuid"])
        assert "@example." in r["email"]


def test_orders(built):
    con, _, _ = built
    for o in q(con, "SELECT * FROM orders"):
        assert re.fullmatch(r"ORD-\d{5}", o["order_number"])
        assert "2026-01-01" <= o["order_date"] <= "2026-06-30"
        assert o["created_at"].startswith(o["order_date"])
        assert o["total_eur"] > 0 and round(o["total_eur"], 2) == o["total_eur"]
    assert q(con, "SELECT COUNT(*) n FROM orders o LEFT JOIN customers c ON c.id=o.customer_id "
                  "WHERE c.id IS NULL")[0]["n"] == 0


def test_charges_consistent_with_order_status(built):
    con, _, _ = built
    ok = {r["order_number"]: r["n"] for r in q(
        con, "SELECT order_number, COUNT(*) n FROM transactions "
             "WHERE type='charge' AND status='succeeded' GROUP BY order_number")}
    for o in q(con, "SELECT order_number, status FROM orders"):
        if o["status"] == "cancelled":
            assert o["order_number"] not in ok
        else:
            assert ok.get(o["order_number"], 0) >= 1
    assert {r["type"] for r in q(con, "SELECT DISTINCT type FROM transactions")} == {"charge", "refund"}
    assert q(con, "SELECT COUNT(*) n FROM transactions WHERE type='charge' AND status='failed'")[0]["n"] > 50


def test_log_users_and_timestamps(built):
    con, _, _ = built
    assert q(con, "SELECT COUNT(*) n FROM logs l LEFT JOIN customers c ON c.uuid=l.user_uuid "
                  "WHERE l.user_uuid IS NOT NULL AND c.uuid IS NULL")[0]["n"] == 0
    ts = [r["ts"] for r in q(con, "SELECT ts FROM logs ORDER BY id")]
    assert ts == sorted(ts)
    assert ts[0] >= "2026-01-01" and ts[-1] < "2026-07-01"


# --- planted incidents ---------------------------------------------------------

def test_duplicate_charges_exactly_as_planted(built):
    con, m, _ = built
    rows = q(con, "SELECT o.order_number, o.customer_id, COUNT(*) n FROM transactions t "
                  "JOIN orders o USING(order_number) "
                  "WHERE t.type='charge' AND t.status='succeeded' "
                  "GROUP BY o.order_number HAVING n > 1")
    found = {(r["customer_id"], r["order_number"], r["n"]) for r in rows}
    planted = {(p["customer_id"], p["order_number"], p["succeeded_charges"])
               for p in m["duplicate_charges"]}
    assert found == planted
    assert len(planted) == 4 and len({p[0] for p in planted}) == 4
    for p in m["duplicate_charges"]:  # each victim has other orders, so the agent must pick the right one
        n = q(con, "SELECT COUNT(*) n FROM orders WHERE customer_id=?", p["customer_id"])[0]["n"]
        assert n >= 2
        amounts = {r["amount_eur"] for r in q(
            con, "SELECT amount_eur FROM transactions WHERE order_number=? AND status='succeeded'",
            p["order_number"])}
        assert amounts == {p["amount_eur"]}


def test_epay502_exactly_as_planted(built):
    con, m, _ = built
    rows = q(con, "SELECT substr(l.ts,1,10) d, c.id cid, COUNT(*) n FROM logs l "
                  "JOIN customers c ON c.uuid=l.user_uuid "
                  "WHERE l.message LIKE '%E-PAY-502%' GROUP BY d, cid")
    found = {(r["d"], r["cid"], r["n"]) for r in rows}
    planted = {(p["date"], p["customer_id"], p["n_lines"]) for p in m["epay502_errors"]}
    assert found == planted
    assert sorted(p["date"] for p in m["epay502_errors"]) == sorted(EPAY502_DATES)
    assert q(con, "SELECT COUNT(*) n FROM logs WHERE message LIKE '%E-PAY-502%' "
                  "AND (user_uuid IS NULL OR message != ?)", EPAY502_MSG)[0]["n"] == 0


def test_epay502_has_lookalike_distractors(built):
    con, _, _ = built
    assert q(con, "SELECT COUNT(*) n FROM logs WHERE message LIKE '%E-PAY-402%'")[0]["n"] > 0


def test_gateway_timeouts_exactly_as_planted(built):
    con, m, _ = built
    assert q(con, "SELECT COUNT(*) n FROM logs WHERE lower(message) LIKE '%timeout%' "
                  "AND message != ?", TIMEOUT_MSG)[0]["n"] == 0
    rows = q(con, "SELECT substr(l.ts,1,10) d, c.id cid FROM logs l "
                  "JOIN customers c ON c.uuid=l.user_uuid WHERE l.message=?", TIMEOUT_MSG)
    by_date = defaultdict(Counter)
    for r in rows:
        by_date[r["d"]][r["cid"]] += 1
    assert set(by_date) == set(TIMEOUT_DATES)
    for p in m["gateway_timeouts"]:
        got = by_date[p["date"]]
        assert len(got) == p["n_distinct_customers"] == TIMEOUT_DATES[p["date"]]
        assert dict(got) == {e["customer_id"]: e["n_lines"] for e in p["customers"]}
        assert sum(got.values()) == p["n_lines"] > p["n_distinct_customers"], \
            "line count must exceed distinct customers so the counting trap exists"
    # Every timeout line has a user (no anonymous lines to miscount).
    assert q(con, "SELECT COUNT(*) n FROM logs WHERE message=? AND user_uuid IS NULL",
             TIMEOUT_MSG)[0]["n"] == 0


def test_refunds_exactly_as_planted(built):
    con, m, _ = built
    rows = q(con, "SELECT o.customer_id cid, t.order_number, t.status FROM transactions t "
                  "JOIN orders o USING(order_number) WHERE t.type='refund'")
    found = defaultdict(set)
    for r in rows:
        found[r["cid"]].add((r["order_number"], r["status"]))
    planted = {p["customer_id"]: {(x["order_number"], x["status"]) for x in p["refunds"]}
               for p in m["refunds"]}
    assert dict(found) == planted
    assert {r["status"] for r in rows} == {"pending", "completed"}
    pending = {p["customer_id"] for p in m["refunds"] if p["has_pending"]}
    completed_only = {p["customer_id"] for p in m["refunds"] if not p["has_pending"]}
    assert len(pending) >= 2 and len(completed_only) >= 2
    assert any(len({s for _, s in v}) == 2 for v in planted.values()), \
        "need a customer with both a completed and a pending refund"
    dup_customers = {p["customer_id"] for p in m["duplicate_charges"]}
    assert not dup_customers & set(planted)


def test_manifest_names_match_db(built):
    con, m, _ = built
    names = {r["id"]: r["full_name"] for r in q(con, "SELECT id, full_name FROM customers")}
    entries = (m["duplicate_charges"] + m["epay502_errors"] + m["refunds"]
               + [e for t in m["gateway_timeouts"] for e in t["customers"]])
    for e in entries:
        assert names[e["customer_id"]] == e["customer_name"]


def test_tickets(built):
    con, m, _ = built
    rows = q(con, "SELECT * FROM tickets")
    assert all(re.fullmatch(r"TCK-\d{4}", r["ticket_id"]) for r in rows)
    reasons = " ".join(r["reason"] for r in rows)
    assert "charged more than once" in reasons and "refund" in reasons


def test_committed_files_match_generator(built):
    """env/shop.sqlite and env/planted.json in the repo are what seed 42 produces."""
    from env.generate import HERE
    _, m, _ = built
    assert json.loads((HERE / "planted.json").read_text()) == m
