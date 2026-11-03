"""Direct tests of every tool in every condition: schemas, responses, errors."""
import json
import re

import pytest

from tools import CONDITIONS, load
from tools import condition_a as A
from tools import condition_b as B
from tools import condition_c as C
from tools import condition_d as D

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

VALID_CALLS = {
    "A": [("list_customers", {}), ("get_customer", {"id": 1}), ("list_orders", {"user": 1}),
          ("list_transactions", {"id": "ORD-10109"}), ("read_logs", {"date": "2026-05-02"}),
          ("list_tickets", {"user": 1})],
    "B": [("list_customers", {}), ("get_customer", {"customer_id": 1}),
          ("list_orders", {"customer_id": 1}), ("list_transactions", {"order_number": "ORD-10109"}),
          ("read_logs", {"date": "2026-05-02"}), ("list_tickets", {"customer_id": 1})],
    "C": [("list_customers", {}), ("list_customers", {"name_contains": "jensen"}),
          ("list_customers", {"city": "Berlin", "response_format": "detailed"}),
          ("get_customer", {"customer_id": 1}), ("list_orders", {"customer_id": 1}),
          ("list_orders", {"customer_id": 1, "month": "2026-03", "response_format": "detailed"}),
          ("list_transactions", {"order_number": "ORD-10109"}),
          ("read_logs", {"date": "2026-05-02"}),
          ("read_logs", {"date": "2026-05-02", "contains": "timeout", "level": "ERROR",
                         "service": "payment-gateway"}),
          ("read_logs", {"date": "2026-05-02", "page": 2}), ("list_tickets", {"customer_id": 1})],
    "D": [("crm_search_customers", {"name_query": "Jensen"}),
          ("crm_get_customer_context", {"customer_name": "Hugo Jensen"}),
          ("crm_get_customer_context", {"customer_name": "hugo jensen", "month": "2026-02",
                                        "response_format": "detailed"}),
          ("logs_search", {"query": "E-PAY-502"}),
          ("logs_search", {"query": "timeout", "date": "2026-05-02", "level": "ERROR"})],
}


def call(cond, name, args):
    return load(cond).dispatch(name, args)


# --- schemas ---------------------------------------------------------------------

@pytest.mark.parametrize("cond", CONDITIONS)
def test_schema_format(cond):
    mod = load(cond)
    names = [t["function"]["name"] for t in mod.TOOLS]
    assert len(names) == len(set(names))
    for t in mod.TOOLS:
        assert t["type"] == "function"
        f = t["function"]
        assert f["description"] and f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])
        for p in f["parameters"]["properties"].values():
            assert p["type"] in {"string", "integer"}


def test_every_tool_has_a_valid_call():
    for cond in CONDITIONS:
        names = {t["function"]["name"] for t in load(cond).TOOLS}
        assert names == {n for n, _ in VALID_CALLS[cond]}, cond


def test_a_b_c_share_tool_names_and_d_is_namespaced():
    names = {c: {t["function"]["name"] for t in load(c).TOOLS} for c in CONDITIONS}
    assert names["A"] == names["B"] == names["C"]
    assert len(names["D"]) == 3
    assert all(n.startswith(("crm_", "logs_")) for n in names["D"])
    assert not any("duplicate" in n for n in names["D"])


def test_a_is_terse_and_ambiguous_b_is_not():
    for t in A.TOOLS:
        assert len(t["function"]["description"]) <= 30
    a_params = {p for t in A.TOOLS for p in t["function"]["parameters"]["properties"]}
    b_params = {p for t in B.TOOLS for p in t["function"]["parameters"]["properties"]}
    assert {"id", "user"} <= a_params
    assert not b_params & {"id", "user"}
    for t in B.TOOLS:
        assert "Example:" in t["function"]["description"]


# --- valid calls -----------------------------------------------------------------

@pytest.mark.parametrize("cond,name,args", [(c, n, a) for c in CONDITIONS for n, a in VALID_CALLS[c]])
def test_valid_call(cond, name, args):
    r = call(cond, name, args)
    assert r["error"] is None, r
    json.loads(r["content"])


def test_b_responses_identical_to_a():
    for (na, aa), (nb, ab) in zip(VALID_CALLS["A"], VALID_CALLS["B"]):
        assert na == nb
        assert call("A", na, aa) == call("B", nb, ab)


def test_a_returns_raw_rows_with_uuids():
    customers = json.loads(call("A", "list_customers", {})["content"])
    assert len(customers) == 60 and "uuid" in customers[0] and "crm_source" in customers[0]
    logs = json.loads(call("A", "read_logs", {"date": "2026-05-02"})["content"])
    assert len(logs) == 35 and UUID_RE.search(json.dumps(logs))


@pytest.mark.parametrize("cond,name,args", [(c, n, a) for c in "CD" for n, a in VALID_CALLS[c]])
def test_c_d_concise_responses_have_no_uuids(cond, name, args):
    assert not UUID_RE.search(call(cond, name, args)["content"])


def test_c_paginates_with_narrowing_note():
    r = json.loads(call("C", "list_customers", {})["content"])
    assert len(r["customers"]) == 20 and r["total"] == 60
    assert "name_contains" in r["note"] and "page=2" in r["note"]
    r = json.loads(call("C", "read_logs", {"date": "2026-05-02", "page": 2})["content"])
    assert len(r["lines"]) == 15 and "contains=" in r["note"] and "page=3" not in r["note"]


def test_c_response_format_adds_fields():
    concise = json.loads(call("C", "get_customer", {"customer_id": 1})["content"])
    detailed = json.loads(call("C", "get_customer", {"customer_id": 1,
                                                     "response_format": "detailed"})["content"])
    assert set(concise) < set(detailed)


def test_d_logs_search_maps_names_and_marks_matches():
    r = json.loads(call("D", "logs_search", {"query": "E-PAY-502", "date": "2026-04-14"})["content"])
    assert r["matches"] == 1
    hits = [x for x in r["lines"] if x.startswith(">>")]
    assert len(hits) == 1 and "[Ana Jensen]" in hits[0]
    assert len(r["lines"]) <= 5


def test_d_logs_search_truncates():
    r = json.loads(call("D", "logs_search", {"query": "INFO"})["content"])
    assert r["matches"] == "20+" and "date" in r["note"]


def test_d_context_payment_summary():
    r = json.loads(call("D", "crm_get_customer_context", {"customer_name": "Hugo Jensen"})["content"])
    o = {x["order_number"]: x for x in r["orders"]}
    assert o["ORD-10109"]["payments"] == "2 succeeded, 2 failed"
    r = json.loads(call("D", "crm_get_customer_context", {"customer_name": "Hannah Tanaka"})["content"])
    assert {x["status"] for x in r["refunds"]} == {"pending", "completed"}


# --- errors ----------------------------------------------------------------------

@pytest.mark.parametrize("cond", "AB")
def test_a_b_errors_are_bare_codes(cond):
    cid = "id" if cond == "A" else "customer_id"
    assert call(cond, "get_customer", {cid: 999}) == {"content": "ERR_404", "error": "invalid_args"}
    assert call(cond, "get_customer", {cid: "Ana Jensen"}) == {"content": "ERR_400", "error": "invalid_args"}
    assert call(cond, "get_customer", {}) == {"content": "ERR_400", "error": "invalid_args"}
    assert call(cond, "read_logs", {"date": "May 2"}) == {"content": "ERR_400", "error": "invalid_args"}
    assert call(cond, "list_customers", {"name": "x"}) == {"content": "ERR_400", "error": "invalid_args"}
    assert call(cond, "search_customers", {}) == {"content": "ERR_404", "error": "invalid_tool"}


def test_c_errors_are_actionable():
    r = call("C", "get_customer", {"customer_id": 999})
    assert r["error"] == "invalid_args" and "list_customers(name_contains=" in r["content"]
    r = call("C", "get_customer", {"customer_id": "Ana Jensen"})
    assert r["error"] == "invalid_args" and "list_customers" in r["content"]
    r = call("C", "list_transactions", {"order_number": "ORD-99999"})
    assert "list_orders" in r["content"]
    r = call("C", "list_transactions", {"order_number": "10109"})
    assert r["error"] == "invalid_args" and "ORD-10234" in r["content"]
    r = call("C", "read_logs", {"date": "May 2"})
    assert "YYYY-MM-DD" in r["content"]
    r = call("C", "list_customers", {"name": "x"})
    assert "name_contains" in r["content"]
    r = call("C", "search_customers", {})
    assert r["error"] == "invalid_tool" and "list_customers" in r["content"]


def test_c_no_match_suggests_names():
    r = json.loads(call("C", "list_customers", {"name_contains": "Olga Fisher"})["content"])
    assert r["customers"] == [] and "Olga Fischer" in r["note"]


def test_d_errors_are_actionable():
    r = call("D", "crm_get_customer_context", {"customer_name": "Jensen"})
    assert r["error"] == "invalid_args" and "Ana Jensen" in r["content"] and "Hugo Jensen" in r["content"]
    r = call("D", "crm_get_customer_context", {"customer_name": "Olga Fisher"})
    assert "Olga Fischer" in r["content"] and "crm_search_customers" in r["content"]
    r = call("D", "logs_search", {"query": "x", "date": "2026-5-2"})
    assert r["error"] == "invalid_args" and "YYYY-MM-DD" in r["content"]
    r = call("D", "list_customers", {})
    assert r["error"] == "invalid_tool" and "crm_search_customers" in r["content"]


def test_d_unique_partial_name_resolves():
    r = json.loads(call("D", "crm_get_customer_context", {"customer_name": "Laila"})["content"])
    assert r["customer"]["name"] == "Laila Wagner"


@pytest.mark.parametrize("cond,style", [("A", "ERR_500"), ("C", "Error: Internal error")])
def test_exceptions_are_caught(monkeypatch, cond, style):
    import tools.common as common

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(common, "rows", boom)
    mod = load(cond)
    monkeypatch.setattr(mod, "rows", boom)
    name, args = VALID_CALLS[cond][1]
    r = mod.dispatch(name, args)
    assert r["error"] == "exception" and r["content"].startswith(style)


def test_argument_coercion_and_json_string_args():
    assert call("B", "get_customer", {"customer_id": "1"})["error"] is None
    assert call("B", "get_customer", '{"customer_id": 1}')["error"] is None
    assert call("C", "list_orders", {"customer_id": 1, "month": None})["error"] is None
    assert call("B", "get_customer", {"customer_id": True})["error"] == "invalid_args"
