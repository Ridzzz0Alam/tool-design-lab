"""Condition D: consolidated, purpose-built, namespaced tools (with B and C improvements).

crm_search_customers, crm_get_customer_context and logs_search replace the six
API wrappers. They stay general-purpose: no tool answers a task type directly.
"""
from tools.common import (LOG_RANGE, actionable_error, bad_args, dumps, make_dispatch,
                          match_customers, norm, not_found, rows, suggest_names, tool)

MAX_ORDERS = 20
MAX_MATCHES = 20
CONTEXT = 2
FORMAT = {"type": "string", "enum": ["concise", "detailed"],
          "description": "'concise' (default) or 'detailed' for extra fields."}


def _resolve(customer_name):
    """Exact full name (case/accent-insensitive), else a unique partial match."""
    q = norm(customer_name)
    exact = [c for c in rows("SELECT * FROM customers") if norm(c["full_name"]) == q]
    if exact:
        return exact[0]
    found = match_customers(customer_name)
    if len(found) == 1:
        return found[0]
    if found:
        names = ", ".join(c["full_name"] for c in found[:10])
        raise bad_args(f"'{customer_name}' matches {len(found)} customers: {names}. "
                       "Pass the full name.")
    sugg = suggest_names(customer_name)
    raise not_found(f"No customer named '{customer_name}'."
                    + (f" Did you mean: {', '.join(sugg)}?" if sugg else "")
                    + " Use crm_search_customers with part of the name.")


def crm_search_customers(name_query, response_format="concise"):
    found = match_customers(name_query)
    if not found:
        sugg = suggest_names(name_query)
        return dumps({"customers": [], "note": f"No customer name contains '{name_query}'."
                      + (f" Did you mean: {', '.join(sugg)}?" if sugg else "")
                      + " Try a shorter part of the name, e.g. only the surname."})
    counts = {r["customer_id"]: r["n"] for r in rows(
        "SELECT customer_id, COUNT(*) n FROM orders GROUP BY customer_id")}
    out = []
    for c in found[:MAX_ORDERS]:
        item = {"name": c["full_name"], "city": c["city"], "orders": counts.get(c["id"], 0)}
        if response_format == "detailed":
            item.update(email=c["email"], customer_since=c["signup_date"])
        out.append(item)
    obj = {"customers": out}
    if len(found) > MAX_ORDERS:
        obj["note"] = f"Showing {MAX_ORDERS} of {len(found)}; use a longer part of the name."
    return dumps(obj)


def _payment_summary(txns):
    charges = [t for t in txns if t["type"] == "charge"]
    parts = []
    for status in ("succeeded", "failed"):
        n = sum(t["status"] == status for t in charges)
        if n:
            parts.append(f"{n} {status}")
    return ", ".join(parts) or "no charges"


def crm_get_customer_context(customer_name, month=None, response_format="concise"):
    c = _resolve(customer_name)
    orders = rows("SELECT * FROM orders WHERE customer_id=? ORDER BY order_number", c["id"])
    txns = rows("SELECT t.* FROM transactions t JOIN orders o USING(order_number) "
                "WHERE o.customer_id=? ORDER BY t.created_at", c["id"])
    by_order = {}
    for t in txns:
        by_order.setdefault(t["order_number"], []).append(t)

    shown = [o for o in orders if o["order_date"].startswith(month)] if month else orders
    notes = []
    if len(shown) > MAX_ORDERS:
        notes.append(f"Showing the {MAX_ORDERS} most recent of {len(shown)} orders; pass "
                     "month='YYYY-MM' for a specific month.")
        shown = shown[-MAX_ORDERS:]
    if month and not shown:
        notes.append(f"No orders in {month}.")
    out_orders = []
    for o in shown:
        item = {"order_number": o["order_number"], "date": o["order_date"],
                "total_eur": o["total_eur"], "status": o["status"],
                "payments": _payment_summary(by_order.get(o["order_number"], []))}
        if response_format == "detailed":
            item["transactions"] = [
                {"time": t["created_at"].replace("T", " "), "type": t["type"],
                 "status": t["status"], "amount_eur": t["amount_eur"]}
                for t in by_order.get(o["order_number"], [])]
        out_orders.append(item)

    refunds = [{"order_number": t["order_number"], "date": t["created_at"][:10],
                "amount_eur": t["amount_eur"], "status": t["status"]}
               for t in txns if t["type"] == "refund"]
    tickets = [{"date": t["created_date"], "reason": t["reason"], "status": t["status"]}
               for t in rows("SELECT * FROM tickets WHERE customer_id=? ORDER BY ticket_id", c["id"])]
    profile = {"name": c["full_name"], "city": c["city"], "email": c["email"],
               "customer_since": c["signup_date"], "total_orders": len(orders)}
    obj = {"customer": profile, "orders": out_orders, "refunds": refunds, "tickets": tickets}
    if notes:
        obj["note"] = " ".join(notes)
    return dumps(obj)


def logs_search(query, date=None, level=None, response_format="concise"):
    if date and not LOG_RANGE[0] <= date <= LOG_RANGE[1]:
        return dumps({"matches": 0, "lines": [],
                      "note": f"No logs on {date}. Logs cover {LOG_RANGE[0]} to {LOG_RANGE[1]}."})
    words = query.lower().split()
    if not words:
        raise bad_args("query must contain at least one word, e.g. 'E-PAY-502' or 'timeout'.")
    sql = ("SELECT l.*, c.full_name FROM logs l LEFT JOIN customers c ON c.uuid=l.user_uuid "
           + ("WHERE substr(l.ts,1,10)=? " if date else "") + "ORDER BY l.id")
    lines = rows(sql, *([date] if date else []))
    hits = [i for i, x in enumerate(lines)
            if (not level or x["level"] == level)
            and all(w in f"{x['service']} {x['level']} {x['message']}".lower() for w in words)]
    if not hits:
        where = f" on {date}" if date else ""
        return dumps({"matches": 0, "lines": [],
                      "note": f"No log lines match '{query}'{where}. Try fewer or shorter words"
                              + (", or drop the date or level filter." if date or level else ".")})
    note = None
    if len(hits) > MAX_MATCHES:
        note = (f"Showing the first {MAX_MATCHES} of {len(hits)} matches. Narrow with date "
                "(YYYY-MM-DD), level, or more words in the query.")
        hits = hits[:MAX_MATCHES]
    hit_set, keep = set(hits), set()
    for i in hits:
        keep.update(range(max(0, i - CONTEXT), min(len(lines), i + CONTEXT + 1)))
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("...")
        x = lines[i]
        who = x["full_name"] or "system"
        line = (f"{'>>' if i in hit_set else '  '} {x['ts'].replace('T', ' ')} {x['service']} "
                f"{x['level']} [{who}] {x['message']}")
        if response_format == "detailed":
            line += f" (log_id {x['id']})"
        out.append(line)
        prev = i
    obj = {"matches": len(hit_set) if not note else f"{MAX_MATCHES}+", "lines": out}
    if note:
        obj["note"] = note
    return dumps(obj)


TOOLS = [
    tool("crm_search_customers",
         "Search customers by part of their name; returns full name, city and number of orders. "
         "Use it when you only have part of a name or are unsure of the spelling. Other crm_ "
         "tools take the full name. Example: crm_search_customers(name_query='Shah')",
         {"name_query": {"type": "string",
                         "description": "Part of the name, case-insensitive, e.g. 'Shah'."},
          "response_format": FORMAT}, ["name_query"]),
    tool("crm_get_customer_context",
         "Everything support needs about one customer in one call: profile; orders "
         "(order_number, date, total_eur, status, and a payment summary like '1 succeeded, "
         "1 failed'; a paid order normally has exactly 1 succeeded charge, 'failed' = declined "
         "attempt with no money taken, cancelled orders were never paid); refunds with status "
         "('pending' = not yet returned, 'completed' = returned); support tickets. "
         "response_format='detailed' lists every transaction. "
         "Example: crm_get_customer_context(customer_name='Priya Shah', month='2026-03')",
         {"customer_name": {"type": "string",
                            "description": "The customer's full name, e.g. 'Priya Shah'."},
          "month": {"type": "string", "pattern": r"^\d{4}-\d{2}$",
                    "description": "Optional: only orders in this month, YYYY-MM, e.g. 2026-03."},
          "response_format": FORMAT}, ["customer_name"]),
    tool("logs_search",
         "Search application logs (2026-01-01 to 2026-06-30). Returns lines containing every "
         "word of query (in service, level or message; case-insensitive) with 2 lines of context "
         "before and after. Matching lines start with '>>'; customers are shown by name in "
         "[brackets]. Error codes look like E-PAY-502. "
         "Example: logs_search(query='E-PAY-502', date='2026-04-14')",
         {"query": {"type": "string",
                    "description": "Words to find, e.g. 'E-PAY-502' or 'payment-gateway timeout'."},
          "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$",
                   "description": "Optional day, YYYY-MM-DD, e.g. 2026-05-02."},
          "level": {"type": "string", "enum": ["INFO", "WARN", "ERROR"],
                    "description": "Optional log level."},
          "response_format": FORMAT}, ["query"]),
]

dispatch = make_dispatch(TOOLS, {
    "crm_search_customers": crm_search_customers,
    "crm_get_customer_context": crm_get_customer_context,
    "logs_search": logs_search,
}, actionable_error)
