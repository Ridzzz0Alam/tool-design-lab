"""Condition C: B's six tools with better responses.

Human-readable names with simple numeric ids instead of UUIDs, low-signal
fields dropped, response_format ("concise" default / "detailed"), 20 rows per
page with a note on how to narrow, and specific, actionable errors.
"""
from tools.common import (LOG_RANGE, actionable_error, bad_args, customer_label, dumps,
                          make_dispatch, match_customers, not_found, rows, suggest_names, tool)

PAGE = 20
FORMAT = {"type": "string", "enum": ["concise", "detailed"],
          "description": "'concise' (default) or 'detailed' for extra fields."}
PAGE_P = {"type": "integer", "description": "Page number, default 1 (20 rows per page)."}
CUSTOMER_ID = {"type": "integer",
               "description": "Numeric customer id from list_customers, e.g. 12 (not the name). "
                              "To find it, call list_customers(name_contains='part of the name')."}
SERVICES = ["web", "auth", "checkout", "payment-gateway", "payments", "inventory", "email"]


def _page(items, page, what, narrow):
    total = len(items)
    pages = max(1, -(-total // PAGE))
    if page < 1 or page > pages:
        raise bad_args(f"page {page} does not exist; there {'is' if pages == 1 else 'are'} "
                       f"{pages} page(s) of {what}.")
    shown = items[(page - 1) * PAGE: page * PAGE]
    note = None
    if total > PAGE:
        first = (page - 1) * PAGE + 1
        note = (f"Showing {what} {first}-{first + len(shown) - 1} of {total}. {narrow}"
                + (f" Or use page={page + 1} for more." if page < pages else ""))
    return shown, note


def _with_note(obj, note):
    if note:
        obj["note"] = note
    return dumps(obj)


def _customer(customer_id):
    r = rows("SELECT * FROM customers WHERE id=?", customer_id)
    if not r:
        raise not_found(f"No customer with id {customer_id}. Customer ids are numbers from 1 to "
                        f"60. Call list_customers(name_contains='part of the name') to find the id.")
    return r[0]


def list_customers(name_contains=None, city=None, page=1, response_format="concise"):
    found = match_customers(name_contains) if name_contains else rows(
        "SELECT * FROM customers ORDER BY id")
    if city:
        found = [c for c in found if c["city"].lower() == city.lower()]
    if not found:
        hint = ""
        if name_contains:
            sugg = suggest_names(name_contains)
            hint = (f" Did you mean: {', '.join(sugg)}?" if sugg else "") + \
                " Try a shorter part of the name, e.g. only the surname."
        return dumps({"customers": [], "note": f"No customers match.{hint}"})
    shown, note = _page(found, page, "customers",
                        "Narrow with name_contains='part of a name'.")
    out = []
    for c in shown:
        item = {"id": c["id"], "name": c["full_name"], "city": c["city"]}
        if response_format == "detailed":
            item.update(email=c["email"], signup_date=c["signup_date"], loyalty_tier=c["loyalty_tier"])
        out.append(item)
    return _with_note({"customers": out, "total": len(found)}, note)


def get_customer(customer_id, response_format="concise"):
    c = _customer(customer_id)
    n = rows("SELECT COUNT(*) n FROM orders WHERE customer_id=?", customer_id)[0]["n"]
    out = {"id": c["id"], "name": c["full_name"], "email": c["email"], "city": c["city"],
           "signup_date": c["signup_date"], "orders": n}
    if response_format == "detailed":
        out.update(loyalty_tier=c["loyalty_tier"], marketing_opt_in=bool(c["marketing_opt_in"]))
    return dumps(out)


def list_orders(customer_id, month=None, page=1, response_format="concise"):
    c = _customer(customer_id)
    orders = rows("SELECT * FROM orders WHERE customer_id=? ORDER BY order_number", customer_id)
    if month:
        orders = [o for o in orders if o["order_date"].startswith(month)]
    if not orders:
        when = f" in {month}" if month else ""
        return dumps({"customer": customer_label(c), "orders": [],
                      "note": f"{c['full_name']} has no orders{when}."})
    shown, note = _page(orders, page, "orders", "Narrow with month='YYYY-MM'.")
    out = []
    for o in shown:
        item = {"order_number": o["order_number"], "date": o["order_date"],
                "total_eur": o["total_eur"], "status": o["status"]}
        if response_format == "detailed":
            item.update(items=o["items_count"], shipping=o["shipping_method"])
        out.append(item)
    return _with_note({"customer": customer_label(c), "orders": out, "total": len(orders)}, note)


def list_transactions(order_number, response_format="concise"):
    o = rows("SELECT o.*, c.full_name, c.id cid FROM orders o JOIN customers c "
             "ON c.id=o.customer_id WHERE order_number=?", order_number)
    if not o:
        raise not_found(f"No order {order_number}. Order numbers look like ORD-10234; call "
                        "list_orders(customer_id=...) to see a customer's order numbers.")
    o = o[0]
    out = []
    for t in rows("SELECT * FROM transactions WHERE order_number=? ORDER BY created_at", order_number):
        item = {"time": t["created_at"].replace("T", " "), "type": t["type"],
                "status": t["status"], "amount_eur": t["amount_eur"]}
        if response_format == "detailed":
            item.update(txn_id=t["txn_id"], card_last4=t["card_last4"])
        out.append(item)
    return dumps({"order_number": order_number,
                  "customer": f"{o['full_name']} (id {o['cid']})",
                  "order_total_eur": o["total_eur"], "order_status": o["status"],
                  "transactions": out})


def read_logs(date, contains=None, level=None, service=None, page=1, response_format="concise"):
    if not LOG_RANGE[0] <= date <= LOG_RANGE[1]:
        return dumps({"date": date, "lines": [],
                      "note": f"No logs on {date}. Logs cover {LOG_RANGE[0]} to {LOG_RANGE[1]}."})
    lines = rows("SELECT l.*, c.id cid, c.full_name FROM logs l LEFT JOIN customers c "
                 "ON c.uuid=l.user_uuid WHERE substr(l.ts,1,10)=? ORDER BY l.id", date)
    if contains:
        lines = [x for x in lines if contains.lower() in x["message"].lower()]
    if level:
        lines = [x for x in lines if x["level"] == level]
    if service:
        lines = [x for x in lines if x["service"] == service]
    if not lines:
        return dumps({"date": date, "lines": [],
                      "note": "No log lines match these filters. Try fewer filters or a shorter "
                              "'contains' text."})
    shown, note = _page(lines, page, "lines",
                        "Narrow with contains='text', level='ERROR' or service='payment-gateway'.")
    out = []
    for x in shown:
        item = {"time": x["ts"][11:], "service": x["service"], "level": x["level"],
                "customer": f"{x['full_name']} (id {x['cid']})" if x["cid"] else None,
                "message": x["message"]}
        if response_format == "detailed":
            item.update(log_id=x["id"])
        out.append(item)
    return _with_note({"date": date, "lines": out, "total": len(lines)}, note)


def list_tickets(customer_id, response_format="concise"):
    c = _customer(customer_id)
    out = []
    for t in rows("SELECT * FROM tickets WHERE customer_id=? ORDER BY ticket_id", customer_id):
        item = {"ticket_id": t["ticket_id"], "date": t["created_date"], "reason": t["reason"],
                "status": t["status"]}
        if response_format == "detailed":
            item.update(channel=t["channel"])
        out.append(item)
    obj = {"customer": customer_label(c), "tickets": out}
    return _with_note(obj, None if out else f"{c['full_name']} has no support tickets.")


TOOLS = [
    tool("list_customers",
         "Find customers by name or city. Use it first to get a customer's numeric id, which "
         "get_customer, list_orders and list_tickets need. Returns {id, name, city} per customer, "
         "20 per page, with a note when more exist. Example: list_customers(name_contains='Shah')",
         {"name_contains": {"type": "string",
                            "description": "Part of the customer's name, case-insensitive, "
                                           "e.g. 'Shah' or 'Priya Shah'."},
          "city": {"type": "string", "description": "City name, e.g. 'Berlin'."},
          "page": PAGE_P, "response_format": FORMAT}),
    tool("get_customer",
         "Get one customer's profile by numeric id: name, email, city, signup date and number of "
         "orders. Example: get_customer(customer_id=12)",
         {"customer_id": CUSTOMER_ID, "response_format": FORMAT}, ["customer_id"]),
    tool("list_orders",
         "List one customer's orders, oldest first: order_number (like ORD-10234), date, "
         "total_eur, status (delivered, shipped, processing, or cancelled = never paid). Payments "
         "for an order are in list_transactions. 20 per page. "
         "Example: list_orders(customer_id=12, month='2026-03')",
         {"customer_id": CUSTOMER_ID,
          "month": {"type": "string", "pattern": r"^\d{4}-\d{2}$",
                    "description": "Only orders in this month, YYYY-MM, e.g. 2026-03."},
          "page": PAGE_P, "response_format": FORMAT}, ["customer_id"]),
    tool("list_transactions",
         "List the payments for one order: time, type ('charge' = money taken, 'refund' = money "
         "returned), amount_eur and status (charge: 'succeeded', or 'failed' = declined attempt, "
         "no money taken; refund: 'pending' = not yet returned, 'completed' = returned). A "
         "normally paid order has exactly one succeeded charge. "
         "Example: list_transactions(order_number='ORD-10234')",
         {"order_number": {"type": "string", "pattern": r"^ORD-\d{5}$",
                           "description": "Order number like ORD-10234, from list_orders."},
          "response_format": FORMAT}, ["order_number"]),
    tool("read_logs",
         "Read application log lines for one day (logs cover 2026-01-01 to 2026-06-30), oldest "
         "first. Each line: time, service, level, customer (name and id, or null for system "
         "lines) and message; error codes look like code=E-PAY-502. Filter with contains, level "
         "and service. 20 lines per page, with a note when more exist. "
         "Example: read_logs(date='2026-04-14', contains='E-PAY-502')",
         {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$",
                   "description": "Day in YYYY-MM-DD format, e.g. 2026-04-14."},
          "contains": {"type": "string",
                       "description": "Only lines whose message contains this text, "
                                      "case-insensitive, e.g. 'timeout' or 'E-PAY-502'."},
          "level": {"type": "string", "enum": ["INFO", "WARN", "ERROR"]},
          "service": {"type": "string", "enum": SERVICES},
          "page": PAGE_P, "response_format": FORMAT}, ["date"]),
    tool("list_tickets",
         "List a customer's support tickets: ticket_id, date, reason (the customer's own words), "
         "status (open, pending, closed). Example: list_tickets(customer_id=12)",
         {"customer_id": CUSTOMER_ID, "response_format": FORMAT}, ["customer_id"]),
]

dispatch = make_dispatch(TOOLS, {
    "list_customers": list_customers, "get_customer": get_customer, "list_orders": list_orders,
    "list_transactions": list_transactions, "read_logs": read_logs, "list_tickets": list_tickets,
}, actionable_error)
