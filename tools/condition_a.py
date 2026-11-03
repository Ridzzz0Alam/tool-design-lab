"""Condition A: naive API wrappers with terse descriptions.

Ambiguous parameter names (id, user), full raw rows including UUIDs and every
field, no limits, and bare error codes (ERR_400 / ERR_404 / ERR_500).
"""
import re

from tools.common import (bad_args, dumps, make_dispatch, not_found, rows, terse_error,
                          tool)


# Raw handlers, shared with condition B (which only renames parameters).
def raw_list_customers():
    return dumps(rows("SELECT * FROM customers ORDER BY id"))


def raw_get_customer(customer_id):
    r = rows("SELECT * FROM customers WHERE id=?", customer_id)
    if not r:
        raise not_found(f"No customer with id {customer_id}.")
    return dumps(r[0])


def _require_customer(customer_id):
    if not rows("SELECT 1 FROM customers WHERE id=?", customer_id):
        raise not_found(f"No customer with id {customer_id}.")


def raw_list_orders(customer_id):
    _require_customer(customer_id)
    return dumps(rows("SELECT * FROM orders WHERE customer_id=? ORDER BY order_number", customer_id))


def raw_list_transactions(order_number):
    if not rows("SELECT 1 FROM orders WHERE order_number=?", order_number):
        raise not_found(f"No order {order_number}.")
    return dumps(rows("SELECT * FROM transactions WHERE order_number=? ORDER BY created_at",
                      order_number))


def raw_read_logs(date):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise bad_args(f"Bad date {date!r}.")
    return dumps(rows("SELECT * FROM logs WHERE substr(ts,1,10)=? ORDER BY id", date))


def raw_list_tickets(customer_id):
    _require_customer(customer_id)
    return dumps(rows("SELECT * FROM tickets WHERE customer_id=? ORDER BY ticket_id", customer_id))


TOOLS = [
    tool("list_customers", "List customers."),
    tool("get_customer", "Get a customer.", {"id": {"type": "integer"}}, ["id"]),
    tool("list_orders", "List orders for a user.", {"user": {"type": "integer"}}, ["user"]),
    tool("list_transactions", "List transactions.", {"id": {"type": "string"}}, ["id"]),
    tool("read_logs", "Read logs.", {"date": {"type": "string"}}, ["date"]),
    tool("list_tickets", "List tickets for a user.", {"user": {"type": "integer"}}, ["user"]),
]

dispatch = make_dispatch(TOOLS, {
    "list_customers": raw_list_customers,
    "get_customer": lambda id: raw_get_customer(id),
    "list_orders": lambda user: raw_list_orders(user),
    "list_transactions": lambda id: raw_list_transactions(id),
    "read_logs": raw_read_logs,
    "list_tickets": lambda user: raw_list_tickets(user),
}, terse_error)
