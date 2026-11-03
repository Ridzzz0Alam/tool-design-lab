"""Condition B: A's tools with prompt-engineered descriptions only.

Descriptions are written for a new hire (purpose, when to use, field meanings,
how resources relate, an example call) and parameter names are unambiguous.
Responses and errors are identical to condition A.
"""
from tools.common import make_dispatch, terse_error, tool
from tools.condition_a import (raw_get_customer, raw_list_customers, raw_list_orders,
                               raw_list_tickets, raw_list_transactions, raw_read_logs)

CUSTOMER_ID = {"type": "integer",
               "description": "Numeric customer id (1-60) from list_customers. Not the uuid, not the name."}

TOOLS = [
    tool("list_customers",
         "List every customer in the shop's CRM. Use it to find a customer's numeric id from "
         "their name (get_customer, list_orders and list_tickets need that id), or to find which "
         "customer a log line's user_uuid belongs to. Returns a JSON array, one object per "
         "customer: id (numeric customer id), uuid (internal id; appears as user_uuid in logs), "
         "full_name, email, city, signup_date (YYYY-MM-DD), loyalty_tier, marketing_opt_in, "
         "crm_source. Example: list_customers()"),
    tool("get_customer",
         "Get one customer's profile by numeric customer id, when you already know the id. "
         "Returns one JSON object with the same fields as list_customers. "
         "Example: get_customer(customer_id=12)",
         {"customer_id": CUSTOMER_ID}, ["customer_id"]),
    tool("list_orders",
         "List all orders placed by one customer. Each order: order_number (like ORD-10234), "
         "customer_id, order_date (YYYY-MM-DD), created_at, total_eur (order total in euros), "
         "status (delivered, shipped, processing, or cancelled = never paid), items_count, "
         "shipping_method, currency, warehouse_code. Payments for an order are in "
         "list_transactions. Example: list_orders(customer_id=12)",
         {"customer_id": CUSTOMER_ID}, ["customer_id"]),
    tool("list_transactions",
         "List all payment transactions for one order. Each: txn_id, order_number, created_at, "
         "type ('charge' = money taken from the customer, 'refund' = money returned), amount_eur, "
         "status (charge: 'succeeded', or 'failed' = declined attempt, no money taken; refund: "
         "'pending' = not yet returned, 'completed' = returned), card_last4, gateway_ref. A "
         "normally paid order has exactly one succeeded charge. "
         "Example: list_transactions(order_number='ORD-10234')",
         {"order_number": {"type": "string", "pattern": r"^ORD-\d{5}$",
                           "description": "Order number like ORD-10234, from list_orders."}},
         ["order_number"]),
    tool("read_logs",
         "Read all application log lines for one day (logs cover 2026-01-01 to 2026-06-30). "
         "Each line: id, ts (timestamp), service (web, auth, checkout, payment-gateway, payments, "
         "inventory, email), level (INFO, WARN, ERROR), user_uuid (the customer's uuid, matching "
         "uuid in list_customers; null for system lines), message (error codes look like "
         "code=E-PAY-502). Example: read_logs(date='2026-04-14')",
         {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$",
                   "description": "Day in YYYY-MM-DD format, e.g. 2026-04-14."}},
         ["date"]),
    tool("list_tickets",
         "List support tickets opened by one customer: ticket_id, customer_id, created_date, "
         "reason (the customer's own words), status (open, pending, closed), channel. "
         "Example: list_tickets(customer_id=12)",
         {"customer_id": CUSTOMER_ID}, ["customer_id"]),
]

dispatch = make_dispatch(TOOLS, {
    "list_customers": raw_list_customers,
    "get_customer": raw_get_customer,
    "list_orders": raw_list_orders,
    "list_transactions": raw_list_transactions,
    "read_logs": raw_read_logs,
    "list_tickets": raw_list_tickets,
}, terse_error)
