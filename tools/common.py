"""Shared plumbing for the tool conditions: DB access, argument validation,
error classification, and customer-name matching.

Every condition module exposes TOOLS (Ollama tool schemas) and
dispatch(name, args) -> {"content": str, "error": None | kind}, where kind is
one of the three error metrics: invalid_tool, invalid_args, exception.
"""
import difflib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "env" / "shop.sqlite"
LOG_RANGE = ("2026-01-01", "2026-06-30")
_con = None


def db():
    global _con
    if _con is None:
        _con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        _con.row_factory = sqlite3.Row
    return _con


def rows(sql, *args):
    return [dict(r) for r in db().execute(sql, args)]


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class ToolError(Exception):
    """kind: metric bucket; code: terse code shown by A/B; message: actionable text shown by C/D."""

    def __init__(self, kind, code, message):
        super().__init__(message)
        self.kind, self.code, self.message = kind, code, message


def bad_args(message):
    return ToolError("invalid_args", "ERR_400", message)


def not_found(message):
    return ToolError("invalid_args", "ERR_404", message)


def tool(name, description, properties=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}}


def validate(schema, args):
    """Check args against a tool schema; coerce numeric strings for integer params."""
    fn = schema["function"]
    name, props = fn["name"], fn["parameters"]["properties"]
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            raise bad_args(f"Arguments for {name} must be a JSON object, got {args!r}.")
    if not isinstance(args, dict):
        raise bad_args(f"Arguments for {name} must be a JSON object, got {args!r}.")
    args = {k: v for k, v in args.items() if v is not None and v != ""}
    unknown = sorted(set(args) - set(props))
    if unknown:
        accepted = ", ".join(props) or "no parameters"
        raise bad_args(f"Unknown parameter(s) {', '.join(unknown)} for {name}. It accepts: {accepted}.")
    for r in fn["parameters"]["required"]:
        if r not in args:
            raise bad_args(f"Missing required parameter '{r}' for {name}. "
                           f"{props[r].get('description', '')}".strip())
    out = {}
    for k, v in args.items():
        spec = props[k]
        hint = spec.get("description", "")
        if spec.get("type") == "integer":
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            elif isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
                v = int(v)
            if isinstance(v, bool) or not isinstance(v, int):
                raise bad_args(f"Parameter '{k}' of {name} must be an integer, got {v!r}. {hint}".strip())
        elif spec.get("type") == "string":
            if isinstance(v, (dict, list)):
                raise bad_args(f"Parameter '{k}' of {name} must be a string, got {v!r}. {hint}".strip())
            v = str(v).strip()
        if "enum" in spec and v not in spec["enum"]:
            raise bad_args(f"Parameter '{k}' of {name} must be one of {spec['enum']}, got {v!r}.")
        if "pattern" in spec and not re.fullmatch(spec["pattern"], v):
            raise bad_args(f"Parameter '{k}' of {name} has the wrong format: {v!r}. {hint}".strip())
        out[k] = v
    return out


def make_dispatch(schemas, handlers, render_error):
    by_name = {s["function"]["name"]: s for s in schemas}
    assert set(by_name) == set(handlers)

    def dispatch(name, args):
        try:
            if name not in by_name:
                raise ToolError("invalid_tool", "ERR_404",
                                f"Unknown tool '{name}'. Available tools: {', '.join(by_name)}.")
            return {"content": handlers[name](**validate(by_name[name], args)), "error": None}
        except ToolError as e:
            return {"content": render_error(e), "error": e.kind}
        except Exception as e:  # noqa: BLE001 - every failure goes back to the model
            err = ToolError("exception", "ERR_500",
                            f"Internal error in {name}: {type(e).__name__}: {e}. "
                            "Check the arguments against the tool description and try again.")
            return {"content": render_error(err), "error": "exception"}

    return dispatch


def terse_error(e):
    return e.code


def actionable_error(e):
    return f"Error: {e.message}"


# --- names -----------------------------------------------------------------------

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


def match_customers(query):
    """Customers whose name contains the query (or every word of it), case/accent-insensitive."""
    q = norm(query)
    words = q.split()
    out = []
    for c in rows("SELECT * FROM customers ORDER BY id"):
        n = norm(c["full_name"])
        if q and (q in n or all(w in n for w in words)):
            out.append(c)
    return out


def suggest_names(query, n=3):
    names = {norm(c["full_name"]): c["full_name"] for c in rows("SELECT full_name FROM customers")}
    q = norm(query)
    close = difflib.get_close_matches(q, list(names), n=n, cutoff=0.5)
    for w in q.split():  # also compare single words, e.g. a misspelled surname
        for name_norm in names:
            if name_norm not in close and any(
                    difflib.SequenceMatcher(None, w, part).ratio() >= 0.8 for part in name_norm.split()):
                close.append(name_norm)
    return [names[c] for c in close[:n]]


def customer_label(c):
    return f"{c['full_name']} (id {c['id']})"
