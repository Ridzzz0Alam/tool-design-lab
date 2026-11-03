"""Environment check: confirm Ollama is reachable and llama3.2 returns native tool_calls.

Also round-trips a role="tool" message to see which fields this Ollama
version accepts for tool results.

Usage: .venv/bin/python scripts/check_ollama.py [model]
"""
import json
import sys
import time

import requests

HOST = "http://localhost:11434"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
OPTIONS = {"temperature": 0.6, "seed": 1, "num_ctx": 4096, "num_thread": 4}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_order_total",
        "description": "Return the total in EUR for an order number like ORD-10234.",
        "parameters": {
            "type": "object",
            "properties": {"order_number": {"type": "string", "description": "e.g. ORD-10234"}},
            "required": ["order_number"],
        },
    },
}]


def chat(messages):
    t0 = time.time()
    r = requests.post(f"{HOST}/api/chat", json={
        "model": MODEL, "messages": messages, "tools": TOOLS,
        "stream": False, "options": OPTIONS,
    }, timeout=600)
    r.raise_for_status()
    data = r.json()
    print(f"  {time.time() - t0:.1f}s  prompt_eval_count={data.get('prompt_eval_count')}"
          f"  eval_count={data.get('eval_count')}")
    return data["message"]


print("Ollama version:", requests.get(f"{HOST}/api/version", timeout=5).json())

messages = [{"role": "user", "content": "What is the total of order ORD-10234? Use the tool."}]
print(f"\n[1] Asking {MODEL} to call a tool...")
msg = chat(messages)
print("  message:", json.dumps(msg, indent=2))

calls = msg.get("tool_calls") or []
if not calls:
    print("\nRESULT: no native tool_calls (content-only reply). Fallback parser will be needed.")
    sys.exit(1)
print(f"\nRESULT: native tool_calls OK ({len(calls)} call(s)).")

# Round-trip a tool result; Ollama documents `tool_name` for tool messages.
messages.append(msg)
messages.append({"role": "tool", "tool_name": calls[0]["function"]["name"],
                 "content": json.dumps({"order_number": "ORD-10234", "total_eur": 87.40})})
print("\n[2] Sending tool result back...")
final = chat(messages)
print("  reply:", final.get("content"))
print("\nRESULT: tool-result round trip OK." if "87" in (final.get("content") or "")
      else "\nRESULT: model did not use the tool result. Check transcript above.")
