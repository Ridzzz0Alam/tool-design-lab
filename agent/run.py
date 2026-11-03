"""Agent loop: one task, one tool condition, one model, one seed per run.

Identical across conditions; only TOOLS and dispatch differ. Every model call
goes through the response cache, every finished run is appended to
results/runs.jsonl, and runs already there are skipped, so the batch can be
stopped and restarted at any point.

Usage:
  .venv/bin/python -m agent.run --conditions A,D --split dev --seeds 1 --limit 2
  .venv/bin/python -m agent.run --tasks spend_lookup_1 --dry-run
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from agent.cache import ResponseCache, cache_key
from agent.verifier import extract_final, load_customer_names, verify
from tasks.make_tasks import DB, OUT as TASKS_PATH
from tools import CONDITIONS, load

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HOST = "http://localhost:11434"
MAX_TURNS = 10  # the longest oracle path needs 8 tool calls plus an answer turn
NUM_CTX = 4096
# num_predict caps runaway replies; the longest normal reply before the cap was 188 tokens
BASE_OPTIONS = {"temperature": 0.6, "num_ctx": NUM_CTX, "num_thread": 4, "num_predict": 512}

SYSTEM_PROMPT = (
    "You are a support analyst for an online shop. Use the tools to look up facts; "
    "do not guess. Before each tool call, write one short sentence saying why you make it. "
    "When you are done, end your reply with exactly one line:\n"
    "FINAL ANSWER: <answer>"
)
# Answer nudge: sent once, without tools, when the agent stops without a FINAL ANSWER line.
NUDGE = "Reply now with exactly one line: FINAL ANSWER: <answer>"

# Prompt tokens of each condition's tool schemas plus the 31-token chat baseline,
# measured with Ollama's tokenizer. Used for the full-context estimate (schema tokens + characters / 4).
SCHEMA_TOKENS = {"A": 304, "B": 889, "C": 1247, "D": 711}
BASELINE_TOKENS = 31
CHARS_PER_TOKEN = 4


class OllamaUnavailable(RuntimeError):
    """Ollama unreachable or crashed: abort the batch without recording the run."""


class OllamaChat:
    """POST /api/chat through the response cache. Returns (response, elapsed_s, cached)."""

    def __init__(self, cache, host=HOST, timeout=1800):
        self.cache, self.host, self.timeout = cache, host, timeout

    def __call__(self, model, messages, tools, options):
        key = cache_key(model, messages, tools, options)
        hit = self.cache.get(key)
        if hit:
            return hit[0], hit[1], True
        t0 = time.time()
        try:
            r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
                "model": model, "messages": messages, "tools": tools,
                "stream": False, "options": options})
        except requests.RequestException as e:
            raise OllamaUnavailable(f"{type(e).__name__}: {e}") from e
        elapsed = time.time() - t0
        if r.status_code >= 500:
            raise OllamaUnavailable(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:  # a property of this request (e.g. prompt too long): not cached
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}, elapsed, False
        data = r.json()
        self.cache.put(key, model, data, elapsed)
        return data, elapsed, False


# --- text-format tool calls --------------------------------------------------------

def parse_text_tool_calls(content):
    """Find tool calls written as JSON in `content` (llama3.2 quirk).

    Accepts {"name": ..., "parameters"|"arguments": {...}}, the same wrapped in
    {"function": ...}, or a list of those. Returns [(name, args_dict), ...].
    Replies containing a FINAL ANSWER line are treated as answers, not calls.
    """
    if not content or re.search(r"final\s+answer\s*[*_]*\s*[:：]", content, re.I):
        return []
    dec, calls, i = json.JSONDecoder(), [], 0
    while (i := _next_json_start(content, i)) != -1:
        try:
            obj, end = dec.raw_decode(content, i)
        except json.JSONDecodeError:
            i += 1
            continue
        for item in obj if isinstance(obj, list) else [obj]:
            call = _as_call(item)
            if call:
                calls.append(call)
        i = end
    return calls


def _next_json_start(s, i):
    hits = [p for p in (s.find("{", i), s.find("[", i)) if p != -1]
    return min(hits) if hits else -1


def _as_call(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("parameters", obj.get("arguments"))
    if not isinstance(name, str) or args is None:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return (name, args) if isinstance(args, dict) else None


# --- one run ------------------------------------------------------------------------

def run_id(task_id, condition, model, seed):
    return f"{re.sub(r'[^A-Za-z0-9.]+', '-', model)}__{condition}__{task_id}__s{seed}"


def estimate_context(condition, messages):
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return SCHEMA_TOKENS.get(condition, 0) + BASELINE_TOKENS + chars // CHARS_PER_TOKEN


def _call(chat, model, messages, tools, options, condition, label):
    """One model call. Returns (response, turn log entry)."""
    context_est = estimate_context(condition if tools else None, messages)
    resp, elapsed, cached = chat(model, messages, tools, options)
    return resp, {"turn": label, "elapsed_s": round(elapsed, 2), "cached": cached,
                  "prompt_eval_count": resp.get("prompt_eval_count", 0),
                  "eval_count": resp.get("eval_count", 0),
                  "output_capped": resp.get("done_reason") == "length",
                  # Ollama's own compute times; prompt_eval_count does not track prefix-cache reuse
                  "prompt_eval_s": round(resp.get("prompt_eval_duration", 0) / 1e9, 2),
                  "eval_s": round(resp.get("eval_duration", 0) / 1e9, 2),
                  "context_tokens_est": context_est, "tool_results": []}


def run_one(task, condition, model, seed, chat, customer_names, max_turns=MAX_TURNS):
    """Run the agent loop once. Returns (record, transcript)."""
    mod = load(condition)
    options = {**BASE_OPTIONS, "seed": seed}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]}]
    turns, tools_used = [], []
    errors = {"invalid_tool": 0, "invalid_args": 0, "exception": 0}
    text_calls, final_text, ollama_error, done = 0, "", None, False

    while len(turns) < max_turns:
        resp, turn = _call(chat, model, messages, mod.TOOLS, options, condition, len(turns) + 1)
        turns.append(turn)
        if "error" in resp:
            ollama_error = turn["ollama_error"] = resp["error"]
            break

        msg = resp["message"]
        messages.append(msg)
        final_text = msg.get("content") or ""
        calls = [(c["function"]["name"], c["function"].get("arguments") or {})
                 for c in msg.get("tool_calls") or []]
        if not calls:
            calls = parse_text_tool_calls(final_text)
            text_calls += len(calls)
            turn["text_format_tool_calls"] = len(calls)
        if not calls:
            done = True
            break

        for name, args in calls:
            if not isinstance(args, dict):
                args = {"_raw": args}  # rejected by the validator as invalid_args
            result = mod.dispatch(name, args)
            tools_used.append(name)
            if result["error"]:
                errors[result["error"]] += 1
            turn["tool_results"].append({"name": name, "arguments": args, "error": result["error"]})
            messages.append({"role": "tool", "tool_name": name, "content": result["content"]})

    strict = verify(final_text, task, customer_names)
    nudge_turn, verdict = None, strict
    if done and extract_final(final_text) is None:
        messages.append({"role": "user", "content": NUDGE})
        resp, nudge_turn = _call(chat, model, messages, [], options, condition, "nudge")
        if "error" in resp:
            nudge_turn["ollama_error"] = resp["error"]
        else:
            messages.append(resp["message"])
            verdict = verify(resp["message"].get("content") or "", task, customer_names)
    log = turns + ([nudge_turn] if nudge_turn else [])
    expected = task["expected_tools"][condition]
    max_ctx = max(t["context_tokens_est"] for t in log)
    record = {
        "run_id": run_id(task["id"], condition, model, seed),
        "task_id": task["id"], "type": task["type"], "split": task["split"],
        "condition": condition, "model": model, "seed": seed,
        "success": verdict["success"], "final_answer": verdict["final_answer"],
        "verify_reason": verdict["reason"], "expected_answer": task["answer"],
        "success_strict": strict["success"], "nudged": nudge_turn is not None,
        "hit_max_turns": not done and ollama_error is None,
        "turns": len(turns), "tool_calls": len(tools_used),
        "tool_errors": errors, "tool_error_count": sum(errors.values()),
        "text_format_tool_calls": text_calls,
        "tools_used": tools_used,
        "expected_tools_used": set(expected) <= set(tools_used),
        "prompt_eval_count": sum(t["prompt_eval_count"] for t in log),
        "eval_count": sum(t["eval_count"] for t in log),
        "prompt_eval_s": round(sum(t["prompt_eval_s"] for t in log), 2),
        "eval_s": round(sum(t["eval_s"] for t in log), 2),
        "context_tokens_est_sum": sum(t["context_tokens_est"] for t in turns),
        "context_tokens_est_max": max_ctx,
        "context_overflow_est": max_ctx >= NUM_CTX,
        "wall_s": round(sum(t["elapsed_s"] for t in log), 2),
        "cached_turns": sum(t["cached"] for t in log),
        "output_capped_turns": sum(t["output_capped"] for t in log),
        "ollama_error": ollama_error,
        "max_turns": max_turns, "options": options,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    transcript = {**record, "tools": mod.TOOLS, "messages": messages, "turn_log": log}
    return record, transcript


# --- batch --------------------------------------------------------------------------

def load_tasks(path=TASKS_PATH):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def done_run_ids(runs_path):
    if not runs_path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in runs_path.read_text().splitlines() if line.strip()}


def plan(tasks, conditions, models, seeds):
    return [(m, s, c, t) for m in models for s in seeds for c in conditions for t in tasks]


def run_batch(jobs, chat, results=RESULTS, max_turns=MAX_TURNS, log=print):
    runs_path, tdir = results / "runs.jsonl", results / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    finished = done_run_ids(runs_path)
    names = load_customer_names(DB)
    todo = [j for j in jobs if run_id(j[3]["id"], j[2], j[0], j[1]) not in finished]
    log(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for n, (model, seed, cond, task) in enumerate(todo, 1):
        t0 = time.time()
        record, transcript = run_one(task, cond, model, seed, chat, names, max_turns)
        (tdir / f"{record['run_id']}.json").write_text(json.dumps(transcript, indent=1, ensure_ascii=False))
        with runs_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"[{n}/{len(todo)}] {record['run_id']}: {'OK ' if record['success'] else 'BAD'}"
            f"{' (nudged)' if record['nudged'] else ''} "
            f"answer={record['final_answer']!r} turns={record['turns']} calls={record['tool_calls']} "
            f"errors={record['tool_error_count']} model_s={record['wall_s']:.0f} "
            f"(this session {time.time() - t0:.0f}s, {record['cached_turns']} cached turns)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="llama3.2:3b", help="comma-separated")
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated, e.g. A,D")
    ap.add_argument("--seeds", default="1", help="comma-separated, e.g. 1,2,3")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all in --split)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, help="run at most N tasks (after filtering)")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)

    tasks = load_tasks()
    if a.tasks:
        wanted = a.tasks.split(",")
        unknown = set(wanted) - {t["id"] for t in tasks}
        if unknown:
            ap.error(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"] in wanted]
    if a.split != "all":
        tasks = [t for t in tasks if t["split"] == a.split]
    tasks = tasks[:a.limit] if a.limit else tasks
    conditions = [c.strip().upper() for c in a.conditions.split(",")]
    if set(conditions) - set(CONDITIONS):
        ap.error(f"conditions must be among {CONDITIONS}")
    jobs = plan(tasks, conditions, a.models.split(","), [int(s) for s in a.seeds.split(",")])

    if a.dry_run:
        done = done_run_ids(a.results / "runs.jsonl")
        for m, s, c, t in jobs:
            rid = run_id(t["id"], c, m, s)
            print(("done  " if rid in done else "todo  ") + rid)
        return
    cache = ResponseCache()
    try:
        run_batch(jobs, OllamaChat(cache, a.host), a.results, a.max_turns)
    except OllamaUnavailable as e:
        sys.exit(f"Ollama unavailable, stopping (finished runs are saved; rerun to resume): {e}")
    except KeyboardInterrupt:
        sys.exit("Interrupted. Finished runs and cached model calls are saved; rerun to resume.")
    finally:
        cache.close()


if __name__ == "__main__":
    main()
