"""Agent loop, cache and resume tests. A scripted fake model replaces Ollama."""
import json

import pytest

from agent import run
from agent.cache import ResponseCache, cache_key
from agent.verifier import load_customer_names
from tasks.make_tasks import DB

TASKS = {t["id"]: t for t in run.load_tasks()}
TASK = TASKS["duplicate_charge_1"]  # Freya Martin -> ORD-10426
NAMES = load_customer_names(DB)


def native(name, args, content=""):
    return {"message": {"role": "assistant", "content": content, "tool_calls": [
        {"id": "call_1", "function": {"index": 0, "name": name, "arguments": args}}]},
        "prompt_eval_count": 100, "eval_count": 10}


def text(content):
    return {"message": {"role": "assistant", "content": content},
            "prompt_eval_count": 50, "eval_count": 5}


class FakeChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, model, messages, tools, options):
        self.calls.append(json.loads(json.dumps(messages)))
        return self.replies.pop(0), 2.0, False


def go(replies, condition="D", task=TASK, max_turns=run.MAX_TURNS):
    chat = FakeChat(replies)
    record, transcript = run.run_one(task, condition, "llama3.2:3b", 1, chat, NAMES, max_turns)
    return record, transcript, chat


# --- text-format tool call parser ------------------------------------------------

@pytest.mark.parametrize("content,expected", [
    ('{"name": "logs_search", "parameters": {"query": "E-PAY-502"}}',
     [("logs_search", {"query": "E-PAY-502"})]),
    ('I will search.\n```json\n{"name": "x", "arguments": {"a": 1}}\n```',
     [("x", {"a": 1})]),
    ('{"type": "function", "function": {"name": "x", "arguments": "{\\"a\\": 1}"}}',
     [("x", {"a": 1})]),
    ('[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {"k": "v"}}]',
     [("a", {}), ("b", {"k": "v"})]),
])
def test_parse_text_tool_calls(content, expected):
    assert run.parse_text_tool_calls(content) == expected


@pytest.mark.parametrize("content", [
    "", "No idea.", "FINAL ANSWER: ORD-10426",
    'Result {"name": "x", "parameters": {}}\nFINAL ANSWER: 3',  # answer wins
    '{"name": "Freya Martin", "city": "Oslo"}',                 # no parameters key
    "{broken json", "[1, 2, 3]",
])
def test_parse_text_tool_calls_ignores(content):
    assert run.parse_text_tool_calls(content) == []


# --- run_one ---------------------------------------------------------------------

def test_success_path_native_calls():
    record, transcript, chat = go([
        native("crm_get_customer_context", {"customer_name": "Freya Martin"}, "Look her up."),
        text("ORD-10426 was charged twice.\nFINAL ANSWER: ORD-10426"),
    ])
    assert record["success"] and record["final_answer"] == "ORD-10426"
    assert (record["turns"], record["tool_calls"], record["tool_error_count"]) == (2, 1, 0)
    assert record["expected_tools_used"] and not record["hit_max_turns"]
    assert (record["prompt_eval_count"], record["eval_count"], record["wall_s"]) == (150, 15, 4.0)
    # turn 2 saw the tool result in the condition's format
    tool_msg = chat.calls[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "crm_get_customer_context"
    assert "ORD-10426" in tool_msg["content"]
    assert chat.calls[0][0] == {"role": "system", "content": run.SYSTEM_PROMPT}
    assert transcript["messages"][-1]["content"].endswith("FINAL ANSWER: ORD-10426")


def test_text_format_call_is_executed_and_counted():
    record, transcript, _ = go([
        text('{"name": "crm_get_customer_context", "parameters": {"customer_name": "Freya Martin"}}'),
        text("FINAL ANSWER: ORD-10426"),
    ])
    assert record["success"] and record["text_format_tool_calls"] == 1
    assert record["tools_used"] == ["crm_get_customer_context"]
    assert transcript["turn_log"][0]["text_format_tool_calls"] == 1


def test_tool_errors_counted_by_kind():
    record, _, _ = go([
        native("orders_find_duplicate_charges", {"customer_name": "Freya Martin"}),
        native("crm_get_customer_context", {"name": "Freya"}),
        native("crm_get_customer_context", "not a dict"),
        text("FINAL ANSWER: ORD-99999"),
    ])
    assert record["tool_errors"] == {"invalid_tool": 1, "invalid_args": 2, "exception": 0}
    assert not record["success"] and record["verify_reason"] == "wrong answer"


def test_multiple_calls_in_one_turn():
    reply = native("crm_search_customers", {"name_query": "Freya"})
    reply["message"]["tool_calls"].append(
        {"id": "call_2", "function": {"index": 1, "name": "crm_get_customer_context",
                                      "arguments": {"customer_name": "Freya Martin"}}})
    record, _, chat = go([reply, text("FINAL ANSWER: ORD-10426")])
    assert record["success"] and (record["turns"], record["tool_calls"]) == (2, 2)
    assert [m["role"] for m in chat.calls[1][-2:]] == ["tool", "tool"]


def test_hit_max_turns():
    loop = [native("crm_search_customers", {"name_query": "Freya"}) for _ in range(3)]
    record, _, chat = go(loop, max_turns=3)
    assert record["hit_max_turns"] and record["turns"] == 3 and len(chat.calls) == 3
    assert not record["nudged"]
    assert not record["success"] and record["verify_reason"] == "no FINAL ANSWER line"


def test_no_answer_line_gets_one_nudge_without_tools():
    record, transcript, chat = go([text("It was ORD-10426."), text("FINAL ANSWER: ORD-10426")])
    assert record["nudged"] and record["success"] and not record["success_strict"]
    assert record["turns"] == 1 and len(chat.calls) == 2  # the nudge is not an agent turn
    assert chat.calls[1][-1] == {"role": "user", "content": run.NUDGE}
    assert transcript["turn_log"][-1]["turn"] == "nudge"
    assert record["eval_count"] == 10  # nudge cost is included


def test_nudge_sends_no_tools():
    seen = []
    chat = FakeChat([text("No idea."), text("FINAL ANSWER: none")])
    run.run_one(TASK, "D", "m", 1, lambda m, msgs, tools, o: (seen.append(tools), chat(m, msgs, tools, o))[1],
                NAMES)
    assert seen[0] and seen[1] == []


def test_nudge_can_still_fail():
    record, _, _ = go([text("I cannot help with that."), text("Sorry, I don't know.")])
    assert record["nudged"] and not record["success"] and not record["success_strict"]
    assert not record["hit_max_turns"] and record["turns"] == 1


def test_no_nudge_when_answer_present_or_max_turns():
    record, _, chat = go([text("FINAL ANSWER: ORD-10426")])
    assert not record["nudged"] and record["success"] and record["success_strict"]
    assert len(chat.calls) == 1
    loop = [native("crm_search_customers", {"name_query": "Freya"}) for _ in range(2)]
    record, _, chat = go(loop, max_turns=2)
    assert record["hit_max_turns"] and not record["nudged"] and len(chat.calls) == 2


def test_ollama_durations_recorded():
    reply = text("FINAL ANSWER: ORD-10426")
    reply.update(prompt_eval_duration=2_500_000_000, eval_duration=4_000_000_000)
    record, _, _ = go([reply])
    assert (record["prompt_eval_s"], record["eval_s"]) == (2.5, 4.0)


def test_ollama_http_error_ends_run():
    record, _, _ = go([{"error": "HTTP 400: prompt too long"}])
    assert record["ollama_error"] == "HTTP 400: prompt too long"
    assert not record["success"] and not record["hit_max_turns"]


def test_context_estimate_grows_and_flags_overflow():
    record, transcript, _ = go([native("list_customers", {}), text("FINAL ANSWER: x")],
                               condition="A")
    est = [t["context_tokens_est"] for t in transcript["turn_log"]]
    assert est[0] >= run.SCHEMA_TOKENS["A"] and est[1] > 3000  # list_customers dump
    assert record["context_tokens_est_max"] == est[1]
    assert record["context_overflow_est"] == (est[1] >= run.NUM_CTX)


@pytest.mark.parametrize("cond", ["A", "B", "C", "D"])
def test_every_condition_runs(cond):
    first = {"A": ("list_customers", {}), "B": ("list_customers", {}),
             "C": ("list_customers", {"name_contains": "Freya"}),
             "D": ("crm_search_customers", {"name_query": "Freya"})}[cond]
    record, _, _ = go([native(*first), text("FINAL ANSWER: ORD-10426")], condition=cond)
    assert record["success"] and record["tool_error_count"] == 0


def test_run_id_is_filename_safe():
    assert run.run_id("spend_lookup_1", "C", "llama3.2:3b", 2) == "llama3.2-3b__C__spend_lookup_1__s2"


# --- cache -----------------------------------------------------------------------

def test_cache_key_sensitivity():
    base = ("m", [{"role": "user", "content": "hi"}], [{"t": 1}], {"seed": 1})
    k = cache_key(*base)
    assert k == cache_key(*base)
    assert k != cache_key("m2", *base[1:])
    assert k != cache_key("m", [{"role": "user", "content": "hi!"}], *base[2:])
    assert k != cache_key("m", base[1], [{"t": 2}], base[3])
    assert k != cache_key(*base[:3], {"seed": 2})


def test_cache_roundtrip(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    assert c.get("k") is None
    c.put("k", "m", {"message": {"content": "x"}}, 12.5)
    c.close()
    assert ResponseCache(tmp_path / "c.sqlite").get("k") == ({"message": {"content": "x"}}, 12.5)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def test_ollama_chat_caches_and_replays(tmp_path, monkeypatch):
    posts = []

    def fake_post(url, json, timeout):
        posts.append(json)
        return FakeResponse(200, text("FINAL ANSWER: 1"))

    monkeypatch.setattr(run.requests, "post", fake_post)
    chat = run.OllamaChat(ResponseCache(tmp_path / "c.sqlite"))
    args = ("llama3.2:3b", [{"role": "user", "content": "q"}], [], {"seed": 1})
    first, second = chat(*args), chat(*args)
    assert len(posts) == 1 and posts[0]["stream"] is False
    assert first[2] is False and second[2] is True and first[0] == second[0]
    assert second[1] == first[1]  # original timing kept


def test_ollama_chat_errors(tmp_path, monkeypatch):
    chat = run.OllamaChat(ResponseCache(tmp_path / "c.sqlite"))
    args = ("m", [], [], {})
    monkeypatch.setattr(run.requests, "post", lambda *a, **k: FakeResponse(400, {"error": "too long"}))
    assert chat(*args)[0]["error"].startswith("HTTP 400")
    assert chat.cache.get(cache_key(*args)) is None  # errors are not cached
    monkeypatch.setattr(run.requests, "post", lambda *a, **k: FakeResponse(500, {"error": "oom"}))
    with pytest.raises(run.OllamaUnavailable):
        chat(*args)

    def refuse(*a, **k):
        raise run.requests.ConnectionError("refused")
    monkeypatch.setattr(run.requests, "post", refuse)
    with pytest.raises(run.OllamaUnavailable):
        chat(*args)


# --- batch and resume ------------------------------------------------------------

class AlwaysAnswer:
    def __init__(self):
        self.n = 0

    def __call__(self, model, messages, tools, options):
        self.n += 1
        return text("FINAL ANSWER: ORD-10426"), 1.0, False


def test_batch_writes_and_resumes(tmp_path):
    jobs = run.plan([TASK, TASKS["spend_lookup_1"]], ["A", "D"], ["llama3.2:3b"], [1])
    chat = AlwaysAnswer()
    run.run_batch(jobs[:3], chat, tmp_path, log=lambda *_: None)
    assert chat.n == 3
    run.run_batch(jobs, chat, tmp_path, log=lambda *_: None)
    assert chat.n == 4  # only the missing run executed
    lines = [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and len({x["run_id"] for x in lines}) == 4
    assert len(list((tmp_path / "transcripts").glob("*.json"))) == 4
    run.run_batch(jobs, chat, tmp_path, log=lambda *_: None)
    assert chat.n == 4


def test_cli_dry_run(tmp_path, capsys):
    run.main(["--conditions", "a,d", "--split", "dev", "--seeds", "1,2", "--limit", "2",
              "--results", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 * 2 * 2 and all(line.startswith("todo  ") for line in out)


def test_cli_rejects_unknown_task(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["--tasks", "nope", "--dry-run", "--results", str(tmp_path)])


def test_capped_reply_is_flagged_and_run_continues():
    loop = text("total total total " * 100)
    loop["done_reason"] = "length"
    record, transcript, _ = go([loop, text("FINAL ANSWER: ORD-10426")])
    assert record["output_capped_turns"] == 1 and record["nudged"] and record["success"]
    assert transcript["turn_log"][0]["output_capped"] and not transcript["turn_log"][1]["output_capped"]


def test_options_cap_output():
    assert run.BASE_OPTIONS["num_predict"] == 512
