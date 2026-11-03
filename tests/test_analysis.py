"""Analysis helpers: bootstrap, paired comparison, failure categories."""
import pytest

from analysis import report


def run(task, seed, success, cond="A", **kw):
    r = {"task_id": task, "seed": seed, "condition": cond, "success": success, "success_strict": success,
         "hit_max_turns": False, "context_overflow_est": False, "final_answer": "x",
         "tool_errors": {"invalid_tool": 0, "invalid_args": 0, "exception": 0}, "expected_tools_used": True}
    r.update(kw)
    return r


def test_bootstrap_degenerate_and_deterministic():
    ones = [run(t, s, True) for t in "abc" for s in (1, 2)]
    assert report.bootstrap_mean(ones, n=200) == (1.0, 1.0, 1.0)
    mixed = [run(t, s, (t, s) in {("a", 1), ("b", 2)}) for t in "abcd" for s in (1, 2)]
    p = report.bootstrap_mean(mixed, n=500)
    assert p[0] == pytest.approx(0.25) and p[1] <= 0.25 <= p[2]
    assert p == report.bootstrap_mean(mixed, n=500)


def test_paired_counts_tasks():
    x = [run(t, s, False) for t in "abc" for s in (1, 2)]
    y = [run("a", s, True) for s in (1, 2)] + [run("b", 1, True), run("b", 2, False)] + \
        [run("c", s, False) for s in (1, 2)]
    p = report.paired(x, y, n=500)
    assert p["diff"] == pytest.approx(0.5) and (p["better"], p["worse"], p["tied"]) == (2, 0, 1)


@pytest.mark.parametrize("kw,expected", [
    ({"success": True}, None),
    ({"hit_max_turns": True, "final_answer": None}, "hit max turns"),
    ({"context_overflow_est": True}, "context overflow"),
    ({"final_answer": None}, "format failure"),
    ({"tool_errors": {"invalid_tool": 1, "invalid_args": 2, "exception": 0}}, "invented tool"),
    ({"tool_errors": {"invalid_tool": 0, "invalid_args": 1, "exception": 0}}, "wrong arguments"),
    ({"expected_tools_used": False}, "wrong tool / incomplete path"),
    ({"final_answer": "Unable to find the order."}, "gave up"),
    ({"final_answer": "I couldn't find any information."}, "gave up"),
    ({"final_answer": "ORD-10201"}, "wrong final answer"),
])
def test_categorize(kw, expected):
    r = run("t", 1, kw.pop("success", False), **kw)
    assert report.categorize(r) == expected


def test_outcome_and_yes_no():
    assert report.outcome(run("t", 1, False, final_answer="13")) == "confidently wrong"
    assert report.outcome(run("t", 1, False, final_answer="Unknown")) == "gave up"
    assert report.outcome(run("t", 1, False, final_answer=None)) == "no answer"
    assert [report._yes_no(a) for a in ("No.", "Yes", "No refunds are pending", None, "Maybe")] == \
        ["no", "yes", "no", "other", "other"]
