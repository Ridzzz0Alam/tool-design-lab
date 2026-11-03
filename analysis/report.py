"""Tables, paired comparisons, failure categories and charts from results/runs.jsonl.

Writes results/summary.md and results/charts/*.png.

Usage: .venv/bin/python -m analysis.report [--model llama3.2:3b]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONDITIONS = ("A", "B", "C", "D")
PAIRS = (("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"))
N_BOOT = 10_000

# dataviz reference palette (light mode): categorical slots 1-2, text and grid inks
SERIES = ("#2a78d6", "#eb6834")
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

GAVE_UP_RE = re.compile(
    r"\b(unable|couldn'?t|could not|cannot|can'?t|not available|insufficient|unknown|"
    r"no (information|data|customer information)|not specified|not found|"
    r"don'?t have|do not have|not determine|large language model)\b", re.I)

CATEGORIES = ("hit max turns", "context overflow", "format failure", "invented tool",
              "wrong arguments", "wrong tool / incomplete path", "gave up", "wrong final answer")


def load_runs(path=RESULTS / "runs.jsonl", model=None):
    runs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in runs if model is None or r["model"] == model]


# --- statistics -----------------------------------------------------------------

def by_task(runs, key="success"):
    """{task_id: [values over seeds]} for one condition's runs."""
    out = defaultdict(list)
    for r in runs:
        out[r["task_id"]].append(float(r[key]))
    return out


def bootstrap_mean(runs, key="success", n=N_BOOT, seed=0):
    """Mean of `key` with a 95% two-level bootstrap CI: tasks, then seeds within each task."""
    groups = [np.array(v) for v in by_task(runs, key).values()]
    rng = np.random.default_rng(seed)
    point = float(np.mean(np.concatenate(groups)))
    stats = np.empty(n)
    for i in range(n):
        picks = rng.integers(len(groups), size=len(groups))
        stats[i] = np.mean(np.concatenate([g[rng.integers(len(g), size=len(g))] for g in
                                           (groups[p] for p in picks)]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired(runs_x, runs_y, key="success", n=N_BOOT, seed=0):
    """Per-task mean difference y - x (same tasks), bootstrapped over tasks."""
    tx, ty = by_task(runs_x, key), by_task(runs_y, key)
    tasks = sorted(set(tx) & set(ty))
    d = np.array([np.mean(ty[t]) - np.mean(tx[t]) for t in tasks])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "tasks": len(d),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum())}


# --- failure categories ---------------------------------------------------------

def categorize(r):
    """Primary failure category of a failed run (rule-based; precedence = CATEGORIES order)."""
    if r["success"]:
        return None
    errs = r["tool_errors"]
    if r["hit_max_turns"]:
        return "hit max turns"
    if r["context_overflow_est"]:
        return "context overflow"
    if not r["final_answer"]:
        return "format failure"
    if errs["invalid_tool"]:
        return "invented tool"
    if errs["invalid_args"] or errs["exception"]:
        return "wrong arguments"
    if not r["expected_tools_used"]:
        return "wrong tool / incomplete path"
    if GAVE_UP_RE.search(r["final_answer"]):
        return "gave up"
    return "wrong final answer"


def outcome(r):
    """What the failed run ended with, independent of cause."""
    if r["success"]:
        return "correct"
    if not r["final_answer"]:
        return "no answer"
    return "gave up" if GAVE_UP_RE.search(r["final_answer"]) else "confidently wrong"


def _yes_no(ans):
    w = re.findall(r"[a-z]+", (ans or "").lower())
    return w[0] if w and w[0] in ("yes", "no") else "other"


# --- tables ---------------------------------------------------------------------

def pct(x):
    return f"{100 * x:.0f}%"


def ci(p):
    return f"{pct(p[0])} [{pct(p[1])}, {pct(p[2])}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def mean(runs, f):
    return sum(f(r) for r in runs) / len(runs) if runs else 0.0


def summarize(runs):
    cond = {c: [r for r in runs if r["condition"] == c] for c in CONDITIONS}
    test = {c: [r for r in v if r["split"] == "test"] for c, v in cond.items()}
    test_open = {c: [r for r in v if r["type"] != "refund_status"] for c, v in test.items()}
    seeds = sorted({r["seed"] for r in runs})
    sec = [f"# Results summary\n\nGenerated by `analysis/report.py` from `results/runs.jsonl`: "
           f"{len(runs)} runs, model(s) {sorted({r['model'] for r in runs})}, seeds {seeds}, "
           f"{len({r['task_id'] for r in runs})} tasks. CIs are 95% two-level bootstrap "
           f"(tasks, then seeds within task), {N_BOOT:,} resamples."]

    rows, success = [], {}
    for c in CONDITIONS:
        success[c] = {"test": bootstrap_mean(test[c]), "all": bootstrap_mean(cond[c])}
        rows.append([c, ci(success[c]["test"]), ci(bootstrap_mean(test[c], "success_strict")),
                     ci(bootstrap_mean(test_open[c])),
                     ci(success[c]["all"]), ci(bootstrap_mean(cond[c], "success_strict")),
                     f"{sum(r['success'] for r in test[c])}/{len(test[c])}"])
    sec.append("## Success rate\n\nHeadline = test split, after the answer nudge (one tool-less reminder to give a FINAL ANSWER line). "
               "Strict = without the nudge. \"Test, no yes/no\" drops the 4 refund-status tasks, "
               "where always answering \"no\" scores 50% (see the table below).\n\n" + md_table(
                   ["Condition", "Test (headline)", "Test strict", "Test, no yes/no", "All 20", "All 20 strict",
                    "Test runs OK"], rows))
    answers = {c: Counter(_yes_no(r["final_answer"]) for r in cond[c] if r["type"] == "refund_status")
               for c in CONDITIONS}
    sec.append("## Yes/no answers on the refund-status tasks (2 \"yes\" and 2 \"no\" tasks × 3 seeds)\n\n" + md_table(
        ["Answer", *CONDITIONS], [[k, *(answers[c][k] for c in CONDITIONS)] for k in ("yes", "no", "other")]))

    by_type = defaultdict(dict)
    for c in CONDITIONS:
        for t in sorted({r["type"] for r in runs}):
            rs = [r for r in cond[c] if r["type"] == t]
            by_type[t][c] = f"{sum(r['success'] for r in rs)}/{len(rs)}"
    sec.append("## Success by task type (all splits, runs OK / runs)\n\n" + md_table(
        ["Task type", *CONDITIONS], [[t, *(v[c] for c in CONDITIONS)] for t, v in by_type.items()]))

    rows = []
    for split, pool in (("test", test), ("all", cond)):
        for x, y in PAIRS:
            p = paired(pool[x], pool[y])
            rows.append([split, f"{x} → {y}", f"{100 * p['diff']:+.0f} pp [{100 * p['lo']:+.0f}, {100 * p['hi']:+.0f}]",
                         f"{p['better']} / {p['worse']} / {p['tied']}"])
    sec.append("## Paired comparisons\n\nPer task: mean success over seeds in the second condition minus "
               "the first; averaged over tasks, CI bootstrapped over tasks.\n\n" + md_table(
                   ["Split", "Change", "Δ success [95% CI]", "Tasks better / worse / tied"], rows))

    def eff(rs):
        return [f"{mean(rs, lambda r: r['turns']):.1f}", f"{mean(rs, lambda r: r['tool_calls']):.1f}",
                f"{mean(rs, lambda r: r['tool_error_count']):.2f}",
                f"{mean(rs, lambda r: r['tool_errors']['invalid_tool']):.2f}",
                f"{mean(rs, lambda r: r['tool_errors']['invalid_args']):.2f}",
                f"{mean(rs, lambda r: r['context_tokens_est_sum']):,.0f}",
                f"{mean(rs, lambda r: r['context_tokens_est_max']):,.0f}",
                f"{mean(rs, lambda r: r['eval_count']):.0f}",
                f"{mean(rs, lambda r: r['wall_s']):.0f}"]
    sec.append("## Efficiency per run (all 20 tasks)\n\nToken use = estimated context size summed over model "
               "calls (schema tokens + characters ÷ 4). Output tokens and time from Ollama.\n\n" + md_table(
                   ["Condition", "Turns", "Tool calls", "Tool errors", "…invalid tool", "…invalid args",
                    "Token use (est.)", "Peak context (est.)", "Output tokens", "Model time (s)"],
                   [[c, *eff(cond[c])] for c in CONDITIONS]))

    sec.append("## Model behaviour counters (all runs)\n\n" + md_table(
        ["Condition", "Runs", "Text-format tool calls", "Native tool calls", "Nudged runs",
         "Capped replies", "Context overflow (est.)", "Hit max turns", "Expected tools used"],
        [[c, len(rs), sum(r["text_format_tool_calls"] for r in rs),
          sum(r["tool_calls"] - r["text_format_tool_calls"] for r in rs), sum(r["nudged"] for r in rs),
          sum(r.get("output_capped_turns", 0) for r in rs), sum(r["context_overflow_est"] for r in rs),
          sum(r["hit_max_turns"] for r in rs), pct(mean(rs, lambda r: r["expected_tools_used"]))]
         for c, rs in cond.items()]))

    cats = {c: Counter(categorize(r) for r in rs if not r["success"]) for c, rs in cond.items()}
    sec.append("## Failure categories (failed runs, all 20 tasks)\n\nRule-based primary cause; the first "
               "matching rule wins, in row order. \"wrong arguments\" includes not-found ids.\n\n" + md_table(
                   ["Category", *CONDITIONS], [[k, *(cats[c][k] for c in CONDITIONS)] for k in CATEGORIES]
                   + [["**total failed**", *(sum(cats[c].values()) for c in CONDITIONS)]]))
    outs = {c: Counter(outcome(r) for r in rs) for c, rs in cond.items()}
    sec.append("## How runs end (all 20 tasks)\n\n" + md_table(
        ["Outcome", *CONDITIONS], [[k, *(outs[c][k] for c in CONDITIONS)]
                                   for k in ("correct", "confidently wrong", "gave up", "no answer")]))
    return "\n\n".join(sec) + "\n", cond, success


# --- charts ---------------------------------------------------------------------

def _axes(title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    return fig, ax


def _save(fig, name, outdir):
    fig.tight_layout()
    fig.savefig(outdir / name, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_success(success, outdir):
    fig, ax = _axes("Task success by tool condition (95% bootstrap CI)", "Success rate")
    x, w = np.arange(len(CONDITIONS)), 0.36
    for i, (split, label) in enumerate((("test", "Test tasks (headline)"), ("all", "All 20 tasks"))):
        pts = [success[c][split] for c in CONDITIONS]
        vals = [p[0] for p in pts]
        err = [[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]]
        xs = x + (i - 0.5) * (w + 0.02)
        ax.bar(xs, vals, w, color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, 0.012, pct(v), ha="center", va="bottom", fontsize=8, color=SURFACE, fontweight="bold")
    ax.set_xticks(x, [f"{c}" for c in CONDITIONS])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _save(fig, "success_by_condition.png", outdir)


def chart_bars(cond, f, title, ylabel, name, outdir, fmt="{:.1f}"):
    fig, ax = _axes(title, ylabel)
    vals = [mean(cond[c], f) for c in CONDITIONS]
    ax.bar(CONDITIONS, vals, 0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for c, v in zip(CONDITIONS, vals):
        ax.text(c, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}" if max(vals) >= 1000 else f"{v:g}")
    _save(fig, name, outdir)


def charts(cond, success, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    chart_success(success, outdir)
    chart_bars(cond, lambda r: r["context_tokens_est_sum"], "Token use per run (estimated context, summed over calls)",
               "Tokens per run", "tokens_by_condition.png", outdir, "{:,.0f}")
    chart_bars(cond, lambda r: r["tool_error_count"], "Tool errors per run", "Errors per run",
               "errors_by_condition.png", outdir, "{:.2f}")
    chart_bars(cond, lambda r: r["tool_calls"], "Tool calls per run", "Calls per run",
               "calls_by_condition.png", outdir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    runs = load_runs(a.results / "runs.jsonl", a.model)
    text, cond, success = summarize(runs)
    (a.results / "summary.md").write_text(text)
    charts(cond, success, a.results / "charts")
    print(text)


if __name__ == "__main__":
    main()
