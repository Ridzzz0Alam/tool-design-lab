# tool-design-lab

A small-scale, fully local reproduction of the approach in Anthropic's ["Writing effective tools for agents"](https://www.anthropic.com/research/writing-tools-for-agents). It measures four tool-design conditions (naive wrappers → better descriptions → better responses → consolidated tools) on `llama3.2:3b` via Ollama, using 20 verifiable support-analyst tasks over a generated shop database.

## Reproduce in 5 commands

Requires Docker and Python 3.13 (Debian). Runs on CPU, with no paid APIs or cloud services.

```bash
docker run -d --name ollama -p 127.0.0.1:11434:11434 -v ollama:/root/.ollama --restart unless-stopped ollama/ollama@sha256:ebaef4f4c5d93660e1f25f8ad085a112f51ccfb0e8458073e71793959dba91c9
docker exec ollama ollama pull llama3.2:3b
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q
.venv/bin/python -m agent.run --seeds 1,2,3        # 240 runs, ~3.5 h on a 2-core laptop; Ctrl-C and rerun to resume
.venv/bin/python -m analysis.report                 # writes results/summary.md and results/charts/*.png
```

`env/shop.sqlite` and `tasks/tasks.jsonl` are included, and can be regenerated: `.venv/bin/python env/generate.py` and `.venv/bin/python tasks/make_tasks.py`. Both are deterministic (seed 42).

## Layout

| Path | What |
|---|---|
| `env/generate.py` | Deterministic data generator (60 customers, 300 orders, 510 transactions, 2,000 log lines, planted incidents) |
| `tasks/make_tasks.py` | 20 tasks with ground truth computed by code; 8 dev / 12 test |
| `tools/condition_a.py` … `condition_d.py` | The four tool sets: `TOOLS` (Ollama JSON schema) + `dispatch(name, args)` |
| `agent/run.py` | Agent loop, batch runner and CLI (`--conditions`, `--seeds`, `--tasks`, `--split`, `--models`, `--dry-run`) |
| `agent/cache.py` | SQLite cache of every model response (`cache/responses.sqlite`, created on first run); reruns repeat no model calls |
| `agent/verifier.py` | Deterministic answer checker (no LLM judge) |
| `analysis/report.py` | Tables, bootstrap CIs, paired comparisons, failure categories, charts |
| `results/` | Created by the runs: `runs.jsonl` (one record per run), `transcripts/`, `summary.md`, `charts/` |
| `tests/` | 307 tests, including oracle solutions proving every task is solvable in every condition |

Running a subset, e.g. two conditions on the dev tasks with seed 1:

```bash
.venv/bin/python -m agent.run --conditions A,D --split dev --seeds 1
```
