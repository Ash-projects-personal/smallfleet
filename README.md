# SmallFleet

I built SmallFleet because running every step of an agent on a frontier
model is expensive, and nobody had measured which steps are safe to
downgrade. So I built a router that lives inside the agent's own loop and
decides, call by call, which model handles that call.

## The one idea

**Agent-loop routing, not LLM routing.** Everything else in this space
routes standalone prompts or chat requests: RouteLLM, FrugalGPT,
LiteLLM, Martian, NotDiamond, RouteLabs Router, llama.cpp's own router
mode. SmallFleet does something different. It lives inside a running
agent loop, classifies each individual tool call by kind (routing,
extraction, tool call, summarization, reasoning, writing), sends it to
the cheapest model tier that can handle that call, and learns the routing
policy from the agent's own runs on its own task set. If this ever becomes
a generic prompt router, it is a RouteLLM clone and I have failed. The
angle is the product.

## How it works

- **Router** (`smallfleet/router.py`): a small classifier plus a rules
  layer tags each outgoing agent call by kind. Routine kinds go to the
  local model; hard kinds go to the frontier model. The policy is a plain
  JSON file, versioned, with the evidence behind every change.
- **Escalation**: when the local model returns malformed JSON or a
  low-confidence answer, the router retries once locally with a repair
  prompt, then escalates to the frontier tier. No silent wrong answers.
- **Ledger** (`smallfleet/ledger.py`): every call is logged in SQLite
  with model, tier, kind, tokens, measured latency, and cost. The
  dashboard shows total cost, total latency, and accuracy on a fixed
  10-task set, side by side against an all-frontier run of the same
  tasks.
- **Learning** (`router.learn_policy`): after a run, the policy is
  updated from the ledger's per-kind local success rates. A kind stays
  local when local calls succeed reliably; it moves to frontier when they
  do not. Kinds with fewer than 3 local calls keep their assignment.
- **Agent** (`smallfleet/agent.py`): a small agent loop with named nodes
  (plan, compute, extract, summarize, reason, write, answer). The same
  nodes run under a plain Python loop and under a LangGraph StateGraph
  (`build_graph`). Every LLM call a node makes goes through the router.
- **Tools** (`smallfleet/tools.py`): deterministic tools (calculator,
  document lookup, notes search, file write) so every task is
  machine-checkable. No LLM judges anything.

## How to run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sh models/download_models.sh                # Qwen2.5-0.5B + 3B, GGUF
.venv/bin/python run_demo.py --learn             # 10 tasks, twice
.venv/bin/python dashboard.py                    # http://127.0.0.1:8100/
.venv/bin/python -m pytest tests/                # unit tests, no models needed
```

The model files are not in the repo: GitHub caps files at 100 MB and the
two GGUF files are about 469 MB and 2.0 GB. Run
`sh models/download_models.sh` (or
`.venv/bin/python -m smallfleet.download_models`) once before the demo.
It downloads both models from Hugging Face. If the models are missing,
`run_demo.py` and `smoke_test.py` say so plainly instead of crashing:
models not downloaded, run models/download_models.sh.

`run_demo.py` runs the same 10-task agent job twice: once all-frontier,
once routed, same tasks, same agent logic. It prints the comparison and
writes a JSON report to `runs/`. `--learn` updates the routing policy
from the routed run.

## What the demo measures

The demo is honest about what is measured and what is modeled:

- **Latency and accuracy are measured.** Every number comes from a real
  run on the demo machine, on the fixed 10-task set.
- **Cost is list-price math, not a bill.** Frontier cost is computed from
  measured tokens at the published GPT-4o-mini list price ($0.15/1M
  input, $0.60/1M output tokens, 2026-10-25). Local calls cost $0.00
  marginal: the model runs on hardware already owned.
- **The frontier tier in this local demo is a local Qwen2.5-3B stand-in,
  not an API model.** I have no API key in this environment, so the
  frontier tier is served by a local 3B model. The code also ships an
  OpenAI-compatible backend (`backends.OpenAICompatibleBackend`): set
  `OPENAI_API_KEY` and the frontier tier becomes a real API model. The
  dashboard and the ledger say which one ran.

## Measured results

<!-- Filled in from real demo runs. Do not edit by hand. -->

Full demo run `runs/demo-20261025-230447.json`, Oct 25 2026. Same 10 tasks,
same agent logic, run twice. Machine: 2 CPUs, 7.7 GB RAM. The frontier tier
here is a local Qwen2.5-3B-Instruct Q4_K_M stand-in, not an API model.
Frontier cost is computed from measured tokens at the GPT-4o-mini list
price ($0.15 per 1M input tokens, $0.60 per 1M output tokens).

| | All-frontier | Routed |
| --- | --- | --- |
| Tasks passed | 8/10 | 3/10 |
| LLM calls | 58 | 48 |
| Tokens (prompt + completion) | 19,850 | 19,348 |
| Total latency | 1,397.1s (23.3 min) | 589.7s (9.8 min) |
| Total cost | $0.00413 | $0.00050 |
| Escalations | 0 | 0 |
| Retries | 0 | 1 |

Cost fell 88.0 percent and latency fell 57.8 percent on the routed run.
That is real, and it comes with a real cost: accuracy fell from 8/10 to
3/10. The routed run passed t1, t2, and t4. It failed t3, t5, t8, t9, and
t10, all tasks the frontier run passed. t6 and t7 failed in both runs.

The reason is plain. The local 0.5B model gives confident wrong answers
on summarization, writing, and triage. Those answers are valid JSON with
the right shape, so structural validation passes and no retry or
escalation fires. The router catches malformed output. It does not catch
wrong content. The checkers catch wrong content, but only after the fact.
That gap is the main thing v2 has to fix.

What did fire live in this run: one retry. On t2 the local extraction
returned JSON missing a required key, the router retried locally, and the
retry succeeded. No escalation fired in this run. Escalation did fire
live in an earlier run (demo-20261025-225512, t2: local failed twice, then
frontier was called), and unit tests cover the escalation path.

The learned policy (v2) made no changes after this run. Routing,
extraction, and tool calls stayed local; reasoning and writing stayed on
the frontier. The learner only had 0 to 1 local calls of evidence for
summarization and writing, so it kept the defaults. Note the learner
counts a call as a success when the call returns valid output. That is
not the same as the task passing, and this run shows the difference.
Learning from task outcomes instead of call validity is v2 work.

Per-task detail is in `runs/demo-20261025-230447.json`.

## Limits, stated plainly

- Savings depend on the task mix. A task set that is all hard reasoning
  saves nothing, and this demo shows a mixed set honestly.
- Local inference speed depends on the demo machine. Measure on the
  machine you demo on. On a 2-CPU machine the 3B tier is slow; the
  ledger shows the real wall-clock time either way.
- The policy learning in v1 is rule-based on per-kind call outcomes. It
  is a start, not a learned router in the RouteLLM sense.
- The 10-task set is small and built for this demo. It proves the loop
  works; it does not prove the policy generalizes to other agents.

## Prior art

RouteLLM (LMSYS/Berkeley, ICLR 2025), FrugalGPT (Stanford), RouteNLP,
LiteLLM, Martian, NotDiamond, RouteLabs Router, Opod, Sakana RL
Conductor, llama.cpp router mode. SmallFleet's angle is different from
all of them: per tool call inside a running agent loop, with the policy
learned from that agent's own runs. Credit where it is due: the
escalation-and-learn shape owes a debt to FrugalGPT's cascades and to
RouteNLP's closed-loop routing.
