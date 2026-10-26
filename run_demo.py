"""Demo runner: the same 10-task agent job, twice.

Run 1 (all-frontier): every agent-loop call goes to the frontier tier.
Run 2 (routed): the router decides per call, with escalation.
Same tasks, same agent logic, same temperature. The ledger proves the
difference: cost, latency, accuracy, and every routing decision.

After the routed run, the policy is learned from the ledger and saved:
the router updates its per-kind assignments from the agent's own runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smallfleet import config
from smallfleet.agent import AgentContext, run_task
from smallfleet.backends import LlamaCppBackend, OpenAICompatibleBackend
from smallfleet.ledger import Ledger
from smallfleet.router import Policy, Router, learn_policy
from smallfleet.tasks import task_list


def build_backends() -> dict:
    """Local tier: Qwen2.5-0.5B. Frontier tier: API if a key exists,
    otherwise the local Qwen2.5-3B stand-in (disclosed in the README)."""
    backends = {}
    backends["local"] = LlamaCppBackend(
        config.LOCAL_MODEL_PATH, config.LOCAL_MODEL_NAME)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        backends["frontier"] = OpenAICompatibleBackend(
            "https://api.openai.com/v1", api_key, "gpt-4o-mini",
            name="gpt-4o-mini (API)")
        print("frontier tier: gpt-4o-mini via API")
    else:
        backends["frontier"] = LlamaCppBackend(
            config.FRONTIER_MODEL_PATH, config.FRONTIER_MODEL_NAME)
        print("frontier tier: local Qwen2.5-3B stand-in "
              "(no API key; costs computed at GPT-4o-mini list price)")
    return backends


def run_job(backends: dict, ledger: Ledger, mode: str,
            policy: Policy, run_id: str,
            task_ids: list[str] | None = None) -> dict:
    force_tier = "frontier" if mode == "all-frontier" else None
    router = Router(policy, backends, force_tier=force_tier,
                    confidence_threshold=config.CONFIDENCE_THRESHOLD)
    ledger.new_run(run_id, mode, policy.version,
                   config.FRONTIER_PRICE_NOTE)
    sandbox = os.path.join(config.BASE_DIR, "runs", run_id, "sandbox")
    os.makedirs(sandbox, exist_ok=True)
    ctx = AgentContext(router=router, ledger=ledger, run_id=run_id,
                       sandbox=sandbox, policy_version=policy.version)
    print(f"\n=== run {run_id} ({mode}) ===")
    jobs = [t for t in task_list() if task_ids is None or t.id in task_ids]
    for task in jobs:
        started = time.monotonic()
        answer, passed, notes = run_task(ctx, task)
        secs = time.monotonic() - started
        ledger.log_task_result(run_id, task.id, passed, notes)
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {task.id} {task.title} ({secs:.1f}s) :: {notes[:100]}")
    ledger.finish_run(run_id)
    totals = ledger.run_totals(run_id)
    print(f"  totals: {totals['tasks_passed']}/{totals['tasks_total']} tasks, "
          f"{totals['calls']} calls, ${totals['cost_usd']:.4f}, "
          f"{totals['latency_ms'] / 1000:.1f}s, "
          f"{totals['escalations']} escalations, {totals['retries']} retries")
    return totals


def print_comparison(comp: dict) -> None:
    a, b = comp["run_a"], comp["run_b"]
    print("\n=== comparison (same 10 tasks, same agent logic) ===")
    print(f"{'':28} {'all-frontier':>14} {'routed':>14}")
    print(f"{'tasks passed':28} {a['tasks_passed']}/{a['tasks_total']:>11} "
          f"{b['tasks_passed']}/{b['tasks_total']:>11}")
    print(f"{'LLM calls':28} {a['calls']:>14} {b['calls']:>14}")
    print(f"{'total cost (USD)':28} ${a['cost_usd']:>13.4f} ${b['cost_usd']:>13.4f}")
    print(f"{'total latency (s)':28} {a['latency_ms'] / 1000:>14.1f} "
          f"{b['latency_ms'] / 1000:>14.1f}")
    print(f"{'escalations':28} {a['escalations']:>14} {b['escalations']:>14}")
    if a['cost_usd'] > 0:
        saved = 100 * (1 - b['cost_usd'] / a['cost_usd'])
        print(f"routed run cost {saved:.1f}% less than all-frontier "
              f"(list-price math, see README)")
    print("\nper-task cost:")
    pa = {r["task_id"]: r for r in comp["per_task_a"]}
    pb = {r["task_id"]: r for r in comp["per_task_b"]}
    for tid in sorted(pa):
        ra, rb = pa[tid], pb[tid]
        print(f"  {tid}: all-frontier ${ra['cost_usd']:.4f} "
              f"({'PASS' if ra['passed'] else 'FAIL'}) | "
              f"routed ${rb['cost_usd']:.4f} "
              f"({'PASS' if rb['passed'] else 'FAIL'}) "
              f"esc={rb['escalations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SmallFleet demo runner")
    parser.add_argument("--mode", choices=["both", "all-frontier", "routed"],
                        default="both")
    parser.add_argument("--learn", action="store_true",
                        help="learn the policy from the routed run and save it")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task ids to run (default: all 10)")
    args = parser.parse_args()
    task_ids = ([t.strip() for t in args.tasks.split(",")]
                if args.tasks else None)

    os.makedirs(os.path.join(config.BASE_DIR, "runs"), exist_ok=True)
    try:
        backends = build_backends()
    except FileNotFoundError as exc:
        print(f"cannot run demo: {exc}")
        print("models not downloaded, run models/download_models.sh")
        sys.exit(2)
    policy = Policy.load(config.POLICY_PATH)
    print(f"policy v{policy.version}: {policy.assignments}")
    ledger = Ledger(config.LEDGER_PATH)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_ids: dict[str, str] = {}

    if args.mode in ("both", "all-frontier"):
        run_ids["all-frontier"] = f"demo-{stamp}-frontier"
        run_job(backends, ledger, "all-frontier", policy,
                run_ids["all-frontier"], task_ids)
    if args.mode in ("both", "routed"):
        run_ids["routed"] = f"demo-{stamp}-routed"
        run_job(backends, ledger, "routed", policy, run_ids["routed"],
                task_ids)

    report: dict = {"stamp": stamp, "run_ids": run_ids}
    if len(run_ids) == 2:
        comp = ledger.comparison(run_ids["all-frontier"], run_ids["routed"])
        print_comparison(comp)
        report["comparison"] = comp
    if args.learn and "routed" in run_ids:
        policy, learn_report = learn_policy(policy, ledger, run_ids["routed"])
        policy.save(config.POLICY_PATH)
        print(f"\nlearned policy v{policy.version}: {policy.assignments}")
        print(f"learning report: {json.dumps(learn_report, indent=2)}")
        report["learned_policy"] = policy.to_dict()
    out = os.path.join(config.BASE_DIR, "runs", f"demo-{stamp}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
