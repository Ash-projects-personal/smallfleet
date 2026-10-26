"""Tests for SmallFleet core: classifier, router escalation, ledger, learning.

Run:  python -m pytest tests/  (or: python tests/test_smallfleet.py)
No model files needed; backends are faked.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smallfleet import kinds
from smallfleet.backends import cost_usd
from smallfleet.kinds import classify
from smallfleet.ledger import Ledger
from smallfleet.router import (Call, Policy, Router, learn_policy)
from smallfleet.tasks import check as check_task, task_list


class FakeBackend:
    """A backend that returns canned text per call count."""

    def __init__(self, name: str, script: list[dict]):
        self.name = name
        self.script = script
        self.calls = 0

    def complete(self, prompt, max_tokens=256, temperature=0.0):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return {"text": item["text"],
                "prompt_tokens": item.get("prompt_tokens", 10),
                "completion_tokens": item.get("completion_tokens", 5),
                "mean_logprob": item.get("mean_logprob", -0.5),
                "model": self.name}


def test_classify_hint_and_score():
    kind, how = classify("Extract the vendor and total from this invoice.",
                         hint=kinds.EXTRACTION)
    assert kind == kinds.EXTRACTION and how in ("hint", "hint+score")
    kind, how = classify("Summarize this memo in two sentences.")
    assert kind == kinds.SUMMARIZATION and how == "score"
    kind, how = classify("hello there")
    assert kind == kinds.TOOL_CALL and how == "default"


def test_router_escalates_after_local_json_failure():
    local = FakeBackend("local-0.5B", [
        {"text": "not json at all"},
        {"text": "still not json {"},
    ])
    frontier = FakeBackend("frontier-3B", [
        {"text": '{"vendor": "X", "total": 1.0}'},
    ])
    router = Router(Policy(), {"local": local, "frontier": frontier})
    outcome = router.complete(Call(
        prompt="Extract vendor and total. Return ONLY JSON.",
        kind_hint=kinds.EXTRACTION, expect_json=True,
        json_schema={"required": ["vendor", "total"]}))
    assert outcome.outcome == "escalated", outcome
    assert outcome.escalated is True
    assert outcome.tier == "frontier"
    assert local.calls == 2  # first try + one retry
    assert frontier.calls == 1


def test_router_local_success_no_escalation():
    local = FakeBackend("local-0.5B", [
        {"text": '{"vendor": "X", "total": 1.0}'},
    ])
    frontier = FakeBackend("frontier-3B", [{"text": "{}"}])
    router = Router(Policy(), {"local": local, "frontier": frontier})
    outcome = router.complete(Call(
        prompt="Extract vendor and total. Return ONLY JSON.",
        kind_hint=kinds.EXTRACTION, expect_json=True,
        json_schema={"required": ["vendor", "total"]}))
    assert outcome.outcome == "ok"
    assert outcome.tier == "local"
    assert frontier.calls == 0


def test_router_low_confidence_escalates():
    local = FakeBackend("local-0.5B", [
        {"text": "some answer", "mean_logprob": -4.0},
        {"text": "some answer", "mean_logprob": -4.0},
    ])
    frontier = FakeBackend("frontier-3B", [
        {"text": "some answer", "mean_logprob": -0.2},
    ])
    router = Router(Policy(), {"local": local, "frontier": frontier},
                    confidence_threshold=-2.5)
    outcome = router.complete(Call(prompt="Summarize this.",
                                   kind_hint=kinds.SUMMARIZATION))
    assert outcome.outcome == "escalated"


def test_ledger_totals_and_kind_stats():
    with tempfile.TemporaryDirectory() as d:
        ledger = Ledger(os.path.join(d, "test.db"))
        ledger.new_run("r1", "routed", 1, "test prices")
        ledger.log_call("r1", "t1", "extract", "extraction", "hint",
                        "local", "local-0.5B", 10, 5, 100.0, -0.5,
                        0.0, "ok", "", False, 0)
        ledger.log_call("r1", "t1", "extract", "extraction", "hint",
                        "local", "local-0.5B", 10, 5, 100.0, -0.5,
                        0.0, "escalated", "bad json", True, 1)
        ledger.log_task_result("r1", "t1", True, "ok")
        totals = ledger.run_totals("r1")
        assert totals["calls"] == 2
        assert totals["escalations"] == 1
        assert totals["tasks_passed"] == 1
        stats = ledger.kind_stats("r1")
        assert stats["extraction"]["local"]["total"] == 2
        assert stats["extraction"]["local"]["ok"] == 1


def test_learn_policy_moves_unreliable_kind_to_frontier():
    with tempfile.TemporaryDirectory() as d:
        ledger = Ledger(os.path.join(d, "test.db"))
        ledger.new_run("r1", "routed", 1, "test prices")
        # extraction: 1 ok out of 4 local calls -> below 0.70, moves to frontier
        for outcome in ("ok", "escalated", "failed", "escalated"):
            ledger.log_call("r1", "t2", "extract", "extraction", "hint",
                            "local", "local-0.5B", 10, 5, 100.0, -0.5,
                            0.0, outcome, "", outcome == "escalated", 0)
        # summarization: 3 ok out of 3 -> stays local
        for _ in range(3):
            ledger.log_call("r1", "t5", "summarize", "summarization", "hint",
                            "local", "local-0.5B", 10, 5, 100.0, -0.5,
                            0.0, "ok", "", False, 0)
        policy = Policy()
        assert policy.assignments["extraction"] == "local"
        policy, report = learn_policy(policy, ledger, "r1")
        assert policy.assignments["extraction"] == "frontier"
        assert policy.assignments["summarization"] == "local"
        assert policy.version == 2


def test_cost_usd():
    assert cost_usd("local", 1000, 500, 0.15, 0.60) == 0.0
    assert abs(cost_usd("frontier", 1_000_000, 1_000_000, 0.15, 0.60)
               - 0.75) < 1e-9


def test_task_checkers_known_answers():
    tasks = {t.id: t for t in task_list()}
    assert len(tasks) == 10
    ok, _ = check_task(tasks["t1"],
                       '{"vendor": "Northwind Office Supply", '
                       '"date": "2026-09-14", "total": 427.68}', "")
    assert ok
    ok, _ = check_task(tasks["t4"], "The total is 86.45.", "")
    assert ok
    ok, _ = check_task(tasks["t6"],
                       "Supplier B is cheapest at $1,420/month. "
                       "Savings versus the current supplier: $560/month.", "")
    assert ok
    ok, _ = check_task(tasks["t1"],
                       '{"vendor": "Wrong", "date": "2026-09-14", '
                       '"total": 427.68}', "")
    assert not ok


def test_langgraph_build_and_run():
    """The same nodes run as a LangGraph StateGraph (fake backends)."""
    import tempfile

    from smallfleet.agent import AgentContext, build_graph
    from smallfleet.ledger import Ledger

    script = [
        {"text": '{"next": "compute", "instruction": "Look up amounts_t4."}'},
        {"text": '{"tool": "lookup_doc", "args": {"doc_id": "amounts_t4"}}'},
        {"text": '{"next": "compute", "instruction": "Add them up."}'},
        {"text": '{"tool": "calculator", "args": {"expr": "18.50 + 12.75 + 46.20 + 9.00"}}'},
        {"text": '{"next": "answer", "instruction": "Done."}'},
    ]
    backend = FakeBackend("fake-local", script)
    router = Router(Policy(), {"local": backend, "frontier": backend})
    with tempfile.TemporaryDirectory() as d:
        ledger = Ledger(os.path.join(d, "test.db"))
        ledger.new_run("g1", "routed", 1, "test")
        ctx = AgentContext(router=router, ledger=ledger, run_id="g1",
                           sandbox=d, policy_version=1)
        graph = build_graph(ctx)
        task = next(t for t in task_list() if t.id == "t4")
        final = graph.invoke({"task": task, "observations": [],
                              "recipe_step": 0, "last_output": "",
                              "final_answer": "", "done": False,
                              "next_node": "plan", "plan_instruction": "",
                              "plan_step": {}})
        assert final.get("done") is True
        assert "86.45" in final.get("final_answer", "")
        ledger.log_task_result("g1", "t4", True, "graph run ok")
        assert ledger.run_totals("g1")["calls"] == 5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
