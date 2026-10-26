"""Routing policy, the router, escalation, and policy learning.

The router is the heart of SmallFleet. It sits inside the agent loop and
decides, per LLM call, which model tier handles that call.

Tiers:
  - "local": a small model served on the demo machine (Qwen2.5-0.5B).
  - "frontier": the capable model, used only for hard calls. In this local
    demo it is served by a local Qwen2.5-3B stand-in; the architecture also
    supports a real API backend (see backends.py). Costs for frontier calls
    are computed at the published list price of the named API model, and the
    README says so plainly.

Policy defaults (from the concept): routine kinds go local, hard kinds go
frontier. The policy is learned from the agent's own runs: after each run,
`learn()` looks at the ledger's per-kind call outcomes and moves a kind
between tiers when the measured local success rate says to. The policy file
records the evidence (call counts) behind every change.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import kinds

LOCAL = "local"
FRONTIER = "frontier"

# Default policy: routine kinds local, hard kinds frontier.
DEFAULT_POLICY: dict[str, str] = {
    kinds.ROUTING: LOCAL,
    kinds.EXTRACTION: LOCAL,
    kinds.TOOL_CALL: LOCAL,
    kinds.SUMMARIZATION: LOCAL,
    kinds.REASONING: FRONTIER,
    kinds.WRITING: FRONTIER,
}

# Learning thresholds (v1): measured local call success rate per kind.
LOCAL_KEEP_MIN = 0.85   # at or above: local is fine, keep it local
LOCAL_DROP_MAX = 0.70   # below: local is not reliable enough, move to frontier


@dataclass
class Call:
    """One LLM call the agent wants to make inside its loop."""

    prompt: str
    kind_hint: str | None = None
    expect_json: bool = False
    json_schema: dict | None = None  # e.g. {"required": ["vendor", "total"]}
    max_tokens: int = 256
    temperature: float = 0.0
    task_id: str = ""
    step: str = ""


@dataclass
class RoutingDecision:
    kind: str
    kind_how: str
    tier: str
    reason: str
    policy_version: int


@dataclass
class CallOutcome:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    mean_logprob: float | None
    tier: str
    model: str
    escalated: bool = False
    retries: int = 0
    outcome: str = "ok"  # ok | retried | escalated | failed
    outcome_reason: str = ""


@dataclass
class Policy:
    assignments: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_POLICY))
    version: int = 1
    evidence: dict = field(default_factory=dict)

    def tier_for(self, kind: str) -> str:
        return self.assignments.get(kind, FRONTIER)

    def to_dict(self) -> dict:
        return {"version": self.version, "assignments": self.assignments,
                "evidence": self.evidence}

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        p = cls(assignments=dict(data.get("assignments", DEFAULT_POLICY)),
                version=int(data.get("version", 1)),
                evidence=dict(data.get("evidence", {})))
        for kind in kinds.ALL_KINDS:
            p.assignments.setdefault(kind, DEFAULT_POLICY[kind])
        return p

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Policy":
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


def _validate(data, spec: dict, path: str = "$") -> tuple[bool, str]:
    """Recursively validate parsed JSON against a schema spec.

    Spec shape: {"required": [keys], "fields": {key: subspec},
                 "items": {key: subspec-for-each-list-element}}.
    Returns (True, "") or (False, reason).
    """
    if not isinstance(data, dict):
        return False, f"{path} is not an object"
    for key in spec.get("required", []):
        if key not in data:
            return False, f"{path} missing required key: {key}"
    for key, subspec in spec.get("fields", {}).items():
        if key in data:
            ok, reason = _validate(data[key], subspec, f"{path}.{key}")
            if not ok:
                return False, reason
    for key, subspec in spec.get("items", {}).items():
        if key in data:
            if not isinstance(data[key], list):
                return False, f"{path}.{key} is not a list"
            for i, item in enumerate(data[key]):
                ok, reason = _validate(item, subspec, f"{path}.{key}[{i}]")
                if not ok:
                    return False, reason
    return True, ""


def _json_valid(text: str, schema: dict | None) -> tuple[bool, str]:
    """Check an output parses as JSON and satisfies the schema spec."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"malformed JSON: {exc.msg} at line {exc.lineno}"
    if schema:
        return _validate(data, schema)
    return True, ""


def _extract_json(text: str) -> str | None:
    """Pull the first {...} block out of model output, if any."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


class Router:
    """Routes each agent-loop LLM call to a model tier, with escalation."""

    def __init__(self, policy: Policy, backends: dict[str, object],
                 force_tier: str | None = None,
                 confidence_threshold: float = -2.5):
        self.policy = policy
        self.backends = backends
        self.force_tier = force_tier  # set to FRONTIER for the all-frontier baseline
        self.confidence_threshold = confidence_threshold

    def decide(self, call: Call) -> RoutingDecision:
        kind, how = kinds.classify(call.prompt, call.kind_hint)
        if self.force_tier is not None:
            return RoutingDecision(kind, how, self.force_tier,
                                   f"forced tier for baseline run ({self.force_tier})",
                                   self.policy.version)
        tier = self.policy.tier_for(kind)
        reason = (f"policy v{self.policy.version}: {kind} -> {tier} "
                  f"(kind determined by {how})")
        return RoutingDecision(kind, how, tier, reason, self.policy.version)

    def complete(self, call: Call) -> CallOutcome:
        """Run one agent-loop call through routing, validation, and escalation.

        Flow: route -> call tier -> validate (JSON shape + confidence).
        On a local failure: retry once locally with a repair prompt, then
        escalate to the frontier tier. Never return a silently wrong answer.
        """
        decision = self.decide(call)
        backend = self.backends[decision.tier]
        outcome = self._call_once(backend, call, decision)
        if outcome.outcome == "ok":
            return outcome

        if decision.tier == FRONTIER:
            # Frontier already failed validation; nothing left to escalate to.
            outcome.outcome = "failed"
            return outcome

        # Retry once locally with a repair prompt.
        repair = Call(
            prompt=(call.prompt + "\n\nYour previous output was invalid: "
                    f"{outcome.outcome_reason}. Return ONLY the corrected "
                    "output, nothing else."),
            kind_hint=call.kind_hint, expect_json=call.expect_json,
            json_schema=call.json_schema, max_tokens=call.max_tokens,
            temperature=call.temperature, task_id=call.task_id, step=call.step)
        retry = self._call_once(backend, repair, decision)
        retry.retries = 1
        if retry.outcome == "ok":
            retry.outcome = "retried"
            retry.outcome_reason = ("local retry succeeded after: "
                                    f"{outcome.outcome_reason}")
            return retry

        # Escalate to the frontier tier.
        frontier = self.backends[FRONTIER]
        esc = self._call_once(frontier, call, decision)
        esc.tier = FRONTIER
        esc.escalated = True
        esc.retries = 1
        if esc.outcome == "ok":
            esc.outcome = "escalated"
            esc.outcome_reason = ("local failed twice "
                                  f"({outcome.outcome_reason}); frontier handled it")
        else:
            esc.outcome = "failed"
            esc.outcome_reason = ("local failed twice and frontier also failed "
                                  f"validation: {esc.outcome_reason}")
        return esc

    def _call_once(self, backend, call: Call, decision: RoutingDecision) -> CallOutcome:
        started = time.monotonic()
        result = backend.complete(
            prompt=call.prompt, max_tokens=call.max_tokens,
            temperature=call.temperature)
        latency_ms = (time.monotonic() - started) * 1000.0
        text = result["text"]
        if call.expect_json:
            candidate = _extract_json(text) or text
            valid, reason = _json_valid(candidate, call.json_schema)
            if valid:
                text = candidate
            else:
                return CallOutcome(
                    text=text, prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    latency_ms=latency_ms,
                    mean_logprob=result.get("mean_logprob"),
                    tier=decision.tier, model=result["model"],
                    outcome="invalid", outcome_reason=reason)
        if (result.get("mean_logprob") is not None
                and result["mean_logprob"] < self.confidence_threshold):
            return CallOutcome(
                text=text, prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                latency_ms=latency_ms, mean_logprob=result["mean_logprob"],
                tier=decision.tier, model=result["model"],
                outcome="invalid",
                outcome_reason=(f"low confidence (mean logprob "
                                f"{result['mean_logprob']:.2f} below "
                                f"{self.confidence_threshold})"))
        return CallOutcome(
            text=text, prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            latency_ms=latency_ms, mean_logprob=result.get("mean_logprob"),
            tier=decision.tier, model=result["model"], outcome="ok")


def learn_policy(policy: Policy, ledger, run_id: str) -> tuple[Policy, dict]:
    """Learn the routing policy from the agent's own runs.

    For each kind, look at the ledger's call outcomes for that run: local
    success rate = ok + retried over all local calls of that kind. A kind
    stays local when local calls succeed reliably; it moves to frontier when
    they do not. Kinds with fewer than 3 local calls keep their assignment
    (not enough evidence to move). Returns the updated policy and a report.
    """
    stats = ledger.kind_stats(run_id)
    report: dict = {"run_id": run_id, "changes": [], "kept": []}
    new_assignments = dict(policy.assignments)
    for kind in kinds.ALL_KINDS:
        local = stats.get(kind, {}).get(LOCAL, {"total": 0, "ok": 0})
        total, ok = local["total"], local["ok"]
        if total < 3:
            report["kept"].append(
                {kind: f"not enough evidence ({total} local calls)"})
            continue
        rate = ok / total
        current = new_assignments[kind]
        if rate >= LOCAL_KEEP_MIN and current != LOCAL:
            new_assignments[kind] = LOCAL
            report["changes"].append(
                {kind: f"local success {ok}/{total} ({rate:.0%}) -> move to local"})
        elif rate < LOCAL_DROP_MAX and current != FRONTIER:
            new_assignments[kind] = FRONTIER
            report["changes"].append(
                {kind: f"local success {ok}/{total} ({rate:.0%}) -> move to frontier"})
        else:
            report["kept"].append(
                {kind: f"local success {ok}/{total} ({rate:.0%}), stays {current}"})
    policy.assignments = new_assignments
    policy.version += 1
    policy.evidence = {"learned_from_run": run_id, "kind_stats": stats,
                       "report": report}
    return policy, report
