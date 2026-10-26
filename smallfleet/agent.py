"""The demo agent: a small LangGraph-style agent loop.

The loop has named nodes (plan, compute, extract, summarize, reason,
write, answer). Every LLM call a node makes goes through the router, which
classifies the call by kind and picks the model tier. That is the whole
point of SmallFleet: routing happens per call, inside the agent loop, not
per prompt at the front door.

Control flow is one transition table, shared by two drivers:
  - run_task(): a plain Python loop (used by the demo runner, no deps).
  - build_graph(): the same nodes and transitions as a LangGraph
    StateGraph, for anyone who wants the graph. Same node functions,
    same transitions, same ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from . import config, kinds
from .backends import cost_usd
from .router import Call, Router
from .tasks import Task, check as check_task
from .tools import DOCS, TOOLS


class AgentState(TypedDict, total=False):
    """LangGraph state: node updates merge into this, nothing is lost."""
    task: Any
    observations: list
    recipe_step: int
    last_output: str
    final_answer: str
    done: bool
    next_node: str
    plan_instruction: str
    plan_step: dict


@dataclass
class AgentContext:
    router: Router
    ledger: object  # smallfleet.ledger.Ledger
    run_id: str
    sandbox: str
    policy_version: int


def llm_call(ctx: AgentContext, task_id: str, step: str,
             call: Call) -> tuple[str, dict]:
    """Run one agent-loop LLM call through the router and log it.

    Returns (text, info) where info carries kind, tier, model, outcome,
    tokens, latency and cost for the caller.
    """
    decision = ctx.router.decide(call)
    outcome = ctx.router.complete(call)
    usd = cost_usd(outcome.tier, outcome.prompt_tokens,
                   outcome.completion_tokens,
                   config.FRONTIER_PRICE_PER_1M_INPUT_USD,
                   config.FRONTIER_PRICE_PER_1M_OUTPUT_USD)
    ctx.ledger.log_call(
        run_id=ctx.run_id, task_id=task_id, step=step,
        kind=decision.kind, kind_how=decision.kind_how,
        tier=outcome.tier, model=outcome.model,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        latency_ms=outcome.latency_ms,
        mean_logprob=outcome.mean_logprob, cost_usd=usd,
        outcome=outcome.outcome, outcome_reason=outcome.outcome_reason,
        escalated=outcome.escalated, retries=outcome.retries)
    info = {"kind": decision.kind, "kind_how": decision.kind_how,
            "tier": outcome.tier, "model": outcome.model,
            "outcome": outcome.outcome,
            "outcome_reason": outcome.outcome_reason,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "latency_ms": outcome.latency_ms, "cost_usd": usd,
            "escalated": outcome.escalated, "retries": outcome.retries}
    return outcome.text, info


def _context_block(state: dict) -> str:
    obs = state.get("observations", [])
    if not obs:
        return "Observations so far: none yet."
    return "Observations so far:\n" + "\n".join(f"- {o}" for o in obs[-6:])


def _remaining_steps(state: dict) -> list[dict]:
    task: Task = state["task"]
    return task.recipe[state["recipe_step"]:]


# ---------------------------------------------------------------- nodes

def plan_node(state: dict, ctx: AgentContext) -> dict:
    """Pick the next node. Kind: routing (a decision call, not content)."""
    remaining = _remaining_steps(state)
    if not remaining:
        return {"next_node": "answer", "plan_instruction": "All steps done."}
    menu = "\n".join(
        f"{i + 1}. {s['node']}: {s['instruction']}"
        for i, s in enumerate(remaining))
    prompt = (
        "You are the planner of an agent. Pick the next step.\n"
        f"Task: {state['task'].title}\n"
        f"Remaining steps:\n{menu}\n"
        f"{_context_block(state)}\n"
        "Reply with ONLY a JSON object like "
        '{"next": "compute", "instruction": "Look up document invoice_1042."} '
        "next must be the node of the first remaining step, unless all steps "
        "are done, in which case next must be \"answer\".")
    text, _ = llm_call(ctx, state["task"].id, "plan", Call(
        prompt=prompt, kind_hint=kinds.ROUTING, expect_json=True,
        json_schema={"required": ["next", "instruction"]},
        task_id=state["task"].id, step="plan"))
    try:
        data = json.loads(text)
        nxt = str(data.get("next", "")).strip().lower()
        instruction = str(data.get("instruction", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        nxt, instruction = "", ""
    first = remaining[0]
    if nxt == "answer":
        return {"next_node": "answer",
                "plan_instruction": instruction or "All steps done."}
    # The recipe order is binding: the next step is always the first
    # remaining one. The plan node's real decision is continue vs answer.
    if nxt != first["node"]:
        nxt = first["node"]
        instruction = first["instruction"]
    return {"next_node": nxt,
            "plan_instruction": instruction or first["instruction"],
            "plan_step": first}


def compute_node(state: dict, ctx: AgentContext) -> dict:
    """Produce a tool call (kind: tool_call) and execute it deterministically."""
    step = state.get("plan_step") or {}
    instruction = step.get("instruction") or state.get("plan_instruction", "")
    tool_list = "\n".join(
        f"- {name}: {spec['desc']}" for name, spec in TOOLS.items())
    docs = ", ".join(state["task"].docs) if state["task"].docs else "none"
    prompt = (
        f"Task: {state['task'].prompt}\n"
        f"Instruction: {instruction}\n"
        f"Documents for this task: {docs}\n"
        f"{_context_block(state)}\n"
        f"Available tools:\n{tool_list}\n"
        "Reply with ONLY a JSON object like "
        '{"tool": "lookup_doc", "args": {"doc_id": "invoice_1042"}} '
        "or {\"tool\": \"calculator\", \"args\": {\"expr\": \"1980 - 1420\"}}.")
    text, _ = llm_call(ctx, state["task"].id, "compute", Call(
        prompt=prompt, kind_hint=kinds.TOOL_CALL, expect_json=True,
        json_schema={"required": ["tool", "args"]},
        task_id=state["task"].id, step="compute"))
    try:
        data = json.loads(text)
        tool = str(data.get("tool", ""))
        args = dict(data.get("args", {}))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {"observations": state["observations"] + [
            "compute: invalid tool-call JSON, could not run a tool."],
            "last_output": "", "recipe_step": state["recipe_step"]}
    if tool not in TOOLS:
        return {"observations": state["observations"] + [
            f"compute: unknown tool {tool!r}, nothing ran."],
            "last_output": "", "recipe_step": state["recipe_step"]}
    try:
        if tool == "calculator":
            result = TOOLS["calculator"]["fn"](str(args["expr"]))
        elif tool == "lookup_doc":
            result = TOOLS["lookup_doc"]["fn"](str(args["doc_id"]))
        elif tool == "search_notes":
            result = TOOLS["search_notes"]["fn"](str(args["query"]))
        elif tool == "write_file":
            from .tools import write_file
            result = write_file(str(args["path"]), str(args["content"]),
                                ctx.sandbox)
        else:  # pragma: no cover
            result = "unknown tool"
    except Exception as exc:  # noqa: BLE001 - tool errors are observations
        result = f"tool error: {exc}"
    obs = f"{tool}({json.dumps(args)}) = {result}"
    return {"observations": state["observations"] + [obs],
            "last_output": result,
            "recipe_step": state["recipe_step"] + 1}


def extract_node(state: dict, ctx: AgentContext) -> dict:
    """Extract structured fields (kind: extraction, strict JSON)."""
    step = state.get("plan_step") or {}
    instruction = step.get("instruction") or state.get("plan_instruction", "")
    schema = step.get("json_schema")
    prompt = (
        f"Task: {state['task'].prompt}\n"
        f"Instruction: {instruction}\n"
        f"{_context_block(state)}\n"
        "Reply with ONLY the JSON object, no other text.")
    text, _ = llm_call(ctx, state["task"].id, "extract", Call(
        prompt=prompt, kind_hint=kinds.EXTRACTION, expect_json=True,
        json_schema=schema, max_tokens=512,
        task_id=state["task"].id, step="extract"))
    return {"observations": state["observations"] + [f"extracted: {text}"],
            "last_output": text,
            "recipe_step": state["recipe_step"] + 1}


def summarize_node(state: dict, ctx: AgentContext) -> dict:
    """Summarize (kind: summarization)."""
    step = state.get("plan_step") or {}
    instruction = step.get("instruction") or state.get("plan_instruction", "")
    prompt = (
        f"Task: {state['task'].prompt}\n"
        f"Instruction: {instruction}\n"
        f"{_context_block(state)}\n"
        "Reply with only the summary, no preamble.")
    text, _ = llm_call(ctx, state["task"].id, "summarize", Call(
        prompt=prompt, kind_hint=kinds.SUMMARIZATION,
        task_id=state["task"].id, step="summarize"))
    return {"observations": state["observations"] + [f"summary: {text}"],
            "last_output": text,
            "recipe_step": state["recipe_step"] + 1}


def reason_node(state: dict, ctx: AgentContext) -> dict:
    """Multi-step reasoning (kind: reasoning)."""
    step = state.get("plan_step") or {}
    instruction = step.get("instruction") or state.get("plan_instruction", "")
    prompt = (
        f"Task: {state['task'].prompt}\n"
        f"Instruction: {instruction}\n"
        f"{_context_block(state)}\n"
        "Think step by step, then give the conclusion clearly.")
    text, _ = llm_call(ctx, state["task"].id, "reason", Call(
        prompt=prompt, kind_hint=kinds.REASONING, max_tokens=320,
        task_id=state["task"].id, step="reason"))
    return {"observations": state["observations"] + [f"reasoning: {text}"],
            "last_output": text,
            "recipe_step": state["recipe_step"] + 1}


def write_node(state: dict, ctx: AgentContext) -> dict:
    """Draft text (kind: writing)."""
    step = state.get("plan_step") or {}
    instruction = step.get("instruction") or state.get("plan_instruction", "")
    prompt = (
        f"Task: {state['task'].prompt}\n"
        f"Instruction: {instruction}\n"
        f"{_context_block(state)}\n"
        "Reply with only the drafted text, no preamble.")
    text, _ = llm_call(ctx, state["task"].id, "write", Call(
        prompt=prompt, kind_hint=kinds.WRITING, max_tokens=320,
        task_id=state["task"].id, step="write"))
    return {"observations": state["observations"] + [f"draft: {text}"],
            "last_output": text,
            "recipe_step": state["recipe_step"] + 1}


def answer_node(state: dict, ctx: AgentContext) -> dict:
    """Assemble the final answer from the last produced output."""
    return {"final_answer": state.get("last_output", ""),
            "done": True}


NODES = {
    "plan": plan_node,
    "compute": compute_node,
    "extract": extract_node,
    "summarize": summarize_node,
    "reason": reason_node,
    "write": write_node,
    "answer": answer_node,
}


def run_task(ctx: AgentContext, task: Task) -> tuple[str, bool, str]:
    """Run one task through the loop. Returns (answer, passed, notes)."""
    state: dict = {"task": task, "observations": [], "recipe_step": 0,
                   "last_output": "", "final_answer": "", "done": False,
                   "next_node": "plan", "plan_instruction": "",
                   "plan_step": {}}
    steps = 0
    while not state["done"] and steps < config.MAX_AGENT_STEPS:
        node_name = state["next_node"]
        node = NODES[node_name]
        updates = node(state, ctx)
        state.update(updates)
        steps += 1
        if node_name == "answer":
            break
        if node_name != "plan":
            state["next_node"] = "plan"
    answer = state.get("final_answer", "") or state.get("last_output", "")
    passed, notes = check_task(task, answer, ctx.sandbox)
    return answer, passed, notes


def build_graph(ctx: AgentContext):
    """The same nodes and transitions as a LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    def _wrap(name):
        def _node(state: dict) -> dict:
            return NODES[name](state, ctx)
        return _node

    def _route(state: dict) -> str:
        return state.get("next_node", "answer")

    graph = StateGraph(AgentState)
    for name in NODES:
        graph.add_node(name, _wrap(name))
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _route, {n: n for n in NODES})
    for name in ("compute", "extract", "summarize", "reason", "write"):
        graph.add_edge(name, "plan")
    graph.add_edge("answer", END)
    return graph.compile()
