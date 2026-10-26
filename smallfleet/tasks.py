"""The demo task set: 10 mixed tasks, each machine-checkable.

Each task names the documents it needs and the steps the agent should take
(a recipe the plan node follows). The checker is deterministic: it reads
the agent's final answer and the run sandbox and returns pass/fail with a
note. No LLM judges anything.

T2 is the escalation task by design: extracting nested line items with
quotes and commas into strict JSON is exactly where a 0.5B model fumbles.
The router is expected to catch the malformed output and escalate.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    title: str
    prompt: str
    docs: list[str] = field(default_factory=list)
    recipe: list[dict] = field(default_factory=list)  # [{"node":..., "instruction":...}]
    checker: str = ""  # name of the checker function


def _as_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _num(v) -> float:
    """Parse a number that may arrive as 427.68, '$427.68', or '427.68'."""
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def _check_t1(answer: str, sandbox: str) -> tuple[bool, str]:
    d = _as_json(answer)
    if d is None:
        return False, "final answer is not valid JSON"
    ok = (d.get("vendor") == "Northwind Office Supply"
          and str(d.get("date")) in ("2026-09-14", "Sep 14, 2026", "14 Sep 2026")
          and abs(_num(d.get("total")) - 427.68) < 0.01)
    return ok, f"extracted {d}"


def _check_t2(answer: str, sandbox: str) -> tuple[bool, str]:
    d = _as_json(answer)
    if d is None:
        return False, "final answer is not valid JSON"
    items = d.get("line_items")
    billing = d.get("billing")

    def _rate(v) -> float:
        try:
            return float(str(v).replace("%", "").strip())
        except (TypeError, ValueError):
            return float("nan")

    ok_items = (isinstance(items, list) and len(items) == 4 and all(
        isinstance(i, dict) and isinstance(i.get("tax"), dict)
        and _rate(i["tax"].get("rate")) in (8.0, 0.08)
        and not math.isnan(_num(i["tax"].get("amount")))
        for i in items))
    ok_billing = (isinstance(billing, dict)
                  and str(billing.get("name", "")).startswith("Acme Corp"))
    ok = (d.get("vendor") == "Bright & Sons, LLC"
          and abs(_num(d.get("total")) - 205.72) < 0.01
          and ok_items and ok_billing)
    return ok, f"extracted {d}"


def _check_t3(answer: str, sandbox: str) -> tuple[bool, str]:
    d = _as_json(answer)
    if d is None:
        return False, "final answer is not valid JSON"
    items = d.get("triage") if isinstance(d, dict) else None
    if not isinstance(items, list) or len(items) != 3:
        return False, f"expected 3 triage entries, got {d}"
    queues = {str(i.get("id", "")).upper(): str(i.get("queue", "")).lower()
              for i in items if isinstance(i, dict)}
    ok = (queues.get("M1") == "billing" and queues.get("M2") == "engineering"
          and queues.get("M3") == "support")
    return ok, f"queues {queues}"


def _check_t4(answer: str, sandbox: str) -> tuple[bool, str]:
    m = re.findall(r"86\.45", answer)
    return bool(m), f"answer: {answer[:200]}"


def _check_t5(answer: str, sandbox: str) -> tuple[bool, str]:
    low = answer.lower()
    has_throughput = "12%" in low or "throughput" in low
    has_hiring = "four of six" in low or "hiring" in low
    sentences = [s for s in re.split(r"[.!?]\s+", answer.strip()) if s.strip()]
    ok = has_throughput and has_hiring and len(sentences) <= 3
    return ok, f"{len(sentences)} sentences, throughput={has_throughput}, hiring={has_hiring}"


def _check_t6(answer: str, sandbox: str) -> tuple[bool, str]:
    low = answer.lower()
    has_b = "supplier b" in low
    has_savings = "560" in low
    return bool(has_b and has_savings), f"answer: {answer[:200]}"


def _check_t7(answer: str, sandbox: str) -> tuple[bool, str]:
    low = answer.lower()
    ok = ("subject" in low and "status" in low
          and ("warehouse" in low or "q3" in low))
    return ok, f"answer: {answer[:200]}"


def _check_t8(answer: str, sandbox: str) -> tuple[bool, str]:
    d = _as_json(answer)
    if d is None:
        return False, "final answer is not valid JSON"
    items = d.get("items") if isinstance(d, dict) else None
    if not isinstance(items, list) or len(items) != 4:
        return False, f"expected 4 items, got {d}"
    owners = {str(i.get("owner", "")).lower() for i in items if isinstance(i, dict)}
    ok = owners == {"priya", "marcus", "dana"}
    return ok, f"owners {owners}"


def _check_t9(answer: str, sandbox: str) -> tuple[bool, str]:
    low = answer.lower()
    picks_b = ("plan b" in low and "better" in low) or "move" in low
    has_math = "1,900" in low or "1900" in low
    return bool(picks_b and has_math), f"answer: {answer[:200]}"


def _check_t10(answer: str, sandbox: str) -> tuple[bool, str]:
    low = answer.lower()
    ok = ("tracking" in low and "text" in low and "checkout" in low)
    return ok, f"answer: {answer[:200]}"


CHECKERS = {
    "t1": _check_t1, "t2": _check_t2, "t3": _check_t3, "t4": _check_t4,
    "t5": _check_t5, "t6": _check_t6, "t7": _check_t7, "t8": _check_t8,
    "t9": _check_t9, "t10": _check_t10,
}


def task_list() -> list[Task]:
    return [
        Task(
            id="t1", title="Extract invoice fields (routine)",
            prompt=("Extract the vendor, invoice date, and total due from "
                    "invoice_1042. Return ONLY a JSON object with keys "
                    "vendor, date, total."),
            docs=["invoice_1042"],
            recipe=[
                {"node": "compute", "instruction": "Look up document invoice_1042 with lookup_doc."},
                {"node": "extract", "instruction": "Extract vendor, date, total from the invoice text into JSON with keys vendor, date, total.", "json_schema": {"required": ["vendor", "date", "total"]}},
                {"node": "answer", "instruction": "Return the extracted JSON as the final answer."},
            ],
            checker="t1",
        ),
        Task(
            id="t2", title="Extract messy invoice with nested items (hard)",
            prompt=("Extract the vendor, invoice date, total due, all line "
                    "items, and the billing block from invoice_1043. Return "
                    "ONLY a JSON object with keys vendor, date, total, "
                    "line_items, billing, where line_items is a list of "
                    "objects with keys description, quantity, unit_price, "
                    "amount, tax (tax is an object with keys rate, amount; "
                    "the tax rate is 8%), and billing is an object with "
                    "keys name, address, terms."),
            docs=["invoice_1043"],
            recipe=[
                {"node": "compute", "instruction": "Look up document invoice_1043 with lookup_doc."},
                {"node": "extract", "instruction": "Extract vendor, date, total, all 4 line items with nested tax objects (8% rate), and the billing block from the invoice text into strict JSON with keys vendor, date, total, line_items, billing.", "json_schema": {"required": ["vendor", "date", "total", "line_items", "billing"], "fields": {"billing": {"required": ["name", "address", "terms"]}}, "items": {"line_items": {"required": ["description", "quantity", "unit_price", "amount", "tax"], "fields": {"tax": {"required": ["rate", "amount"]}}}}}},
                {"node": "answer", "instruction": "Return the extracted JSON as the final answer."},
            ],
            checker="t2",
        ),
        Task(
            id="t3", title="Triage three incoming messages",
            prompt=("Read the three messages in messages_triage and assign "
                    "each to exactly one queue: billing, engineering, or "
                    "support. Return ONLY a JSON object with key triage, a "
                    "list of objects with keys id, queue, reason."),
            docs=["messages_triage"],
            recipe=[
                {"node": "compute", "instruction": "Look up document messages_triage with lookup_doc."},
                {"node": "extract", "instruction": "Assign each message M1, M2, M3 to one queue (billing, engineering, support) with a one-line reason, as JSON with key triage.", "json_schema": {"required": ["triage"]}},
                {"node": "answer", "instruction": "Return the triage JSON as the final answer."},
            ],
            checker="t3",
        ),
        Task(
            id="t4", title="Total the expenses with the calculator",
            prompt=("Add up the four expenses in amounts_t4 and report the "
                    "total. Use the calculator tool for the addition."),
            docs=["amounts_t4"],
            recipe=[
                {"node": "compute", "instruction": "Look up document amounts_t4 with lookup_doc."},
                {"node": "compute", "instruction": "Add 18.50 + 12.75 + 46.20 + 9.00 with the calculator tool."},
                {"node": "answer", "instruction": "Report the total as the final answer."},
            ],
            checker="t4",
        ),
        Task(
            id="t5", title="Summarize the Q3 memo in two sentences",
            prompt=("Summarize memo_q3 in at most two sentences. Keep the "
                    "throughput gain and the hiring status."),
            docs=["memo_q3"],
            recipe=[
                {"node": "compute", "instruction": "Look up document memo_q3 with lookup_doc."},
                {"node": "summarize", "instruction": "Summarize the memo in at most two sentences, keeping the throughput gain and the hiring status."},
                {"node": "answer", "instruction": "Return the summary as the final answer."},
            ],
            checker="t5",
        ),
        Task(
            id="t6", title="Cheapest supplier and monthly savings",
            prompt=("Find the cheapest supplier in quotes_suppliers and "
                    "compute how much per month we save versus the current "
                    "supplier. Use the calculator tool. Report the supplier "
                    "name and the monthly savings."),
            docs=["quotes_suppliers"],
            recipe=[
                {"node": "compute", "instruction": "Look up document quotes_suppliers with lookup_doc."},
                {"node": "reason", "instruction": "Compare the four monthly prices, pick the cheapest supplier, and say which calculator step computes the savings versus the current supplier at $1,980/month."},
                {"node": "compute", "instruction": "Compute 1980 - 1420 with the calculator tool."},
                {"node": "write", "instruction": "Write one sentence reporting the cheapest supplier and the monthly savings versus the current supplier."},
                {"node": "answer", "instruction": "Report the cheapest supplier and the monthly savings as the final answer."},
            ],
            checker="t6",
        ),
        Task(
            id="t7", title="Draft a status update email",
            prompt=("Draft a short status update email about the Q3 memo. "
                    "Include a subject line. Keep it under 80 words."),
            docs=["memo_q3"],
            recipe=[
                {"node": "compute", "instruction": "Look up document memo_q3 with lookup_doc."},
                {"node": "write", "instruction": "Draft a short status update email with a subject line, under 80 words, based on the memo."},
                {"node": "answer", "instruction": "Return the drafted email as the final answer."},
            ],
            checker="t7",
        ),
        Task(
            id="t8", title="Extract dates and owners from meeting notes",
            prompt=("From meeting_notes, extract every item with its owner "
                    "and due date. Return ONLY a JSON object with key items, "
                    "a list of objects with keys item, owner, due."),
            docs=["meeting_notes"],
            recipe=[
                {"node": "compute", "instruction": "Look up document meeting_notes with lookup_doc."},
                {"node": "extract", "instruction": "Extract all 4 items with owner and due date into JSON with key items.", "json_schema": {"required": ["items"]}},
                {"node": "answer", "instruction": "Return the extracted JSON as the final answer."},
            ],
            checker="t8",
        ),
        Task(
            id="t9", title="Compare two warehouse plans",
            prompt=("Compare plan_a and plan_b. Which plan is better for "
                    "throughput, and what does it cost per month? Show the "
                    "throughput numbers in your answer."),
            docs=["plan_a", "plan_b"],
            recipe=[
                {"node": "compute", "instruction": "Look up document plan_a with lookup_doc."},
                {"node": "compute", "instruction": "Look up document plan_b with lookup_doc."},
                {"node": "reason", "instruction": "Compare the two plans on throughput and monthly cost, and decide which is better for throughput."},
                {"node": "answer", "instruction": "Report the better plan, its throughput, and its monthly cost as the final answer."},
            ],
            checker="t9",
        ),
        Task(
            id="t10", title="Summarize feedback and draft a response",
            prompt=("Summarize the customer feedback in feedback_doc in one "
                    "sentence, then draft a one-paragraph reply that "
                    "mentions the checkout issue and the text-message "
                    "tracking request."),
            docs=["feedback_doc"],
            recipe=[
                {"node": "compute", "instruction": "Look up document feedback_doc with lookup_doc."},
                {"node": "summarize", "instruction": "Summarize the feedback in one sentence."},
                {"node": "write", "instruction": "Draft a one-paragraph reply mentioning the mobile checkout freeze and the text-message tracking request."},
                {"node": "answer", "instruction": "Return the summary sentence and the drafted reply as the final answer."},
            ],
            checker="t10",
        ),
    ]


def check(task: Task, answer: str, sandbox: str) -> tuple[bool, str]:
    return CHECKERS[task.checker](answer, sandbox)
