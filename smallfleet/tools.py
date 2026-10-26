"""Deterministic tools for the demo agent.

These tools are plain Python functions with exact outputs. The agent's LLM
calls decide *which* tool to call and with what arguments; the tools
themselves never guess. That keeps every task machine-checkable.
"""

from __future__ import annotations

import ast
import os


def calculator(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
               ast.Mod, ast.Pow, ast.FloorDiv)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported expression: {expr!r}")
        if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float)):
            raise ValueError(f"unsupported constant in: {expr!r}")
    return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}))


DOCS = {
    "invoice_1042": (
        "INVOICE #1042\n"
        "Vendor: Northwind Office Supply\n"
        "Invoice date: 2026-09-14\n"
        "Bill to: Acme Corp, 400 Main St\n"
        "Items:\n"
        "  - Paper, A4 ream x 10 @ $6.50 = $65.00\n"
        "  - Pens, box x 4 @ $8.25 = $33.00\n"
        "  - Desk chairs x 2 @ $149.00 = $298.00\n"
        "Subtotal: $396.00\n"
        "Tax (8%): $31.68\n"
        "Total due: $427.68\n"
        "Payment terms: Net 30\n"
    ),
    "invoice_1043": (
        "INVOICE #1043 (\"rush\" order, per client email \"please hurry, the "
        "event is Friday\")\n"
        "Vendor: Bright & Sons, LLC\n"
        "Invoice date: 2026-10-02\n"
        "Bill to: Acme Corp, 400 Main St, Suite 9, Denton, TX 76207.\n"
        "Terms: Net 30, \"pay within 30 days, no exceptions\".\n"
        "Line items:\n"
        "  1) \"Widget, deluxe\" (2 units @ $45.99) = $91.98\n"
        "  2) Cable, 6ft (5 units @ $7.50) = $37.50\n"
        "  3) \"Adapter, USB-C\" (3 units @ $12.00) = $36.00\n"
        "  4) Rush fee (flat) = $25.00\n"
        "Subtotal: $190.48\n"
        "Tax (8%): $15.24\n"
        "Total due: $205.72\n"
        "Note: the client asked \"can you waive the rush fee?\" - no, fee stands.\n"
    ),
    "memo_q3": (
        "MEMO: Q3 operations update\n"
        "Paragraph 1: Warehouse throughput rose 12% in Q3 after the new "
        "sorting line went live in August. Backorders fell from 340 to 210.\n"
        "Paragraph 2: Two suppliers missed delivery windows in September, "
        "which cost us three rush shipments. The team has added a second "
        "source for packaging material.\n"
        "Paragraph 3: Hiring is on track. Four of six open roles are filled, "
        "and the night shift is fully staffed for the first time this year.\n"
    ),
    "quotes_suppliers": (
        "SUPPLIER QUOTES (packaging material, per month)\n"
        "Supplier A: $1,850/month, delivery in 5 days, 12-month contract.\n"
        "Supplier B: $1,420/month, delivery in 9 days, 6-month contract.\n"
        "Supplier C: $1,640/month, delivery in 3 days, 12-month contract.\n"
        "Current supplier: $1,980/month, delivery in 7 days.\n"
    ),
    "meeting_notes": (
        "MEETING NOTES 2026-10-20\n"
        "- Website relaunch: owner Priya, due 2026-11-05.\n"
        "- Q4 budget review: owner Marcus, due 2026-10-28.\n"
        "- Vendor contract renewal: owner Priya, due 2026-11-12.\n"
        "- Holiday coverage plan: owner Dana, due 2026-11-20.\n"
    ),
    "plan_a": (
        "PLAN A: Keep the current warehouse. Cost $8,400/month. Throughput "
        "1,200 orders/day. No moving cost. Downtime 0 days.\n"
    ),
    "plan_b": (
        "PLAN B: Move to the larger warehouse. Cost $11,200/month. "
        "Throughput 1,900 orders/day. One-time moving cost $22,000. "
        "Downtime 4 days (about 4,800 orders delayed).\n"
    ),
    "feedback_doc": (
        "CUSTOMER FEEDBACK, WEEK OF OCT 12\n"
        "- Three customers praised the faster delivery times this month.\n"
        "- Two customers reported the checkout page froze on mobile.\n"
        "- One customer asked for a bulk-order discount page.\n"
        "- The most common request: order tracking updates by text message.\n"
    ),
    "messages_triage": (
        "INCOMING MESSAGES\n"
        "M1: \"My invoice shows the wrong total, I was charged twice.\"\n"
        "M2: \"The app crashes every time I open the settings page.\"\n"
        "M3: \"Can I change the delivery address on order #8812?\"\n"
    ),
    "amounts_t4": (
        "EXPENSES TO TOTAL\n"
        "Taxi: $18.50\n"
        "Lunch: $12.75\n"
        "Supplies: $46.20\n"
        "Parking: $9.00\n"
    ),
}


def lookup_doc(doc_id: str) -> str:
    """Return a document by id."""
    if doc_id not in DOCS:
        raise ValueError(f"unknown document: {doc_id!r}")
    return DOCS[doc_id]


def search_notes(query: str) -> str:
    """Keyword search over the notes corpus. Returns matching lines."""
    q = query.lower().strip()
    hits = []
    for doc_id, text in DOCS.items():
        for line in text.splitlines():
            if q and q in line.lower():
                hits.append(f"[{doc_id}] {line.strip()}")
    if not hits:
        return "No matching notes found."
    return "\n".join(hits[:8])


def write_file(path: str, content: str, sandbox: str) -> str:
    """Write content to a file inside the run sandbox. No outside writes."""
    safe = os.path.basename(path)
    full = os.path.join(sandbox, safe)
    os.makedirs(sandbox, exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Wrote {safe} ({len(content)} chars)."


TOOLS = {
    "calculator": {
        "fn": calculator,
        "args": ["expr"],
        "desc": "Evaluate an arithmetic expression, e.g. expr='12.5 + 3.75'.",
    },
    "lookup_doc": {
        "fn": lookup_doc,
        "args": ["doc_id"],
        "desc": "Return a document by id. Known ids: " + ", ".join(sorted(DOCS)),
    },
    "search_notes": {
        "fn": search_notes,
        "args": ["query"],
        "desc": "Keyword search over the notes corpus.",
    },
    "write_file": {
        "fn": None,  # bound per run with the sandbox path
        "args": ["path", "content"],
        "desc": "Write content to a file in the run sandbox.",
    },
}
