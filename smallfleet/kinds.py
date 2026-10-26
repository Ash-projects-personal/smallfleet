"""Call kinds and the classifier.

Every LLM call an agent makes inside its execution loop has a kind: what the
call is for. The classifier tags each call so the router can decide which
model tier handles it.

The classifier is deliberately small and transparent: the agent node that
makes the call declares its intent (the hint), and a keyword/pattern scorer
checks the prompt and can override the hint when the prompt strongly matches
another kind. Every decision records how the kind was determined, so the
ledger shows the reasoning, not just the label.
"""

from __future__ import annotations

import re

ROUTING = "routing"
EXTRACTION = "extraction"
TOOL_CALL = "tool_call"
SUMMARIZATION = "summarization"
REASONING = "reasoning"
WRITING = "writing"

ALL_KINDS = [ROUTING, EXTRACTION, TOOL_CALL, SUMMARIZATION, REASONING, WRITING]

# Weighted keyword/pattern signals per kind. A call's prompt is scored
# against each kind; the highest score wins when no hint is given, and a
# strong score can override a hint.
_SIGNALS = {
    ROUTING: [
        (r"\bnext (step|action)\b", 3),
        (r"\bwhich (tool|node|step)\b", 3),
        (r"\bdecide\b.*\b(next|what to do)\b", 3),
        (r"\btriage\b", 3),
        (r"\bclassify\b.*\b(request|message|ticket)\b", 3),
        (r"\broute\b", 2),
        (r"\bassign\b.*\b(queue|team|category)\b", 2),
    ],
    EXTRACTION: [
        (r"\bextract\b", 3),
        (r"\breturn (only )?json\b", 3),
        (r"\bfields?\b.*\bjson\b", 2),
        (r"\bpull (out|the)\b.*\b(fields?|values?|dates?|names?)\b", 3),
        (r"\bstrict json\b", 3),
        (r"\bparse\b.*\b(text|document|invoice|note)\b", 2),
    ],
    TOOL_CALL: [
        (r"\bcall the\b.*\btool\b", 3),
        (r"\btool call\b", 3),
        (r"\bcompute\b.*\busing\b", 2),
        (r"\bcalculate\b", 2),
        (r"\blook ?up\b", 2),
        (r"\bsearch\b.*\b(notes?|docs?|corpus)\b", 2),
        (r"\baction\s*:\s*\{", 2),
    ],
    SUMMARIZATION: [
        (r"\bsummar(y|ize|ise)\b", 3),
        (r"\bin (two|three|\d+) sentences?\b", 2),
        (r"\bkey points?\b", 2),
        (r"\btl;dr\b", 2),
        (r"\bcondense\b", 2),
    ],
    REASONING: [
        (r"\bcompare\b.*\b(and|versus|vs\.?)\b", 3),
        (r"\bcheapest\b", 2),
        (r"\bsavings?\b", 2),
        (r"\bstep by step\b", 2),
        (r"\bwork(ing)? out\b", 2),
        (r"\bwhich (option|plan|supplier) is (better|best|cheapest)\b", 3),
        (r"\breason\b", 1),
        (r"\bmulti-?step\b", 2),
    ],
    WRITING: [
        (r"\bdraft\b", 3),
        (r"\bwrite (a|an|the)\b", 3),
        (r"\bemail\b", 2),
        (r"\bstatus update\b", 2),
        (r"\breply\b", 2),
        (r"\bcompose\b", 2),
    ],
}


def score_prompt(prompt: str) -> dict[str, int]:
    """Score a prompt against every kind. Returns {kind: score}."""
    text = prompt.lower()
    scores: dict[str, int] = {}
    for kind, patterns in _SIGNALS.items():
        total = 0
        for pattern, weight in patterns:
            if re.search(pattern, text):
                total += weight
        scores[kind] = total
    return scores


def classify(prompt: str, hint: str | None = None) -> tuple[str, str]:
    """Tag a call with its kind.

    Returns (kind, how) where how is one of:
      - "hint"      the agent node declared the kind and the prompt agreed
      - "hint+score" the node declared the kind and the scorer agreed
      - "override"  the scorer strongly disagreed with the hint
      - "score"     no hint; the scorer decided
      - "default"   no hint and no signal; falls back to tool_call
    """
    scores = score_prompt(prompt)
    best_kind = max(scores, key=lambda k: scores[k])
    best_score = scores[best_kind]

    if hint is not None and hint in ALL_KINDS:
        # Override only when the hint has no support at all in the prompt.
        # (A plan prompt about an extraction step mentions "extract" and
        # "JSON" heavily, but the call itself is still a routing decision.)
        if best_score >= 5 and best_kind != hint and scores[hint] == 0:
            return best_kind, "override"
        if best_kind == hint and best_score > 0:
            return hint, "hint+score"
        return hint, "hint"

    if best_score > 0:
        return best_kind, "score"
    return TOOL_CALL, "default"
