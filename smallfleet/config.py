"""Configuration: models, prices, paths.

Prices are published list prices, recorded with their source and date so
anyone can check them. They are used to compute cost from measured token
counts; they are not a bill.

Frontier list price reference: GPT-4o-mini published pricing at the time of
writing (2026-10-25): $0.15 per 1M input tokens, $0.60 per 1M output
tokens (OpenAI API pricing page). If the frontier tier is later served by
a real API model, set FRONTIER_MODEL and its prices to that model's
published rates.

Local tier: $0.00 marginal cost. The model runs on hardware already owned.
"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local small model (tier: local). Qwen2.5-0.5B-Instruct, GGUF Q4_K_M.
LOCAL_MODEL_NAME = "Qwen2.5-0.5B-Instruct-Q4_K_M"
LOCAL_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

# Frontier tier stand-in for the local demo: a larger local model.
# Qwen2.5-3B, not 7B: this VM has 2 CPUs and ~8 GB RAM, and a 7B Q4
# (~4.7 GB) does not fit comfortably next to the 0.5B model. The 3B is
# still clearly more capable than the 0.5B (JSON, reasoning), which is
# what the demo measures. In production this tier is an API model; see
# backends.py. The demo says so in the README and on the dashboard.
# Do not present the stand-in as an API.
FRONTIER_MODEL_NAME = "Qwen2.5-3B-Instruct-Q4_K_M"
FRONTIER_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
FRONTIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen2.5-3b-instruct-q4_k_m.gguf")

# Frontier cost is computed at this published list price.
FRONTIER_PRICE_PER_1M_INPUT_USD = 0.15
FRONTIER_PRICE_PER_1M_OUTPUT_USD = 0.60
FRONTIER_PRICE_NOTE = (
    "GPT-4o-mini published list price, 2026-10-25: $0.15/1M input tokens, "
    "$0.60/1M output tokens. Cost is list-price math from measured tokens, "
    "not a bill. Frontier tier in this local demo is served by a local "
    "Qwen2.5-3B stand-in; see README."
)

LOCAL_PRICE_NOTE = "Local tier: $0.00 marginal cost (compute already owned)."

POLICY_PATH = os.path.join(BASE_DIR, "runs", "policy.json")
LEDGER_PATH = os.path.join(BASE_DIR, "runs", "ledger.db")
DASHBOARD_PORT = int(os.environ.get("SMALLFLEET_PORT", "8100"))

# Confidence: mean token logprob below this escalates a local call.
CONFIDENCE_THRESHOLD = -2.5

# Agent loop guardrails.
MAX_AGENT_STEPS = 12
