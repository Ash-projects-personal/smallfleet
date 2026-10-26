"""Model backends.

Two tiers, one interface. Both return the same dict:
  {"text", "prompt_tokens", "completion_tokens", "mean_logprob", "model"}

- LlamaCppBackend: a GGUF model served in-process through llama-cpp-python.
  Used for both tiers in the local demo: Qwen2.5-0.5B (local tier) and
  Qwen2.5-3B (frontier-tier stand-in).
- OpenAICompatibleBackend: any OpenAI-compatible chat API. Used when a
  real API key is provided; the local demo does not use it, and the README
  says so.

mean_logprob is the mean token log-probability of the completion, when the
backend reports it. The router uses it as a confidence signal: a local call
whose mean logprob falls below the threshold escalates.
"""

from __future__ import annotations

import json
import os
import urllib.request

import requests


class LlamaCppBackend:
    def __init__(self, model_path: str, name: str, n_ctx: int = 2048,
                 n_threads: int | None = None, verbose: bool = False,
                 logits_all: bool = False):
        from llama_cpp import Llama  # imported here so the module loads lazily
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model file not found: {model_path}. Models are not "
                "downloaded, run models/download_models.sh first "
                "(or: python -m smallfleet.download_models).")
        self.name = name
        # logprobs need logits_all=True, which allocates a logits buffer
        # of n_ctx * vocab * 4 bytes (about 600 MB at n_ctx=1024 for
        # Qwen). That is too much on a small demo machine, so it stays
        # off by default: mean_logprob is then None and the router's
        # structural validation (JSON parse + required keys) carries the
        # escalation decision. API backends report logprobs cheaply.
        self.logits_all = logits_all
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_threads=n_threads, verbose=verbose,
                         logits_all=logits_all)

    def complete(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0) -> dict:
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
            logprobs=self.logits_all, top_logprobs=1 if self.logits_all else 0)
        choice = out["choices"][0]
        text = choice["message"]["content"] or ""
        usage = out.get("usage", {})
        mean_logprob = None
        content = (choice.get("logprobs") or {}).get("content") or []
        lps = [t.get("logprob") for t in content
               if t.get("logprob") is not None]
        if lps:
            mean_logprob = sum(lps) / len(lps)
        return {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "mean_logprob": mean_logprob,
            "model": self.name,
        }


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str, model: str,
                 name: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name or model

    def complete(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0) -> dict:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature,
                  "logprobs": True, "top_logprobs": 1},
            timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage", {})
        mean_logprob = None
        content = (choice.get("logprobs") or {}).get("content") or []
        lps = [t.get("logprob") for t in content
               if t.get("logprob") is not None]
        if lps:
            mean_logprob = sum(lps) / len(lps)
        return {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "mean_logprob": mean_logprob,
            "model": self.name,
        }


def download_file(url: str, dest: str) -> None:
    """Download a file with a small progress print. Skips if present."""
    if os.path.exists(dest):
        print(f"already present: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "smallfleet/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", "0"))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB",
                      end="", flush=True)
    print(f"\ndone: {dest}")


def cost_usd(tier: str, prompt_tokens: int, completion_tokens: int,
             input_price: float, output_price: float) -> float:
    """Cost from measured tokens at the published list price.

    Local tier is $0.00 marginal. Frontier tier is list-price math.
    """
    if tier == "local":
        return 0.0
    return (prompt_tokens * input_price / 1_000_000
            + completion_tokens * output_price / 1_000_000)
