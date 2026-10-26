"""Smoke test: load both demo models, one completion each, print speed.

If the models are not downloaded yet, this skips cleanly instead of
crashing: run models/download_models.sh first.
"""

import sys
import time

from smallfleet import config
from smallfleet.backends import LlamaCppBackend

PROMPT = (
    "Extract the vendor and total from this invoice. "
    "Vendor: Northwind Office Supply. Total due: $427.68. "
    "Return ONLY a JSON object with keys vendor, total."
)


def timed(name, backend, prompt, max_tokens=64):
    t0 = time.time()
    r = backend.complete(prompt, max_tokens=max_tokens)
    dt = time.time() - t0
    tps = r["completion_tokens"] / dt if dt > 0 else 0
    print(f"{name}: {r['text'][:160]!r}")
    print(f"  tokens={r['completion_tokens']} time={dt:.1f}s "
          f"{tps:.1f} tok/s logprob={r['mean_logprob']}")
    return r


def main() -> int:
    try:
        t0 = time.time()
        local = LlamaCppBackend(config.LOCAL_MODEL_PATH, config.LOCAL_MODEL_NAME)
        print(f"loaded 0.5B in {time.time() - t0:.1f}s")
        timed("0.5B", local, PROMPT)

        t0 = time.time()
        frontier = LlamaCppBackend(config.FRONTIER_MODEL_PATH,
                                   config.FRONTIER_MODEL_NAME)
        print(f"loaded 3B in {time.time() - t0:.1f}s")
        timed("3B", frontier, PROMPT)
        timed("3B-json-hard", frontier,
              "Extract vendor, date, total, and all 4 line items from this invoice "
              "text into strict JSON with keys vendor, date, total, line_items. "
              "Invoice #1043, Vendor: Bright & Sons, LLC, date 2026-10-02. "
              "Items: Widget deluxe 2 x $45.99; Cable 6ft 5 x $7.50; Adapter USB-C "
              "3 x $12.00; Rush fee $25.00. Total due $205.72.",
              max_tokens=256)
    except FileNotFoundError as exc:
        print(f"SKIP: {exc}")
        print("models not downloaded, run models/download_models.sh")
        return 0
    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
