"""Dashboard: serves the latest demo comparison as HTML + JSON.

Stdlib only (http.server), so it runs anywhere. Start with:
    python dashboard.py
then open http://localhost:8100/
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smallfleet import config  # noqa: E402
from smallfleet.ledger import Ledger  # noqa: E402


def latest_comparison(ledger: Ledger) -> dict | None:
    rows = ledger.conn.execute(
        "SELECT id, mode FROM runs ORDER BY started_at DESC LIMIT 10"
    ).fetchall()
    modes = {}
    for r in rows:
        modes.setdefault(r["mode"], r["id"])
    if "all-frontier" not in modes or "routed" not in modes:
        return None
    return ledger.comparison(modes["all-frontier"], modes["routed"])


def render_html(comp: dict | None, policy: dict | None) -> str:
    if comp is None:
        body = ("<p>No demo runs yet. Run <code>python run_demo.py</code> "
                "first.</p>")
    else:
        a, b = comp["run_a"], comp["run_b"]
        saved = (100 * (1 - b["cost_usd"] / a["cost_usd"])
                 if a["cost_usd"] > 0 else 0.0)
        rows = ""
        pa = {r["task_id"]: r for r in comp["per_task_a"]}
        pb = {r["task_id"]: r for r in comp["per_task_b"]}
        for tid in sorted(pa):
            ra, rb = pa[tid], pb[tid]
            rows += (
                f"<tr><td>{tid}</td>"
                f"<td>{'PASS' if ra['passed'] else 'FAIL'}</td>"
                f"<td>${ra['cost_usd']:.4f}</td>"
                f"<td>{'PASS' if rb['passed'] else 'FAIL'}</td>"
                f"<td>${rb['cost_usd']:.4f}</td>"
                f"<td>{rb['escalations']}</td></tr>")
        body = f"""
<h2>Same 10 tasks, same agent logic</h2>
<table>
<tr><th></th><th>All-frontier</th><th>Routed</th></tr>
<tr><td>Tasks passed</td><td>{a['tasks_passed']}/{a['tasks_total']}</td>
    <td>{b['tasks_passed']}/{b['tasks_total']}</td></tr>
<tr><td>LLM calls</td><td>{a['calls']}</td><td>{b['calls']}</td></tr>
<tr><td>Total cost (USD)</td><td>${a['cost_usd']:.4f}</td>
    <td>${b['cost_usd']:.4f} ({saved:.1f}% less)</td></tr>
<tr><td>Total latency</td><td>{a['latency_ms'] / 1000:.1f}s</td>
    <td>{b['latency_ms'] / 1000:.1f}s</td></tr>
<tr><td>Escalations</td><td>{a['escalations']}</td><td>{b['escalations']}</td></tr>
<tr><td>Retries</td><td>{a['retries']}</td><td>{b['retries']}</td></tr>
</table>
<h2>Per-task cost and result</h2>
<table>
<tr><th>Task</th><th>Frontier result</th><th>Frontier cost</th>
<th>Routed result</th><th>Routed cost</th><th>Escalations</th></tr>
{rows}
</table>
<p>Cost is list-price math from measured tokens, not a bill. Local calls
cost $0.00 marginal (compute already owned). Frontier tier in this local
demo is served by a local Qwen2.5-3B stand-in; see README.</p>"""
    policy_html = ""
    if policy:
        items = "".join(
            f"<li>{k}: {v}</li>" for k, v in policy["assignments"].items())
        policy_html = (f"<h2>Learned policy (v{policy['version']})</h2>"
                       f"<ul>{items}</ul>")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SmallFleet</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style></head><body>
<h1>SmallFleet: agent-loop routing</h1>
<p>Routing per tool call inside the agent loop, with a live cost ledger.
Not a prompt router: the router lives inside the agent's execution loop
and classifies each individual call by kind.</p>
{body}
{policy_html}
<p><a href="/api/comparison">comparison as JSON</a></p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        ledger = Ledger(config.LEDGER_PATH)
        if urlparse(self.path).path == "/api/comparison":
            comp = latest_comparison(ledger)
            payload = json.dumps(comp or {"error": "no demo runs yet"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        comp = latest_comparison(ledger)
        policy = None
        if os.path.exists(config.POLICY_PATH):
            with open(config.POLICY_PATH) as f:
                policy = json.load(f)
        html = render_html(comp, policy).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):  # keep the console quiet
        pass


def main() -> None:
    port = config.DASHBOARD_PORT
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"SmallFleet dashboard at http://127.0.0.1:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
