"""The cost ledger.

Every LLM call the agent makes is logged: model, tier, kind, tokens,
measured latency, and cost. Cost is computed from tokens and a published
list price, never from a bill, and the price source is recorded with the
run so the numbers are reproducible.

Local calls cost $0.00 marginal: the model runs on compute already owned.
That is stated honestly in every report, not hidden.
"""

from __future__ import annotations

import json
import sqlite3
import time


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            policy_version INTEGER,
            price_note TEXT
        );
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            step TEXT NOT NULL,
            kind TEXT NOT NULL,
            kind_how TEXT,
            tier TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            mean_logprob REAL,
            cost_usd REAL NOT NULL,
            outcome TEXT NOT NULL,
            outcome_reason TEXT,
            escalated INTEGER NOT NULL DEFAULT 0,
            retries INTEGER NOT NULL DEFAULT 0,
            ts REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_results (
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            passed INTEGER NOT NULL,
            notes TEXT,
            PRIMARY KEY (run_id, task_id)
        );
        """)
        self.conn.commit()

    def new_run(self, run_id: str, mode: str, policy_version: int,
                price_note: str) -> None:
        self.conn.execute(
            "INSERT INTO runs (id, mode, started_at, policy_version, price_note)"
            " VALUES (?, ?, ?, ?, ?)",
            (run_id, mode, time.time(), policy_version, price_note))
        self.conn.commit()

    def finish_run(self, run_id: str) -> None:
        self.conn.execute("UPDATE runs SET finished_at = ? WHERE id = ?",
                          (time.time(), run_id))
        self.conn.commit()

    def log_call(self, run_id: str, task_id: str, step: str, kind: str,
                 kind_how: str, tier: str, model: str, prompt_tokens: int,
                 completion_tokens: int, latency_ms: float,
                 mean_logprob: float | None, cost_usd: float, outcome: str,
                 outcome_reason: str, escalated: bool, retries: int) -> None:
        self.conn.execute(
            """INSERT INTO calls (run_id, task_id, step, kind, kind_how, tier,
               model, prompt_tokens, completion_tokens, latency_ms,
               mean_logprob, cost_usd, outcome, outcome_reason, escalated,
               retries, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, task_id, step, kind, kind_how, tier, model,
             prompt_tokens, completion_tokens, latency_ms, mean_logprob,
             cost_usd, outcome, outcome_reason, 1 if escalated else 0,
             retries, time.time()))
        self.conn.commit()

    def log_task_result(self, run_id: str, task_id: str, passed: bool,
                        notes: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO task_results (run_id, task_id, passed, notes)"
            " VALUES (?, ?, ?, ?)", (run_id, task_id, 1 if passed else 0, notes))
        self.conn.commit()

    def run_totals(self, run_id: str) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(latency_ms), 0) AS latency_ms,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COALESCE(SUM(escalated), 0) AS escalations,
               COALESCE(SUM(retries), 0) AS retries
               FROM calls WHERE run_id = ?""", (run_id,)).fetchone()
        tasks = self.conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(passed), 0) AS passed"
            " FROM task_results WHERE run_id = ?", (run_id,)).fetchone()
        per_tier = self.conn.execute(
            """SELECT tier, COUNT(*) AS calls,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COALESCE(SUM(latency_ms), 0) AS latency_ms
               FROM calls WHERE run_id = ? GROUP BY tier""",
            (run_id,)).fetchall()
        return {
            "calls": row["calls"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "latency_ms": row["latency_ms"],
            "cost_usd": row["cost_usd"],
            "escalations": row["escalations"],
            "retries": row["retries"],
            "tasks_total": tasks["total"],
            "tasks_passed": tasks["passed"],
            "per_tier": {r["tier"]: {"calls": r["calls"],
                                     "cost_usd": r["cost_usd"],
                                     "latency_ms": r["latency_ms"]}
                         for r in per_tier},
        }

    def per_task(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT tr.task_id, tr.passed, tr.notes,
               COUNT(c.id) AS calls,
               COALESCE(SUM(c.cost_usd), 0) AS cost_usd,
               COALESCE(SUM(c.latency_ms), 0) AS latency_ms,
               COALESCE(SUM(c.escalated), 0) AS escalations
               FROM task_results tr LEFT JOIN calls c
               ON c.run_id = tr.run_id AND c.task_id = tr.task_id
               WHERE tr.run_id = ? GROUP BY tr.task_id ORDER BY tr.task_id""",
            (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def kind_stats(self, run_id: str) -> dict:
        """Per-kind, per-tier call outcomes for policy learning.

        A local call counts as a success when its final outcome was ok or
        retried (the local tier handled it). Escalated and failed calls
        count against the local tier.
        """
        rows = self.conn.execute(
            "SELECT kind, tier, outcome, COUNT(*) AS n FROM calls"
            " WHERE run_id = ? GROUP BY kind, tier, outcome",
            (run_id,)).fetchall()
        stats: dict = {}
        for r in rows:
            per_kind = stats.setdefault(r["kind"], {})
            per_tier = per_kind.setdefault(r["tier"], {"total": 0, "ok": 0})
            per_tier["total"] += r["n"]
            if r["outcome"] in ("ok", "retried"):
                per_tier["ok"] += r["n"]
        return stats

    def calls_for_task(self, run_id: str, task_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM calls WHERE run_id = ? AND task_id = ?"
            " ORDER BY id", (run_id, task_id)).fetchall()
        return [dict(r) for r in rows]

    def comparison(self, run_a: str, run_b: str) -> dict:
        """Side-by-side totals for two runs of the same task set."""
        return {"run_a": {"id": run_a, **self.run_totals(run_a)},
                "run_b": {"id": run_b, **self.run_totals(run_b)},
                "per_task_a": self.per_task(run_a),
                "per_task_b": self.per_task(run_b)}

    def export_json(self, run_id: str) -> dict:
        return {"run_id": run_id, "totals": self.run_totals(run_id),
                "per_task": self.per_task(run_id),
                "kind_stats": self.kind_stats(run_id)}
