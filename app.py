"""
TriageBot Gradio app.

The brain of the project: it wraps the triage engine in named API endpoints
(which the Slack bot calls through gradio_client) and gives admins a UI.
"""

import json
import sqlite3
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from triage_engine import triage

DB_PATH = "triagebot.db"

COLUMNS = [
    "ticket_id", "created_at", "source", "redacted_text", "category", "priority",
    "sentiment", "confidence", "rationale", "rule_applied", "needs_review",
    "degraded", "error", "model", "prompt_version", "latency_ms",
]


# ---------------------------------------------------------------------------
# Audit log: every triage decision is stored, redacted text only
# ---------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY, created_at TEXT, source TEXT, redacted_text TEXT,
                category TEXT, priority TEXT, sentiment TEXT, confidence REAL, rationale TEXT,
                rule_applied TEXT, needs_review INTEGER, degraded INTEGER, error TEXT,
                model TEXT, prompt_version TEXT, latency_ms INTEGER
            )"""
        )


def log_ticket(result: dict, source: str):
    row = {**result, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"INSERT INTO tickets ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
            [row.get(c) for c in COLUMNS],
        )


def load_tickets(limit: int = 200) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at, ticket_id, priority, category, sentiment, confidence, "
            "needs_review, source, redacted_text FROM tickets ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,),
        )


# ---------------------------------------------------------------------------
# API functions (each one is an endpoint the Slack bot can call)
# ---------------------------------------------------------------------------
def triage_ticket(ticket_text: str, source: str = "playground") -> dict:
    """Triage one ticket and log it. Exposed as /triage."""
    if not ticket_text or not ticket_text.strip():
        raise gr.Error("Enter the ticket text first.")
    result = triage(ticket_text)
    log_ticket(result, source)
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
init_db()

with gr.Blocks(title="TriageBot") as demo:
    gr.Markdown("# TriageBot\nAI support-ticket triage. Slack is the front door; this app is the brain.")

    with gr.Tab("Playground"):
        ticket_in = gr.Textbox(
            label="Ticket text", lines=5,
            placeholder="I was charged twice for my order and it still isn't showing in my account.",
        )
        source_in = gr.Textbox(value="playground", visible=False)
        run_btn = gr.Button("Triage ticket", variant="primary")
        result_out = gr.JSON(label="Triage result")
        run_btn.click(triage_ticket, inputs=[ticket_in, source_in], outputs=result_out, api_name="triage")

    with gr.Tab("Live queue"):
        refresh_btn = gr.Button("Refresh")
        queue_table = gr.Dataframe(value=load_tickets, interactive=False, wrap=True)
        refresh_btn.click(load_tickets, outputs=queue_table, api_name=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)