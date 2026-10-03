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

from incidents import INCIDENT_WINDOW_MINUTES, check_incident, embed, init_incidents, sync_incident_priority
from triage_engine import CATEGORY_LIST, triage

DB_PATH = "triagebot.db"

COLUMNS = [
    "ticket_id", "created_at", "source", "redacted_text", "category", "priority",
    "sentiment", "confidence", "rationale", "rule_applied", "needs_review",
    "degraded", "error", "model", "prompt_version", "latency_ms", "review_reasons",
]

REVIEW_COLUMNS = {
    "review_reasons": "TEXT",     # why the engine asked for human review
    "review_action": "TEXT",      # approve, change_priority, reroute
    "final_priority": "TEXT",     # what a human decided (same as priority if approved)
    "final_category": "TEXT",
    "reviewed_by": "TEXT",
    "reviewed_at": "TEXT",
}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}


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
        # Review columns, added to existing databases without losing data
        for col, kind in REVIEW_COLUMNS.items():
            try:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {col} {kind}")
            except sqlite3.OperationalError:
                pass  # column already exists
        init_incidents(conn)

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
            "CASE WHEN needs_review = 1 THEN 'Yes' ELSE 'No' END AS needs_review, review_reasons, "
            "incident_id, review_action, final_priority, final_category, reviewed_by, source, redacted_text "
            "FROM tickets ORDER BY created_at DESC LIMIT ?",
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

    # Incident detection: does this ticket look like others from the last hour?
    result.update(similar_count=0, incident_id=None, incident_alert=False,
                  incident_window_minutes=INCIDENT_WINDOW_MINUTES, incident_priority=None)
    if not result["degraded"]:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                result.update(check_incident(conn, result["ticket_id"], result["redacted_text"], result["category"]))
                if result["incident_id"]:
                    top, raised, why = sync_incident_priority(conn, result["incident_id"])
                    result["incident_priority"] = top
                    if result["ticket_id"] in raised:
                        result["priority"] = top
                        result["rule_applied"] = why
        except Exception as exc:  # never let incident detection break triage
            print(f"Incident check failed: {exc}")
    return result


def record_feedback(ticket_id: str, action: str, new_value: str = "", reviewer: str = "") -> dict:
    """Store a human review decision. Exposed as /feedback.

    action: "approve", "change_priority" (new_value = P1..P4) or "reroute" (new_value = a category).
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT priority, category FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise gr.Error(f"Unknown ticket {ticket_id}.")
        priority, category = row

        if action == "approve":
            final_priority, final_category = priority, category
        elif action == "change_priority":
            if new_value not in VALID_PRIORITIES:
                raise gr.Error(f"Priority must be one of {sorted(VALID_PRIORITIES)}.")
            final_priority, final_category = new_value, category
        elif action == "reroute":
            if new_value not in CATEGORY_LIST:
                raise gr.Error(f"Category must be one of {CATEGORY_LIST}.")
            final_priority, final_category = priority, new_value
        else:
            raise gr.Error("Action must be approve, change_priority or reroute.")

        conn.execute(
            "UPDATE tickets SET review_action = ?, final_priority = ?, final_category = ?, "
            "reviewed_by = ?, reviewed_at = ? WHERE ticket_id = ?",
            (action, final_priority, final_category, reviewer or "unknown",
             datetime.now(timezone.utc).isoformat(timespec="seconds"), ticket_id),
        )
    return {
        "ticket_id": ticket_id, "action": action,
        "original_priority": priority, "final_priority": final_priority,
        "original_category": category, "final_category": final_category,
        "overridden": (final_priority, final_category) != (priority, category),
    }


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

    # API-only endpoint (no UI): the Slack review buttons call this
    gr.api(record_feedback, api_name="feedback")

if __name__ == "__main__":
    print("Loading the embedding model for incident detection (first run downloads about 90 MB)...")
    embed("warm up")
    demo.launch(server_name="127.0.0.1", server_port=7860)