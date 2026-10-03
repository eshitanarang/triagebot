"""
Incident detection for TriageBot.

An outage first shows up as several customers reporting the same problem at once,
but each agent only sees their own ticket. This module embeds every ticket and,
when K similar tickets arrive within N minutes, groups them into one incident and
signals an alert (once per incident, not once per ticket).
"""

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

SIMILARITY_THRESHOLD = float(os.getenv("INCIDENT_SIMILARITY", "0.6"))  # cosine, 0 to 1
INCIDENT_MIN_TICKETS = int(os.getenv("INCIDENT_MIN_TICKETS", "3"))      # K, including the new ticket
INCIDENT_WINDOW_MINUTES = int(os.getenv("INCIDENT_WINDOW_MINUTES", "60"))  # N

_model = None


def embed(text: str) -> np.ndarray:
    """Turn text into a normalized vector. The model loads once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model.encode(text, normalize_embeddings=True).astype(np.float32)


def init_incidents(conn: sqlite3.Connection):
    for col, kind in {"embedding": "BLOB", "incident_id": "TEXT"}.items():
        try:
            conn.execute(f"ALTER TABLE tickets ADD COLUMN {col} {kind}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY, created_at TEXT, category TEXT,
            ticket_count INTEGER, sample_text TEXT
        )"""
    )


def check_incident(conn: sqlite3.Connection, ticket_id: str, text: str, category: str,
                   embed_fn=None) -> dict:
    """Store this ticket's embedding, find recent similar tickets, open or join an incident.

    Returns similar_count (other recent tickets that look alike), incident_id (or None)
    and incident_alert (True only when this ticket opened a new incident).
    """
    vector = (embed_fn or embed)(text)
    conn.execute("UPDATE tickets SET embedding = ? WHERE ticket_id = ?", (vector.tobytes(), ticket_id))

    since = (datetime.now(timezone.utc) - timedelta(minutes=INCIDENT_WINDOW_MINUTES)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT ticket_id, embedding, incident_id FROM tickets "
        "WHERE embedding IS NOT NULL AND ticket_id != ? AND created_at >= ?",
        (ticket_id, since),
    ).fetchall()

    similar = []
    for other_id, blob, other_incident in rows:
        score = float(np.dot(vector, np.frombuffer(blob, dtype=np.float32)))
        if score >= SIMILARITY_THRESHOLD:
            similar.append((other_id, other_incident))

    result = {"similar_count": len(similar), "incident_id": None, "incident_alert": False}
    if len(similar) + 1 < INCIDENT_MIN_TICKETS:
        return result

    existing = next((inc for _, inc in similar if inc), None)
    if existing:
        # Join the open incident quietly; the team was already alerted.
        conn.execute("UPDATE tickets SET incident_id = ? WHERE ticket_id = ?", (existing, ticket_id))
        conn.execute("UPDATE incidents SET ticket_count = ticket_count + 1 WHERE incident_id = ?", (existing,))
        result["incident_id"] = existing
        return result

    incident_id = "INC-" + uuid.uuid4().hex[:6].upper()
    members = [ticket_id] + [other_id for other_id, _ in similar]
    conn.execute(
        "INSERT INTO incidents VALUES (?, ?, ?, ?, ?)",
        (incident_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), category,
         len(members), text[:200]),
    )
    conn.executemany("UPDATE tickets SET incident_id = ? WHERE ticket_id = ?",
                     [(incident_id, m) for m in members])
    result.update(incident_id=incident_id, incident_alert=True)
    result["similar_count"] = len(similar)
    return result


def sync_incident_priority(conn: sqlite3.Connection, incident_id: str) -> tuple[str, list[str], str]:
    """One problem, one priority: raise every ticket in the incident to one shared level.

    The level is the most urgent ticket in the group, and at least P1 when most of
    the customers are complaining (negative sentiment): the rubric's "many customers
    affected at once". A burst of neutral questions keeps its normal priority.
    Uses a human's final priority when a ticket was reviewed. Returns the incident
    priority, the ticket IDs that were raised, and the reason.
    """
    rows = conn.execute(
        "SELECT ticket_id, COALESCE(final_priority, priority), sentiment FROM tickets WHERE incident_id = ?",
        (incident_id,),
    ).fetchall()
    top = min(p for _, p, _ in rows)  # "P1" < "P2" < ...
    negative_share = sum(1 for _, _, s in rows if s == "negative") / len(rows)
    if negative_share > 0.5:
        top = "P1"
        why = f"raised to P1: incident {incident_id}, several customers reporting the same problem"
    else:
        why = f"raised to {top} because incident {incident_id} is {top}"

    raised = [tid for tid, p, _ in rows if p > top]
    for tid in raised:
        conn.execute("UPDATE tickets SET priority = ?, rule_applied = ? WHERE ticket_id = ?", (top, why, tid))
    return top, raised, why

