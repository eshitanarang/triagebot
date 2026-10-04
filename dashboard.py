"""
Numbers for the Analytics and Model quality tabs, computed from the audit log.
Every figure here comes from real logged tickets; nothing is estimated.
"""

import sqlite3

import gradio as gr
import pandas as pd

PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]
CONFIDENCE_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
CONFIDENCE_LABELS = ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def _tickets(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ticket_id, created_at, category, priority, sentiment, confidence, needs_review, "
            "review_reasons, degraded, latency_ms, model, incident_id, review_action, "
            "final_priority, final_category FROM tickets",
            conn,
        )


def tiles(items: list[tuple[str, str, str]]) -> str:
    """Stat tiles: (label, value, note). Plain HTML so it follows Gradio's light and dark themes."""
    cells = "".join(
        f"<div style='flex:1 1 140px;padding:12px 14px;border-radius:8px;"
        f"background:var(--block-background-fill);border:1px solid var(--border-color-primary)'>"
        f"<div style='font-size:13px;color:var(--body-text-color-subdued)'>{label}</div>"
        f"<div style='font-size:26px;font-weight:600;color:var(--body-text-color)'>{value}</div>"
        f"<div style='font-size:12px;color:var(--body-text-color-subdued)'>{note}</div></div>"
        for label, value, note in items
    )
    return f"<div style='display:flex;flex-wrap:wrap;gap:12px'>{cells}</div>"


def bars(df: pd.DataFrame):
    """Bar data with the y-axis pinned to start at 0 (a bar chart must never start above zero)."""
    top = max(1, int(df["tickets"].max())) if len(df) else 1
    return gr.BarPlot(value=df, y_lim=[0, top * 1.15])


MIN_REVIEWS_FOR_RATE = 5


def highlight_agreement(matrix: pd.DataFrame):
    """Tint the diagonal (AI and human agree) green and non-zero corrections amber."""
    def style_row(row):
        styles = [""]  # the row-label column
        for col in matrix.columns[1:]:
            if col == row.iloc[0]:
                styles.append("background-color: rgba(34,160,90,0.18); font-weight: 600")
            elif row[col] > 0:
                styles.append("background-color: rgba(230,150,30,0.25); font-weight: 600")
            else:
                styles.append("")
        return styles
    return matrix.style.apply(style_row, axis=1)


def _count(series: pd.Series, name: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = series.value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    return counts.rename_axis(name).reset_index(name="tickets")


def analytics(db_path: str):
    df = _tickets(db_path)
    if df.empty:
        empty = pd.DataFrame({"x": [], "tickets": []})
        return tiles([("Tickets", "0", "send one from Slack")]), empty, empty, empty, empty, pd.DataFrame()

    open_p1 = int(((df["priority"] == "P1") & df["review_action"].isna()).sum())
    kpis = tiles([
        ("Tickets triaged", f"{len(df)}", "all sources"),
        ("P1 not yet reviewed", f"{open_p1}", "critical, awaiting a human"),
        ("Flagged for review", f"{int((df['needs_review'] == 1).sum())}", "by review triggers"),
        ("Incidents detected", f"{df['incident_id'].nunique()}", "groups of similar tickets"),
    ])

    by_category = _count(df["category"], "category").sort_values("tickets", ascending=False)
    by_priority = _count(df["priority"], "priority", PRIORITY_ORDER)
    by_sentiment = _count(df["sentiment"], "sentiment", ["negative", "neutral", "positive"])
    bins = pd.cut(df["confidence"], bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS, right=False)
    by_confidence = _count(bins.astype(str), "confidence", CONFIDENCE_LABELS)

    with sqlite3.connect(db_path) as conn:
        incidents = pd.read_sql_query(
            "SELECT i.incident_id, i.created_at, i.category, i.ticket_count AS tickets, "
            "MIN(COALESCE(t.final_priority, t.priority)) AS priority, i.sample_text "
            "FROM incidents i LEFT JOIN tickets t ON t.incident_id = i.incident_id "
            "GROUP BY i.incident_id ORDER BY i.created_at DESC",
            conn,
        )
    return kpis, bars(by_category), bars(by_priority), bars(by_sentiment), bars(by_confidence), incidents


def model_quality(db_path: str):
    df = _tickets(db_path)
    if df.empty:
        return tiles([("Reviewed tickets", "0", "use the Slack buttons")]), pd.DataFrame(), pd.DataFrame()

    reviewed = df[df["review_action"].notna()]
    overridden = reviewed[
        (reviewed["final_priority"] != reviewed["priority"]) | (reviewed["final_category"] != reviewed["category"])
    ]
    # A rate from a handful of reviews is noise, so say so instead of showing a bold percentage.
    if len(reviewed) >= MIN_REVIEWS_FOR_RATE:
        override, override_note = f"{len(overridden) / len(reviewed):.0%}", f"{len(overridden)} of {len(reviewed)} reviews changed the AI's answer"
    else:
        override = f"{len(overridden)} of {len(reviewed)}" if len(reviewed) else "-"
        override_note = f"too few reviews for a rate (need {MIN_REVIEWS_FOR_RATE}+)"
    rerouted = int((reviewed["final_category"] != reviewed["category"]).sum())
    lat = df["latency_ms"].dropna()
    kpis = tiles([
        ("Reviewed by a human", f"{len(reviewed)}", f"of {len(df)} tickets"),
        ("Human overrides", override, override_note),
        ("Category re-routes", f"{rerouted}", "included in overrides; not in the grid below"),
        ("Latency p50 / p95", f"{lat.median() / 1000:.1f} / {lat.quantile(0.95) / 1000:.1f} s", "model call only"),
        ("Degraded (triage failed)", f"{int((df['degraded'] == 1).sum())}", "sent to a human"),
    ])

    # Rows = what the AI said, columns = what the human decided. The diagonal is agreement.
    if len(reviewed):
        matrix = pd.crosstab(reviewed["priority"], reviewed["final_priority"])
        matrix = matrix.reindex(index=PRIORITY_ORDER, columns=PRIORITY_ORDER, fill_value=0)
        matrix.index.name = "AI \\ human"
        matrix = highlight_agreement(matrix.reset_index())
    else:
        matrix = pd.DataFrame({"note": ["No reviewed tickets yet. Use Approve or Change priority in Slack."]})

    reasons = (
        df["review_reasons"].dropna().loc[lambda s: s != ""]
        # tickets logged before the wording change used a semicolon inside this reason
        .str.replace("model under-rated priority; safety rule raised it",
                     "model under-rated priority (safety rule raised it)", regex=False)
        .str.split("; ").explode()
    )
    by_reason = _count(reasons, "review reason") if len(reasons) else pd.DataFrame({"review reason": [], "tickets": []})
    return kpis, matrix, bars(by_reason)


# ---------------------------------------------------------------------------
# Offline evaluation results (read from the eval_summary_*.json files)
# ---------------------------------------------------------------------------
def _load_eval_runs(folder: str) -> list[dict]:
    import json
    import re
    from pathlib import Path

    runs = []
    for path in Path(folder).glob("eval_summary_*.json"):
        s = json.loads(path.read_text())
        m = re.search(r"eval_summary_(v\d+)_seed(\d+)", path.name)
        s.setdefault("prompt_version", m.group(1) if m else "?")
        s.setdefault("seed", int(m.group(2)) if m else None)
        s["_mtime"] = path.stat().st_mtime
        runs.append(s)
    return sorted(runs, key=lambda s: s["_mtime"])


def evaluation(folder: str = "."):
    runs = _load_eval_runs(folder)
    if not runs:
        empty = pd.DataFrame({"note": ["No evaluation yet. Run: python evaluate.py"]})
        return tiles([("Evaluation runs", "0", "run evaluate.py")]), empty, empty, empty

    first, latest = runs[0], runs[-1]
    kpis = tiles([
        ("Category accuracy, latest", f"{latest['category_accuracy']:.1%}",
         f"prompt {latest['prompt_version']}, {latest['tickets']} fresh tickets"),
        ("Baseline", f"{first['category_accuracy']:.1%}", f"prompt {first['prompt_version']}, before definitions"),
        ("Errors caught by review", latest["errors_caught_by_review"], "latest run"),
        ("Latency p50 / p95", f"{latest['latency_p50_s']} / {latest['latency_p95_s']} s", f"{latest['model']}"),
    ])

    table = pd.DataFrame([{
        "prompt": r["prompt_version"], "seed": r["seed"], "model": r["model"], "tickets": r["tickets"],
        "accuracy": f"{r['category_accuracy']:.1%}", "sent to review": f"{r['review_rate']:.0%}",
        "errors caught by review": r["errors_caught_by_review"], "failed calls": r["failed_calls"],
    } for r in runs])

    # Per-category recall: one column per run, so before/after sits side by side
    recall = pd.DataFrame({
        f"{r['prompt_version']} (seed {r['seed']})": {c: v["recall"] for c, v in r["per_category_recall"].items()}
        for r in runs
    })
    recall = recall.sort_values(recall.columns[-1]).rename_axis("category").reset_index()
    pct_cols = list(recall.columns[1:])

    def shade(v):
        if not isinstance(v, (int, float)):
            return ""
        if v >= 0.999:
            return "background-color: rgba(34,160,90,0.18)"
        if v < 0.8:
            return "background-color: rgba(230,150,30,0.25); font-weight: 600"
        return ""

    recall_styled = recall.style.format({c: "{:.0%}" for c in pct_cols}).map(shade, subset=pct_cols)

    mistakes = pd.DataFrame(latest.get("top_confusions") or [])
    if mistakes.empty:
        mistakes = pd.DataFrame({"note": ["No mistakes in the latest run."]})
    else:
        mistakes.columns = ["true category", "predicted", "count"]
    return kpis, table, recall_styled, mistakes