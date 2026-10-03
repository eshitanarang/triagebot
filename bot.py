"""
TriageBot Slack bot, built from the Gradio app (per Gradio's
"Creating a Slack Bot from a Gradio App" guide).

This file holds NO triage logic. It listens for @TriageBot mentions,
sends the ticket text to the Gradio app's /triage endpoint through
gradio_client, and formats the answer as a Slack card in the thread.
"""

import os
import re
import threading

from dotenv import load_dotenv
from gradio_client import Client
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

GRADIO_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860/")

app = App(token=os.environ["SLACK_BOT_TOKEN"])
gradio = Client(GRADIO_URL, verbose=False)  # the app must already be running

PRIORITY_LABELS = {
    "P1": ":red_circle: P1 · Critical",
    "P2": ":large_orange_circle: P2 · High",
    "P3": ":large_blue_circle: P3 · Normal",
    "P4": ":white_circle: P4 · Low",
}

# Slack can deliver the same event twice; remember which ones we handled.
_seen_events: set[str] = set()
_seen_lock = threading.Lock()


def already_handled(event_id: str | None) -> bool:
    if not event_id:
        return False
    with _seen_lock:
        if event_id in _seen_events:
            return True
        _seen_events.add(event_id)
        if len(_seen_events) > 5000:  # keep memory bounded
            _seen_events.clear()
        return False


def clean_mention(text: str) -> str:
    """Remove the <@U123ABC> mention codes Slack puts in the raw text."""
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


def build_card(r: dict) -> tuple[list[dict], str]:
    """Turn a triage result into Slack Block Kit blocks plus plain fallback text."""
    priority = PRIORITY_LABELS.get(r["priority"], r["priority"])
    fallback = f"{r['priority']} · {r['category']} · {r['sentiment']}"

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{priority}*"}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Category*\n{r['category'].title()}"},
                {"type": "mrkdwn", "text": f"*Sentiment*\n{r['sentiment'].title()}"},
                {"type": "mrkdwn", "text": f"*Confidence*\n{r['confidence']:.2f}"},
                {"type": "mrkdwn", "text": f"*Ticket ID*\n`{r['ticket_id']}`"},
            ],
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Why:* {r['rationale']}"}},
    ]

    if r.get("rule_applied"):
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f":shield: Safety rule: {r['rule_applied']}"}]})

    if r.get("degraded"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": ":warning: *Automatic triage failed.* A human needs to triage this ticket."}})
    elif r.get("needs_review"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": ":warning: *Low confidence.* Needs human review."}})

    blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
        "text": f"{r['model']} · prompt {r['prompt_version']} · {r['latency_ms']} ms"}]})
    return blocks, fallback


@app.event("app_mention")
def on_mention(body, event, client, logger):
    if already_handled(body.get("event_id")):
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])  # always answer in the ticket's thread
    ticket_text = clean_mention(event.get("text", ""))

    if not ticket_text:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts,
            text="Mention me with the ticket text, for example: `@TriageBot My order never arrived`")
        return

    # 1. Reply at once so the agent knows the bot is working.
    placeholder = client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                          text=":hourglass_flowing_sand: Triaging...")

    # 2. Call the Gradio app. All the AI work happens there.
    try:
        result = gradio.predict(ticket_text, "slack", api_name="/triage")
    except Exception as exc:
        logger.exception("Gradio call failed")
        client.chat_update(channel=channel, ts=placeholder["ts"],
            text=f":x: Couldn't reach the triage app. Is app.py running? ({type(exc).__name__})")
        return

    # 3. Replace the placeholder with the finished card (one message, no spam).
    blocks, fallback = build_card(result)
    client.chat_update(channel=channel, ts=placeholder["ts"], text=fallback, blocks=blocks)


if __name__ == "__main__":
    print(f"TriageBot connected to Gradio app at {GRADIO_URL}")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()