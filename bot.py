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
# Channel name or channel ID. If Slack says channel_not_found, put the channel ID here instead.
ESCALATION_CHANNEL = os.getenv("ESCALATION_CHANNEL", "support-escalations")

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

    if r.get("incident_id"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f":rotating_light: *Part of possible incident {r['incident_id']}* · "
                    f"{r['similar_count']} similar tickets in the last {r['incident_window_minutes']} min"}})
    elif r.get("similar_count"):
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
            "text": f":link: {r['similar_count']} similar ticket(s) in the last {r['incident_window_minutes']} min"}]})

    if r.get("degraded"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": ":warning: *Automatic triage failed.* A human needs to triage this ticket."}})
    elif r.get("needs_review"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f":warning: *Needs human review:* {r.get('review_reasons') or 'flagged by triage'}"}})

    blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
        "text": f"{r['model']} · prompt {r['prompt_version']} · {r['latency_ms']} ms"}]})
    blocks.append(review_controls(r["ticket_id"]))
    return blocks, fallback


CATEGORIES = [
    "account", "cancel", "contact", "delivery", "feedback", "invoice",
    "order", "payment", "refund", "shipping", "subscription", "technical", "other",
]


def review_controls(ticket_id: str) -> dict:
    """Approve button plus two dropdowns. The ticket ID travels inside each value."""
    return {
        "type": "actions",
        "block_id": "review",
        "elements": [
            {"type": "button", "action_id": "review_approve", "style": "primary",
             "text": {"type": "plain_text", "text": "Approve"}, "value": ticket_id},
            {"type": "static_select", "action_id": "review_priority",
             "placeholder": {"type": "plain_text", "text": "Change priority"},
             "options": [{"text": {"type": "plain_text", "text": p}, "value": f"{ticket_id}|{p}"}
                         for p in ["P1", "P2", "P3", "P4"]]},
            {"type": "static_select", "action_id": "review_reroute",
             "placeholder": {"type": "plain_text", "text": "Re-route"},
             "options": [{"text": {"type": "plain_text", "text": c.title()}, "value": f"{ticket_id}|{c}"}
                         for c in CATEGORIES]},
        ],
    }


def handle_review(ack, body, client, logger, action_type: str):
    ack()  # Slack needs an answer within 3 seconds
    action = body["actions"][0]
    if action["type"] == "button":
        ticket_id, new_value = action["value"], ""
    else:
        ticket_id, new_value = action["selected_option"]["value"].split("|", 1)

    user_id = body["user"]["id"]
    reviewer = body["user"].get("username") or body["user"].get("name") or user_id
    channel = body["channel"]["id"]
    message = body["message"]

    try:
        res = gradio.predict(ticket_id, action_type, new_value, reviewer, api_name="/feedback")
    except Exception as exc:
        logger.exception("Feedback call failed")
        client.chat_postEphemeral(channel=channel, user=user_id,
            text=f":x: Couldn't save your review: {str(exc)[:150]}")
        return

    # Rebuild the card: drop the controls, reflect the change, record who reviewed it.
    blocks = [b for b in message["blocks"] if b.get("type") != "actions"]
    if action_type == "change_priority":
        blocks[0]["text"]["text"] = f"*{PRIORITY_LABELS[res['final_priority']]}*"
        note = f":pencil2: Priority changed {res['original_priority']} → {res['final_priority']} by <@{user_id}>"
    elif action_type == "reroute":
        for b in blocks:
            if b.get("fields"):
                b["fields"][0]["text"] = f"*Category*\n{res['final_category'].title()}"
                break
        note = f":twisted_rightwards_arrows: Re-routed {res['original_category']} → {res['final_category']} by <@{user_id}>"
    else:
        note = f":white_check_mark: Approved by <@{user_id}>"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": note}]})

    client.chat_update(channel=channel, ts=message["ts"], text=message.get("text", note), blocks=blocks)

    # A human raised this ticket to P1: escalate it now.
    if action_type == "change_priority" and res["final_priority"] == "P1" and res["original_priority"] != "P1":
        escalate(client, logger, channel, message.get("thread_ts", message["ts"]), ticket_id,
                 res["final_category"], f"raised to P1 by <@{user_id}>")


@app.action("review_approve")
def on_approve(ack, body, client, logger):
    handle_review(ack, body, client, logger, "approve")


@app.action("review_priority")
def on_change_priority(ack, body, client, logger):
    handle_review(ack, body, client, logger, "change_priority")


@app.action("review_reroute")
def on_reroute(ack, body, client, logger):
    handle_review(ack, body, client, logger, "reroute")


def escalate(client, logger, source_channel: str, message_ts: str, ticket_id: str,
             category: str, reason: str):
    """Post a P1 alert to the escalation channel with a link back to the ticket."""
    link = ""
    try:
        permalink = client.chat_getPermalink(channel=source_channel, message_ts=message_ts)["permalink"]
        link = f" · <{permalink}|View ticket>"
    except Exception:
        logger.warning("Couldn't get a link to the ticket; posting the alert without it")
    try:
        client.chat_postMessage(
            channel=ESCALATION_CHANNEL,
            text=f":rotating_light: *P1 ticket* `{ticket_id}` · {category.title()} · {reason}{link}",
        )
    except Exception:
        logger.exception("Escalation post failed. Is the bot invited to the escalation channel?")


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

    # 4. Escalate critical tickets automatically. Tickets inside an incident are
    #    covered by the single incident alert, so they don't each post their own.
    if result["priority"] == "P1" and not result.get("degraded") and not result.get("incident_id"):
        reason = result.get("rule_applied") or result["rationale"]
        escalate(client, logger, channel, thread_ts, result["ticket_id"], result["category"], reason)

    # 5. A new incident was just detected: alert once, with a link to the latest ticket.
    if result.get("incident_alert"):
        post_incident_alert(client, logger, channel, thread_ts, result)


def post_incident_alert(client, logger, source_channel: str, message_ts: str, r: dict):
    link = ""
    try:
        permalink = client.chat_getPermalink(channel=source_channel, message_ts=message_ts)["permalink"]
        link = f" · <{permalink}|Latest ticket>"
    except Exception:
        pass
    count = r["similar_count"] + 1
    try:
        client.chat_postMessage(
            channel=ESCALATION_CHANNEL,
            text=(f":rotating_light: *Possible incident {r['incident_id']}:* {count} similar "
                  f"{r['category']} tickets in the last {r['incident_window_minutes']} min. "
                  f"All set to *{r.get('incident_priority') or r['priority']}*.\n"
                  f"> {r['redacted_text'][:200]}{link}"),
        )
    except Exception:
        logger.exception("Incident alert failed")


if __name__ == "__main__":
    print(f"TriageBot connected to Gradio app at {GRADIO_URL}")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()