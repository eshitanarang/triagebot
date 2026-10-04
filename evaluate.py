"""
Offline evaluation of TriageBot's category accuracy on the Bitext customer-support dataset.

Usage:
    python evaluate.py                                   # 60 tickets, production model, seed 42
    python evaluate.py --seed 7 --exclude eval_results_v1_seed42.csv   # fresh sample, no repeats
    python evaluate.py --n 20 --model dev

It calls triage() directly (not through Slack or the app), so evaluation tickets never
appear in your live dashboard. Calls are spaced out to stay under Groq's free-tier
tokens-per-minute limit. Results are saved to eval_results_<prompt>_seed<seed>.csv and
eval_summary_<prompt>_seed<seed>.json, so earlier runs are never overwritten.
"""

import argparse
import json
import os
import time

import pandas as pd

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def load_bitext() -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset(DATASET, split="train").to_pandas()
    return df[["instruction", "category", "intent"]]


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """The same number of tickets from every category, so small categories still count."""
    per_cat = max(1, n // df["category"].nunique())
    return (df.groupby("category", group_keys=False)
              .apply(lambda g: g.sample(min(per_cat, len(g)), random_state=seed))
              .reset_index(drop=True))


def run(n: int, model: str, delay: float, seed: int = 42, exclude: str | None = None, loader=load_bitext):
    os.environ["APP_ENV"] = "prod" if model == "prod" else "dev"
    from triage_engine import CATEGORY_LIST, MODEL, PROMPT_VERSION, triage  # imported after APP_ENV is set

    data = loader()
    data["true_category"] = data["category"].str.strip().str.lower()
    unknown = sorted(set(data["true_category"]) - set(CATEGORY_LIST))
    if unknown:
        print(f"Warning: dataset categories not in TriageBot's list: {unknown}")
    if exclude:
        seen = set(pd.read_csv(exclude)["ticket"])
        before = len(data)
        data = data[~data["instruction"].isin(seen)]
        print(f"Excluded {before - len(data)} tickets already used in {exclude}")
    sample = stratified_sample(data, n, seed=seed)
    print(f"Evaluating {len(sample)} tickets ({sample['true_category'].nunique()} categories) with {MODEL}")
    print(f"Spacing calls {delay:.0f} s apart for the free-tier token limit: about "
          f"{len(sample) * delay / 60:.0f} minutes.\n")

    rows = []
    for i, r in sample.iterrows():
        started = time.time()
        out = triage(r["instruction"])
        correct = out["category"] == r["true_category"]
        rows.append({
            "ticket": r["instruction"], "intent": r["intent"], "true_category": r["true_category"],
            "predicted_category": out["category"], "correct": correct,
            "priority": out["priority"], "sentiment": out["sentiment"], "confidence": out["confidence"],
            "needs_review": out["needs_review"], "review_reasons": out["review_reasons"],
            "degraded": out["degraded"], "latency_ms": out["latency_ms"], "error": out["error"],
        })
        mark = "ok " if correct else "MISS"
        print(f"[{i + 1:>3}/{len(sample)}] {mark} {r['true_category']:<13} -> {out['category']:<13} "
              f"conf {out['confidence']:.2f}  {r['instruction'][:50]}")
        time.sleep(max(0.0, delay - (time.time() - started)))

    res = pd.DataFrame(rows)
    tag = f"{PROMPT_VERSION}_seed{seed}"
    res.to_csv(f"eval_results_{tag}.csv", index=False)
    summary = summarize(res, MODEL)
    summary.update(prompt_version=PROMPT_VERSION, seed=seed, excluded_from=exclude)
    with open(f"eval_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"Saved eval_results_{tag}.csv and eval_summary_{tag}.json")
    return summary


def summarize(res: pd.DataFrame, model: str) -> dict:
    valid = res[~res["degraded"]]
    per_cat = (valid.groupby("true_category")["correct"].agg(["mean", "count"])
                    .rename(columns={"mean": "recall", "count": "tickets"}))
    flagged, unflagged = valid[valid["needs_review"]], valid[~valid["needs_review"]]
    wrong = valid[~valid["correct"]]

    # Calibration: is the model right as often as its confidence claims?
    bins = pd.cut(valid["confidence"], [0, 0.7, 0.8, 0.9, 1.0001], right=False,
                  labels=["<0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"])
    calibration = valid.groupby(bins, observed=True)["correct"].agg(["mean", "count"])

    return {
        "model": model,
        "tickets": int(len(res)),
        "failed_calls": int(res["degraded"].sum()),
        "category_accuracy": round(float(valid["correct"].mean()), 3) if len(valid) else None,
        "per_category_recall": {k: {"recall": round(float(v["recall"]), 2), "tickets": int(v["tickets"])}
                                for k, v in per_cat.iterrows()},
        "review_rate": round(float(valid["needs_review"].mean()), 3) if len(valid) else None,
        "accuracy_when_flagged": round(float(flagged["correct"].mean()), 3) if len(flagged) else None,
        "accuracy_when_not_flagged": round(float(unflagged["correct"].mean()), 3) if len(unflagged) else None,
        "errors_caught_by_review": f"{int(wrong['needs_review'].sum())} of {len(wrong)}",
        "calibration": {str(k): {"accuracy": round(float(v["mean"]), 2), "tickets": int(v["count"])}
                        for k, v in calibration.iterrows()},
        "latency_p50_s": round(float(valid["latency_ms"].median()) / 1000, 2) if len(valid) else None,
        "latency_p95_s": round(float(valid["latency_ms"].quantile(0.95)) / 1000, 2) if len(valid) else None,
        "top_confusions": (wrong.groupby(["true_category", "predicted_category"]).size()
                                .sort_values(ascending=False).head(5)
                                .reset_index(name="count").to_dict("records")),
    }


def print_summary(s: dict):
    print("\n" + "=" * 60)
    print(f"Model: {s['model']}   Prompt: {s.get('prompt_version', '?')}   Tickets: {s['tickets']}   Failed calls: {s['failed_calls']}")
    print(f"Category accuracy: {s['category_accuracy']:.1%}")
    print(f"Latency p50 / p95: {s['latency_p50_s']} s / {s['latency_p95_s']} s")
    print(f"Sent to review: {s['review_rate']:.0%} of tickets; errors caught by review: {s['errors_caught_by_review']}")
    print("\nRecall by category:")
    for cat, v in sorted(s["per_category_recall"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {cat:<13} {v['recall']:.0%}  ({v['tickets']} tickets)")
    print("\nCalibration (accuracy at each self-reported confidence level):")
    for b, v in s["calibration"].items():
        print(f"  confidence {b:<8} correct {v['accuracy']:.0%}  ({v['tickets']} tickets)")
    if s["top_confusions"]:
        print("\nMost common mistakes (true -> predicted):")
        for c in s["top_confusions"]:
            print(f"  {c['true_category']} -> {c['predicted_category']}: {c['count']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60, help="number of tickets (spread evenly over categories)")
    p.add_argument("--model", choices=["prod", "dev"], default="prod")
    p.add_argument("--delay", type=float, default=10.0, help="seconds between calls")
    p.add_argument("--seed", type=int, default=42, help="which random sample to draw")
    p.add_argument("--exclude", help="a previous eval_results CSV whose tickets must not be reused")
    args = p.parse_args()
    run(args.n, args.model, args.delay, args.seed, args.exclude)