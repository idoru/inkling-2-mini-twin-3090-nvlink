#!/usr/bin/env python3
"""
A/B test of LoopCD on AA-Omniscience-Public v1.1 (300 questions).

Both arms run against the same server (started with LOOPCD=1) at the same
time, so the only difference between them is the request parameter:

  baseline   no vllm_xargs
  loopcd     "vllm_xargs": {"loopcd": <strength>}   (adaptive unless --fixed)

The questions, the answer prompt and the grader prompt come from the dataset
card. Differences from the official protocol, which matter when comparing
with published numbers:

  * The grader is the served model itself (unguided, temperature 0), not
    the grading model Artificial Analysis uses.
  * Sampling settings are the server defaults unless overridden here.

Metrics (per arm, over questions x samples):
  index            100 * (correct - incorrect) / total
  accuracy         correct / total
  hallucination    incorrect / (incorrect + partial + not attempted)
  non-hallucination = 1 - hallucination

Usage:
  python bench/loopcd_omniscience.py --samples 3 --output results/loopcd_omniscience.json
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import statistics

from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "inkling-2-mini")
DATASET = "ArtificialAnalysis/AA-Omniscience-Public-v1.1"
GRADES = {"A": "correct", "B": "incorrect", "C": "partial", "D": "not_attempted"}

client = AsyncOpenAI(base_url=API_BASE, api_key="test", timeout=3600, max_retries=2)


def load_benchmark():
    csv_path = hf_hub_download(DATASET, "AA-Omniscience_dataset_public.csv", repo_type="dataset")
    card = open(hf_hub_download(DATASET, "README.md", repo_type="dataset")).read()
    answer_prompt = re.search(r'OMNISCIENCE_ANSWER_PROMPT = """(.*?)"""', card, re.S).group(1).strip()
    grader = re.search(r'OMNISCIENCE_GRADER_TEMPLATE = """(.*?)"""', card, re.S).group(1).strip()
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f)), answer_prompt, grader


async def answer(sem, row, answer_prompt, arm, a):
    kw = {}
    if arm == "loopcd":
        kw["extra_body"] = {"vllm_xargs": {"loopcd": a.strength, "loopcd_adaptive": 0 if a.fixed else 1}}
    if a.temperature is not None:
        kw["temperature"] = a.temperature
    if a.effort:
        kw["reasoning_effort"] = a.effort
    system = answer_prompt.format(domain=row["domain"], subtopic=row["subtopic"])
    async with sem:
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=a.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": row["question"]}],
            **kw)
    return (r.choices[0].message.content or "").strip(), r.usage.completion_tokens


async def grade(sem, row, predicted, grader):
    prompt = (grader.replace("{question}", row["question"]).replace("{target}", row["answer"])
              .replace("{predicted_answer}", predicted))
    async with sem:
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=4096, temperature=0, reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\b([ABCD])\b", (r.choices[0].message.content or "").strip())
    return GRADES[m.group(1)] if m else "ungraded"


async def one(sem, row, sample, arm, answer_prompt, grader, a):
    try:
        predicted, tokens = await answer(sem, row, answer_prompt, arm, a)
        # An empty reply is an abstention by the grader's own definition.
        g = await grade(sem, row, predicted, grader) if predicted else "not_attempted"
    except Exception as e:  # noqa: BLE001
        predicted, tokens, g = f"ERROR: {e!r}"[:300], 0, "error"
    return {"question_id": row["question_id"], "domain": row["domain"], "sample": sample, "arm": arm,
            "predicted": predicted, "target": row["answer"], "grade": g, "completion_tokens": tokens}


def metrics(records):
    n = len(records)
    c = {g: sum(r["grade"] == g for r in records) for g in ("correct", "incorrect", "partial", "not_attempted")}
    not_correct = c["incorrect"] + c["partial"] + c["not_attempted"]
    halluc = c["incorrect"] / not_correct if not_correct else 0.0
    return {
        "n": n, **c,
        "index": round(100 * (c["correct"] - c["incorrect"]) / n, 2),
        "accuracy": round(100 * c["correct"] / n, 2),
        "non_hallucination": round(100 * (1 - halluc), 2),
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in records), 1),
    }


def paired_bootstrap(records, key, iters=5000, seed=0):
    """95% interval of (loopcd - baseline) for `key`, resampling questions."""
    by_q = {}
    for r in records:
        by_q.setdefault(r["question_id"], {"baseline": [], "loopcd": []})[r["arm"]].append(r)
    qids = list(by_q)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        pick = [by_q[q] for q in rng.choices(qids, k=len(qids))]
        base = [r for p in pick for r in p["baseline"]]
        loop = [r for p in pick for r in p["loopcd"]]
        diffs.append(metrics(loop)[key] - metrics(base)[key])
    diffs.sort()
    return [round(diffs[int(0.025 * iters)], 2), round(diffs[int(0.975 * iters)], 2)]


async def main_async(a):
    rows, answer_prompt, grader = load_benchmark()
    if a.limit:
        rows = rows[: a.limit]
    sem = asyncio.Semaphore(a.concurrency)
    jobs = [one(sem, row, s, arm, answer_prompt, grader, a)
            for s in range(a.samples) for row in rows for arm in ("baseline", "loopcd")]
    records = []
    for i, fut in enumerate(asyncio.as_completed(jobs), 1):
        records.append(await fut)
        if i % 100 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} done", flush=True)
    bad = [r for r in records if r["grade"] in ("error", "ungraded")]
    good = [r for r in records if r["grade"] not in ("error", "ungraded")]
    out = {"args": vars(a), "questions": len(rows), "excluded_error_or_ungraded": len(bad)}
    for arm in ("baseline", "loopcd"):
        out[arm] = metrics([r for r in good if r["arm"] == arm])
    out["delta_loopcd_minus_baseline"] = {
        k: {"delta": round(out["loopcd"][k] - out["baseline"][k], 2), "ci95": paired_bootstrap(good, k)}
        for k in ("index", "accuracy", "non_hallucination")}
    print(json.dumps({k: v for k, v in out.items() if k != "args"}, indent=1))
    if a.output:
        with open(a.output, "w") as f:
            json.dump({**out, "records": sorted(records, key=lambda r: (int(r["question_id"]), r["sample"], r["arm"]))}, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3, help="answers per question per arm")
    p.add_argument("--strength", type=float, default=0.5, help="loopcd value")
    p.add_argument("--fixed", action="store_true", help="fixed strength instead of adaptive")
    p.add_argument("--temperature", type=float, default=None, help="default: server/model default")
    p.add_argument("--effort", default=None, help="reasoning_effort; default: server default")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="first N questions only")
    p.add_argument("--output", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
