#!/usr/bin/env python3
"""
Throughput / concurrency / long-context benchmark for the Inkling-2-Mini profile.

Design choices:
  * /v1/chat/completions with distinct, realistic prompts per stream. DSpark
    acceptance depends on the text being drafted, so a repeated filler prompt
    through /v1/completions would overstate (or, for random tokens, understate)
    speculative-decoding speedup.
  * Output tokens come from the API `usage` block (reasoning + answer tokens).
  * Long-context filler is unique per request, so prefix caching cannot hide
    the cold prefill cost.

Modes:
  conc     concurrency sweep on short prompts          (--levels 1 2 4 8 16 ...)
  ctxconc  N concurrent requests of --ctx tokens each  (--levels, --ctx)
  needle   needle-in-haystack at --ctx tokens (cold prefill time + correctness)
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time

import requests
from openai import AsyncOpenAI

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/v1")
API_ROOT = API_BASE.rsplit("/v1", 1)[0]
MODEL = os.environ.get("MODEL", "inkling-2-mini")
client = AsyncOpenAI(base_url=API_BASE, api_key="test", timeout=3600, max_retries=0)

TASKS = [
    "Write a Python function that merges overlapping intervals, with a docstring and three unit tests.",
    "Explain how a B-tree differs from an LSM tree and when you would choose each for a storage engine.",
    "Write a Go HTTP middleware that adds request IDs and structured logging. Include a usage example.",
    "Describe the causes of the 2008 financial crisis for a first-year economics student.",
    "Implement an LRU cache in Rust with O(1) get and put, and explain the ownership choices.",
    "Write a SQL query plan walkthrough: why might an index on (a, b) not be used for WHERE b = 5?",
    "Summarize how TCP congestion control evolved from Tahoe to BBR.",
    "Write a bash script that rotates log files older than seven days and compresses them, with comments.",
    "Explain the difference between processes, threads and coroutines using a restaurant kitchen analogy.",
    "Write a TypeScript generic function `groupBy` with full type inference and a few example calls.",
    "Draft a design doc outline for adding rate limiting to a public REST API.",
    "Explain how transformers use attention, to a reader who knows linear algebra but not ML.",
    "Write a Python asyncio producer/consumer pipeline with backpressure and graceful shutdown.",
    "Compare PostgreSQL's MVCC with MySQL InnoDB's approach to isolation.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "Explain what NVLink is and why tensor parallelism benefits from it more than pipeline parallelism.",
    "Write a C function that reverses a singly linked list in place and discuss its edge cases.",
    "Give a step-by-step derivation of the quadratic formula by completing the square.",
    "Write a Kubernetes Deployment and Service manifest for a stateless web app with health probes.",
    "Explain how garbage collection works in Go, including the tri-color mark and sweep algorithm.",
    "Write a regular expression that validates ISO 8601 dates and explain each part.",
    "Describe the lifecycle of an HTTP request from browser address bar to rendered page.",
    "Write a Python class implementing a trie with insert, search and prefix-count methods.",
    "Explain eventual consistency and give two concrete examples where it causes user-visible anomalies.",
    "Write a Dockerfile for a multi-stage Rust build that produces a minimal final image.",
    "Explain how public-key cryptography enables TLS, without using any formulas.",
    "Write a JavaScript debounce and throttle implementation and explain when to use each.",
    "Outline a migration plan from a monolith to services for a mid-sized e-commerce company.",
    "Write a Python script that parses a CSV of transactions and reports monthly totals per category.",
    "Explain the CAP theorem and why 'pick two' is a misleading summary.",
    "Write unit tests in pytest for a function that converts Roman numerals to integers.",
    "Explain how a CPU branch predictor works and why mispredictions are expensive.",
]

WORDS = (
    "river mountain ledger harbor copper window engine garden signal market winter bridge "
    "pattern history letter machine forest island station journal theory village captain "
    "library morning silver compass meadow archive lantern orchard voyage summit canyon "
    "thread cabinet anchor quarry tunnel harvest beacon ribbon timber valley"
).split()


def count_tokens(text: str) -> int:
    r = requests.post(f"{API_ROOT}/tokenize", json={"model": MODEL, "prompt": text}, timeout=600)
    r.raise_for_status()
    return r.json()["count"]


def filler(target_tokens: int, seed: int, needle: str | None = None) -> str:
    """Unique numbered pseudo-prose of ~target_tokens tokens (verified via /tokenize)."""
    rng = random.Random(seed)

    def build(n: int) -> str:
        lines = []
        for i in range(n):
            w = rng.choices(WORDS, k=12)
            lines.append(f"Entry {seed}-{i}: the {w[0]} near the {w[1]} kept a {w[2]} beside the {w[3]}, "
                         f"while the {w[4]} and the {w[5]} moved toward the {w[6]} {w[7]}.")
        if needle:
            lines.insert(int(n * 0.37), needle)
        return "\n".join(lines)

    probe = build(200)
    per_line = count_tokens(probe) / 200
    n = int(target_tokens / per_line)
    rng = random.Random(seed)
    text = build(n)
    actual = count_tokens(text)
    while actual > target_tokens:  # trim until under target
        n = int(n * target_tokens / actual * 0.995)
        rng = random.Random(seed)
        text = build(n)
        actual = count_tokens(text)
    return text


async def one(messages, max_tokens: int, effort: str | None, temperature: float | None):
    kw = {}
    if effort:
        kw["reasoning_effort"] = effort
    if temperature is not None:
        kw["temperature"] = temperature
    t0 = time.perf_counter()
    ttft = None
    usage = None
    text = []
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, **kw)
    async for ev in stream:
        if ev.usage:
            usage = ev.usage
        if not ev.choices:
            continue
        d = ev.choices[0].delta
        piece = d.content or getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
        if piece and ttft is None:
            ttft = time.perf_counter() - t0
        if d.content:
            text.append(d.content)
    t1 = time.perf_counter()
    out = usage.completion_tokens if usage else 0
    ttft = ttft if ttft is not None else t1 - t0
    return {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "out": out, "ttft": ttft, "total": t1 - t0,
        "decode_tps": (out - 1) / max(t1 - t0 - ttft, 1e-6) if out > 1 else 0.0,
        "text": "".join(text),
    }


async def batch(msgs_list, max_tokens, effort, temperature):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[one(m, max_tokens, effort, temperature) for m in msgs_list],
                               return_exceptions=True)
    wall = time.perf_counter() - t0
    ok = [r for r in res if isinstance(r, dict)]
    errs = [repr(r)[:200] for r in res if not isinstance(r, dict)]
    row = {
        "conc": len(msgs_list), "ok": len(ok), "errors": errs[:2], "wall_s": round(wall, 2),
        "out_tokens": sum(r["out"] for r in ok),
        "agg_tps": round(sum(r["out"] for r in ok) / wall, 1),
    }
    if ok:
        row |= {
            "per_stream_decode_tps": round(statistics.mean(r["decode_tps"] for r in ok), 1),
            "min_stream_decode_tps": round(min(r["decode_tps"] for r in ok), 1),
            "ttft_avg": round(statistics.mean(r["ttft"] for r in ok), 2),
            "ttft_max": round(max(r["ttft"] for r in ok), 2),
            "prompt_tokens_avg": int(statistics.mean(r["prompt_tokens"] for r in ok)),
        }
    return row, ok


def spec_counters():
    """Sum the speculative-decoding counters from /metrics (0s if spec is off)."""
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    try:
        for line in requests.get(f"{API_ROOT}/metrics", timeout=10).text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                out["drafts"] += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                out["draft_tokens"] += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                out["accepted"] += float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return out


def spec_delta(a, b):
    d = {k: b[k] - a[k] for k in a}
    if d["draft_tokens"] > 0:
        return {"accept_rate": round(d["accepted"] / d["draft_tokens"], 3),
                "mean_accepted_per_step": round(d["accepted"] / max(d["drafts"], 1), 2)}
    return {}


async def mode_conc(a):
    rows = []
    for c in a.levels:
        msgs = [[{"role": "user", "content": TASKS[(i + c) % len(TASKS)] + (f" (variant {i})" if i >= len(TASKS) else "")}]
                for i in range(c)]
        s0 = spec_counters()
        row, _ = await batch(msgs, a.max_tokens, a.effort, a.temperature)
        row |= spec_delta(s0, spec_counters())
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


async def mode_ctxconc(a):
    rows = []
    for c in a.levels:
        msgs = []
        for i in range(c):
            seed = int(time.time() * 1000) % 10_000_000 + i
            doc = filler(a.ctx, seed)
            msgs.append([{"role": "user", "content": doc + "\n\nIn two paragraphs, describe the style and "
                          "recurring vocabulary of the entries above."}])
        s0 = spec_counters()
        row, _ = await batch(msgs, a.max_tokens, a.effort, a.temperature)
        row |= spec_delta(s0, spec_counters()) | {"ctx_target": a.ctx}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


async def mode_needle(a):
    seed = int(time.time() * 1000) % 10_000_000
    code = f"{random.Random(seed).randint(100000, 999999)}"
    needle = f"IMPORTANT NOTE: the secret vault access code is {code}. Remember it."
    doc = filler(a.ctx, seed, needle)
    msgs = [{"role": "user", "content": doc + "\n\nWhat is the secret vault access code mentioned in "
             "the entries above? Reply with just the number."}]
    row, ok = await batch([msgs], a.max_tokens, a.effort, a.temperature)
    ans = ok[0]["text"] if ok else ""
    row |= {"expected": code, "answer": ans.strip()[:200], "correct": code in ans, "ctx_target": a.ctx}
    if ok and ok[0]["ttft"] > 0:
        row["prefill_tok_per_s"] = round(ok[0]["prompt_tokens"] / ok[0]["ttft"])
    print(json.dumps(row), flush=True)
    return [row]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["conc", "ctxconc", "needle"])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--ctx", type=int, default=250000)
    p.add_argument("--effort", default=None, help="reasoning_effort (none/minimal/low/medium/high/...)")
    p.add_argument("--temperature", type=float, default=None, help="default: server/model default")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    rows = asyncio.run({"conc": mode_conc, "ctxconc": mode_ctxconc, "needle": mode_needle}[a.mode](a))
    if a.output:
        with open(a.output, "w") as f:
            json.dump({"args": vars(a), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
