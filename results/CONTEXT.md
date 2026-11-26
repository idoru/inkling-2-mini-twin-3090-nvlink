# Context / concurrency profiles — 2026-11-25/26

Same hardware and pinned vLLM nightly as RESULTS.md; image built with all patches
(INKLING_SM86_PATCH=1, INKLING_FP8KV_PATCH=1). Raw data: profiles/benchmarks.json,
profiles/fit_probe.txt.

## What fits (profiles/fit_probe.txt)

Each row is a real boot at that max-model-len; "×" is vLLM's "Maximum concurrency for N tokens
per request" after warmup. KV token counts are only comparable at the same max-model-len.

| max-model-len | KV dtype | spec | batched tok | max-num-seqs | KV tokens | × |
|---:|---|---|---:|---:|---:|---:|
| 262,144 | bf16 | DSpark | 1024 | 32 | 304,385 | 1.16 (default) |
| 262,144 | bf16 | off | 1024 | 64 | 415,228 | 1.58 (`throughput`) |
| 262,144 | bf16 | DSpark | 512 | 8 | 476,043 | 1.82 |
| 262,144 | bf16 | off | 512 | 8 | 592,411 | 2.26 |
| 262,144 | fp8 | DSpark | 1024 | 32 | 456,675 | 1.74 |
| 262,144 | fp8 | off | 1024 | 32 | 674,477 | 2.57 |
| 262,144 | fp8 | off | 1024 | 64 | 651,882 | 2.49 |
| 524,288 | bf16 | DSpark | 512 | 4 | 534,178 | 1.02 (`ctx-512k`) |
| 524,288 | bf16 | off | 512 | 4 | 654,599 | 1.25 |
| 524,288 | fp8 | DSpark | 512 | 4 | 912,550 | 1.74 |
| 524,288 | fp8 | off | 512 | 8 | 1,197,525 | 2.28 |
| 786,432 | bf16 | off | 512 | 2 | — | does not fit (est. max 671,616) |
| 786,432 | fp8 | DSpark | 512 | 2 | 956,939 | 1.22 |
| 786,432 | fp8 | off | 512 | 4 | 1,277,973 | 1.63 (`ctx-768k`) |
| 1,048,576 | fp8 | off | 512 | 4 | 1,318,295 | 1.26 (`ctx-1m`) |
| 1,048,576 | fp8 | DSpark | 512 | 1 | — | does not fit (est. max 974,464) |
| 1,048,576 | fp8 | DSpark | 256 | 1 | 1,073,107 | 1.02 |

The prefill chunk matters as much as the dtype: 512 instead of 1024 leaves about 1 GiB more per
GPU for KV (less transient activation memory, less per-request sliding-window KV).

## Profile benchmarks (profiles/benchmarks.json)

`bench/bench_inkling.py`. Decode: `--mode conc`, 512 output tokens, default temperature and
reasoning effort. Needles: `--mode needle --plain-needle --effort low`, cold, unique filler.

| profile | conc → agg tok/s (per-stream) | needle 64k: TTFT, prefill tok/s | needle near max: tokens, TTFT, prefill tok/s, result |
|---|---|---|---|
| `throughput` | 1 → 85.5* (120.4); 8 → 499.5 (63.8); 32 → 1098.2 (39.0); 64 → 1852.4 (29.9) | 13.7 s, 4,665 | 248,770; 91.7 s; 2,712; correct |
| `ctx-512k` | 1 → 130.8 (141.8); 2 → 211.8 (134.1); 4 → 347.3 (94.5) | 15.3 s, 4,176 | 497,506; 297.8 s; 1,671; correct |
| `ctx-768k` | 1 → 105.3 (112.9); 2 → 167.9 (86.1); 4 → 291.6 (74.8) | 24.8 s, 2,565 | 746,202; 1,745.9 s; 427; correct |
| `ctx-1m` | 1 → 105.2 (112.9); 2 → 167.4 (85.9); 4 → 291.3 (74.6) | 24.9 s, 2,559 | 995,019; 3,034.4 s; 328; correct |

\* first request after boot (TTFT 1.74 s); per-stream decode is the steady-state figure.

DSpark acceptance on `ctx-512k`: 1.26–1.43 of 7 drafted tokens per step.

eval_correctness.py (5 needles at ~111k tokens, math, code, JSON): 20/20 on `ctx-512k` (BF16) and
20/20 on `ctx-768k` (FP8).

## FP8 prefill curve (1M max-model-len, 1 sequence, speculation off, 512-token chunks)

| prompt tok | TTFT | prefill tok/s | decode tok/s at that depth | needle |
|---:|---:|---:|---:|---|
| 8,023 | 2.1 s | 3,826 | 110 | correct |
| 63,706 | 24.4 s | 2,607 | 93 | correct |
| 248,736 | 226.5 s | 1,098 | 60 | correct |
| 497,466 | 809.8 s | 614 | 41 | correct |
| 995,058 | 3,034.0 s | 328 | 25 | refused* |

\* With the benchmark's default wording ("secret vault access code") the model answered "I can’t
reveal the secret vault access code." The rerun with `--plain-needle` on `ctx-1m` (995,019 tokens)
returned the correct number. Decode figures come from short answers (65–460 tokens) and are
approximate.

For comparison, BF16 prefill on the default profile: 4,748 tok/s at 64k and 2,735 at 249k
(RESULTS.md), i.e. FP8 is about 1.8× slower at 64k and 2.5× slower at 250k. The E4M3 conversion
runs in software on SM86.

## FP8 conversion (tests/test_e4m3.py)

Against torch `float8_e4m3fn` on an RTX 3090: decode matches on all 254 finite byte values; encode
matches on 8,000,011 samples (four scales plus edge cases) except x = 2^-10, an exact tie in the
subnormal range that this encoder rounds up and torch rounds to even. One further byte differs
with the same value (-0.0 is stored as +0.0).

## FP8 quality check (profiles/fp8_quality_omniscience.json)

AA-Omniscience-Public v1.1, reasoning effort none, default temperature, graded by the served model
(harness from the LoopCD branch, parameter unused). Server: default profile with
KV_CACHE_DTYPE=fp8, SPEC_TOKENS=0.

| KV | answers | index | accuracy % | non-hallucination % |
|---|---:|---:|---:|---:|
| BF16 (earlier run) | 898 | −0.2 | 14.9 | 82.2 |
| FP8, run 1 | 900 | 2.4 | 16.4 | 83.2 |
| FP8, run 2 | 898 | −0.7 | 15.7 | 80.6 |
| FP8, pooled | 1,798 | 0.9 | 16.1 | 81.9 |

The two FP8 runs are the same configuration and differ by 3.1 index points (95% paired interval
−6.7 to +0.3), which is the noise floor of this benchmark at 3 samples per question. FP8 and BF16
are not distinguishable on it.
