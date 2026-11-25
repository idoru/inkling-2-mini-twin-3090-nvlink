# Inkling-2-Mini — 2026-11-24 (vLLM nightly 0.34.0rc1.dev1235+gf2f9edcac, 2× RTX 3090 NVLink, TP=2)

Config (docker-compose.yml defaults): official checkpoint (NVFP4 experts L0–19, MXFP8 L20–23,
everything else BF16), max-model-len 262144, BF16 KV, DSpark k=7 off the bundled `mtp.safetensors`,
max-num-seqs 32, max-num-batched-tokens 1024, prefix caching on. Image built with INKLING_SM86_PATCH=1
(base `vllm/vllm-openai:nightly-f2f9edcac30d128bc090a3c032e4ba125716a641`). Driver 620.73.04.
Bench: `bench/bench_inkling.py` — chat completions, distinct realistic prompts, 512 out tokens, default
temperature (1.0), default reasoning effort; tokens from API `usage`.

## Concurrency sweep (final defaults) — conc_dspark7_seqs32.json
| conc | agg tok/s | per-stream | slowest stream | avg TTFT s |
|---:|---:|---:|---:|---:|
| 1  | 125.0 | 134.1 | 134.1 | 0.28 |
| 2  | 203.5 | 135.4 | 103.5 | 0.07 |
| 4  | 334.3 | 97.7  | 85.0  | 0.10 |
| 8  | 515.7 | 75.6  | 65.6  | 0.13 |
| 16 | 730.3 | 55.2  | 46.4  | 0.19 |
| 24 | 827.8 | 40.9  | 35.1  | 0.28 |
| 32 | 983.4 | 36.6  | 31.3  | 0.34 |

max-num-seqs 64 (conc_dspark7_seqs64.json): 48 → 1046 tok/s (26.0/stream), 64 → 1161 tok/s
(21.8/stream). Still fits 262k (KV 284,610 tok, 1.09×).

## Speculation on vs off (max-num-seqs 16) — off: conc_nospec_seqs16.json
| conc | DSpark k=7 | off | Δ |
|---:|---:|---:|---:|
| 1  | 128.3 | 111.9 | +15% |
| 2  | 209.4 | 176.0 | +19% |
| 4  | 346.3 | 304.2 | +14% |
| 8  | 527.9 | 496.5 | +6%  |
| 16 | 767.5 | 784.0 | −2%  |

Acceptance is low at temperature 1.0: ~1.1–1.5 of 7 drafted tokens accepted per step. k is not tunable
(vLLM: "Inkling2 DSpark requires num_speculative_tokens = 7"), and `method: "mtp"` is rejected for this
checkpoint — the MTP head is a DSpark drafter. Speculation costs KV: 433,746 tok (1.65×) off vs
329,103 tok (1.26×) on, at max-num-seqs 16.

## KV capacity at 262144 (per GPU ≈ 23.58 GiB; weights 10.0 GiB)
| max-num-seqs | batched tokens | images | KV tokens | × 262k |
|---:|---:|---|---:|---:|
| 16 | 2048 (default) | on | — | does not fit (est. max len 231,936) |
| 8  | 2048 | on  | — | does not fit (258,560) |
| 8  | 2048 | off | 277,826 | 1.06× |
| 8  | 1024 | on  | 345,346 | 1.32× |
| 16 | 1024 | on  | 329,103 | 1.26× |
| **32** | **1024** | **on** | **304,385** | **1.16×** |
| 64 | 1024 | on  | 284,610 | 1.09× |

## Long context (final defaults)
- Needle, cold, unique filler: 248,773 prompt tok → **correct**, TTFT 92.1 s (2,701 tok/s prefill), 135 tok/s decode.
- 29,854 tok cold: TTFT 5.8 s (5,171 tok/s prefill).
- 4 × 60k concurrent (cold): all OK, TTFT avg 32 s / max 52 s.
- 8k-context stress, 300 out: 8 → 160 tok/s, 16 → 171, 32 → 175 (prefill-bound; 32 × 8k ≈ the whole KV pool).

## Correctness
- bench/eval_correctness.py (eval_correctness.json): 20/20 (5 needles @ 111,670 tok, math, code, JSON).
- Tool calling (`inkling` parser): correct `get_weather` call. Image input: shapes, colours and text read correctly.
