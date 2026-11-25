# LoopCD on Inkling-2-Mini — 2026-11-25

Same hardware, image and pinned vLLM nightly as RESULTS.md, with the image's LoopCD patch
(`inkling_loopcd_patch.py`) active (`LOOPCD=1`). Method: LoopCD-Logits from arXiv:2610.02185, weak
reference = loop pass 1 of 3 decoded through the post-loop layers, adaptive strength capped at 0.5
(`"vllm_xargs": {"loopcd": 0.5}`).

## Quality: AA-Omniscience-Public v1.1 (300 questions) — loopcd_omniscience.json

`bench/loopcd_omniscience.py`. Both arms interleaved on one `LOOPCD=1` server (speculation off), so
the request parameter is the only difference. Default temperature (1.0). Intervals are 95% paired
bootstrap over questions.

| reasoning effort | samples/arm | metric | baseline | LoopCD | Δ (95% interval) |
|---|---:|---|---:|---:|---:|
| none | 3 | index | −0.2 | 4.2 | **+4.5** (+1.6 to +7.4) |
| none | 3 | non-hallucination % | 82.2 | 87.7 | **+5.5** (+3.3 to +7.9) |
| none | 3 | accuracy % | 14.9 | 14.7 | −0.2 (−1.9 to +1.4) |
| low | 3 | index | −1.5 | 2.0 | +3.5 (−0.1 to +7.2) |
| low | 3 | non-hallucination % | 79.2 | 85.1 | **+5.9** (+2.9 to +9.3) |
| low | 3 | accuracy % | 16.0 | 14.7 | −1.3 (−2.9 to +0.2) |
| default (high) | 1 | index | −1.7 | −2.0 | −0.3 (−6.7 to +6.0) |
| default (high) | 1 | non-hallucination % | 76.5 | 77.6 | +1.1 (−4.3 to +6.5) |
| default (high) | 1 | accuracy % | 17.7 | 16.7 | −1.0 (−4.0 to +2.0) |

The gain comes from fewer wrong answers (more abstentions), not more correct ones: at effort none,
incorrect 136 → 94 and not-attempted 593 → 633 of ~899, correct 134 → 132. LoopCD answers were also
shorter: mean completion tokens 1,073 → 657 (none), 565 → 428 (low), 3,629 → 2,276 (default).

For reference, the LoopCD author's post reports −9.0 → −3.4 (index, +5.6) and 66.5% → 71.8%
(non-hallucination, +5.3) on the same public set. Those absolute numbers are not comparable with
the ones here:

- The grader here is Inkling-2-Mini itself (unguided, temperature 0), not the official grading model.
- The post does not state its sampling or reasoning settings.
- The default-effort row is a single sample per arm and cannot resolve an effect of this size.

## Cost — loopcd_throughput.json

`bench/bench_inkling.py --mode conc`, 512 output tokens, aggregate tok/s.

Speculation off, max-num-seqs 32:

| conc | LOOPCD=0 | LOOPCD=1, parameter not sent | LOOPCD=1, loopcd=0.5 |
|---:|---:|---:|---:|
| 1  | 111.0  | 98.2 (−12%)   | 87.0 (−22%)  |
| 4  | 301.6  | 271.9 (−10%)  | 245.0 (−19%) |
| 16 | 770.2  | 671.2 (−13%)  | 628.9 (−18%) |
| 32 | 1186.2 | 1033.6 (−13%) | 972.1 (−18%) |

DSpark k=7, max-num-seqs 16, LOOPCD=1: parameter not sent 121.0 / 315.1 / 631.8 at 1 / 4 / 16;
loopcd=0.5 96.2 / 291.1 / 564.8. Draft acceptance per step 1.28–1.41 without the parameter,
0.87–1.33 with it (the drafter proposes from the unguided model).

KV capacity at max-model-len 262144:

| LOOPCD | speculation | max-num-seqs | KV tokens | × 262k |
|---|---|---:|---:|---:|
| 0 | off | 32 | 427,336 | 1.63× |
| 1 | off | 32 | 348,991 | 1.33× |
| 0 | DSpark | 32 | 304,385 | 1.16× |
| 1 | DSpark | 32 | — | does not fit (est. max len 235,264) |
| 1 | DSpark | 16 | 268,367 | 1.02× |

## Correctness checks

- Requests without the parameter, and with `loopcd: 0`, give greedy output identical to each other on a `LOOPCD=1` server.
- With `LOOPCD=0` (the default) the patched image measures the same as before the patch: 126.6 tok/s at 1 stream, 977.1 at 32.
- Guided greedy output with DSpark on matches guided greedy output with it off on the probe prompts.
- `loopcd: -1` (fixed) decodes from the first pass alone and gives on-topic but repetitive, degenerate text, as expected of the weak reference.
- Cold 248,757-token needle with `loopcd: 0.5` (DSpark on, max-num-seqs 16): correct, TTFT 104.9 s (2,373 tok/s prefill).
