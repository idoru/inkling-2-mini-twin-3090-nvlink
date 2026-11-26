# Inkling-2-Mini on twin RTX 3090s over NVLink

A Docker Compose setup that serves
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with vLLM on two NVLink-bridged RTX 3090s: tensor parallel across both cards,
speculative decoding from the checkpoint's own MTP head, and an
OpenAI-compatible API on `localhost:8000`. The default serves 262,144 tokens
of context to 32 concurrent requests; [other profiles](#choosing-a-configuration)
trade concurrency for context, up to the model's full 1,048,576 tokens.

Measured on the default configuration:

| | |
|---|---|
| Decode, 1 stream | 125 tok/s |
| Decode, 32 streams | 983 tok/s aggregate (37 tok/s per stream) |
| Prefill, 8k–32k prompt | about 5,200–5,400 tok/s (32k tokens in 6 s) |
| Prefill, 128k prompt | 3,800 tok/s (33 s) |
| Prefill, 250k prompt | 2,700 tok/s (91 s), and the needle in it is found |

> **This is not a configuration vLLM supports.** Upstream runs Inkling 2 on
> Hopper and Blackwell GPUs only and refuses to start on a 3090. The image
> here applies four small source edits at build time to make it run on Ampere.
> See [The SM86 patch](#the-sm86-patch) before relying on it.

## Requirements

| | Needed | Validated on |
|---|---|---|
| GPUs | 2 × RTX 3090 (24 GB each) with an NVLink bridge | 2 × RTX 3090, 4 links per card at 14.062 GB/s |
| NVIDIA driver | 580 or newer (the image is built for CUDA 13) | 620.73.04 |
| Docker | Engine with Compose 2.30 or newer (`gpus: all`) | Engine 29.10.0, Compose 5.8.0 |
| NVIDIA Container Toolkit | installed and registered with Docker | 1.20.2 |
| Disk | about 20 GB for weights, about 25 GB for the image | |
| OS | Linux | Arch, kernel 7.3.5 |

Both GPUs must be otherwise idle: at 262,144 context the server uses nearly
all 24 GB on each card. A desktop session on one of them may not leave enough.

## Quick start

```sh
git clone https://github.com/idoru/inkling-2-mini-twin-3090-nvlink.git && cd inkling-2-mini-twin-3090-nvlink
scripts/preflight.sh          # checks GPUs, NVLink, driver, Docker, disk
docker compose up -d --build  # builds the patched image, downloads weights on first start
docker compose logs -f vllm   # wait for "Application startup complete"
scripts/smoke.sh              # model list, one chat completion, one tool call
```

The first start downloads the checkpoint (about 20 GB) into
`~/.cache/huggingface` and compiles kernels. After that a cold start takes
about 2 minutes and a warm restart about 45 seconds.

If your user is not uid/gid 1000, copy `.env.example` to `.env` and set
`DOCKER_UID` and `DOCKER_GID` before building.

```sh
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "reasoning_effort": "low",
  "messages": [{"role": "user", "content": "Explain NVLink in two sentences."}]
}'
```

The model is served as `inkling-2-mini`, `Inkling-2-Mini` and
`thinkingmachines/Inkling-2-Mini`. `reasoning_effort` accepts `none`,
`minimal`, `low`, `medium`, `high` (the default), `xhigh` and `max`. Tool
calling and image input are enabled.

## Choosing a configuration

Two 24 GB cards hold the weights (about 10 GB each) plus a fixed budget of KV
cache. That budget can be spent on many short requests or on a few long
ones, so the recipe ships as a set of profiles. Pick one by passing its file
to Compose:

```sh
docker compose up -d                                        # default
docker compose --env-file profiles/ctx-512k.env up -d       # a named profile
```

| Profile | Max context | Concurrent requests | KV cache | Speculation | Full-length requests that fit at once |
|---|---:|---:|---|---|---:|
| default (no file) | 262,144 | 32 | BF16 | DSpark | 1.16 |
| `throughput` | 262,144 | 64 | BF16 | off | 1.58 |
| `ctx-512k` | 524,288 | 4 | BF16 | DSpark | 1.02 |
| `ctx-768k` | 786,432 | 4 | FP8 | off | 1.63 |
| `ctx-1m` | 1,048,576 | 4 | FP8 | off | 1.26 |

"Concurrent requests" is how many the engine will run at the same time.
The last column is how many requests of the *maximum* length the KV cache can
hold together: all profiles take one full-length request at a time (plus
shorter ones alongside it), and none takes two.

### What each profile measured

Decode is on short prompts, 512 output tokens per stream. Prefill is one cold
request with nothing in the prefix cache; time to first token is the prefill.

| Profile | Decode, 1 stream | Decode, all streams | Prefill 64k | Prefill near the maximum |
|---|---:|---:|---:|---|
| default | 125 tok/s | 983 tok/s at 32 | 4,748 tok/s (13 s) | 249k: 2,735 tok/s (91 s) |
| `throughput` | 120 tok/s | 1,852 tok/s at 64 | 4,665 tok/s (14 s) | 249k: 2,712 tok/s (92 s) |
| `ctx-512k` | 131 tok/s | 347 tok/s at 4 | 4,176 tok/s (15 s) | 497k: 1,671 tok/s (5 min) |
| `ctx-768k` | 105 tok/s | 292 tok/s at 4 | 2,565 tok/s (25 s) | 746k: 427 tok/s (29 min) |
| `ctx-1m` | 105 tok/s | 291 tok/s at 4 | 2,559 tok/s (25 s) | 995k: 328 tok/s (51 min) |

A needle placed in each near-maximum prompt was retrieved correctly in every
profile, including at 995,019 tokens.

### How to choose

- **Most requests at once:** `throughput`. Speculation only helps below about
  8 streams, so turning it off buys both KV room and aggregate speed.
- **Interactive use, a few streams:** the default. DSpark makes one to four
  streams 14–19% faster.
- **Up to 512k tokens:** `ctx-512k`. Nothing is quantized beyond the official
  checkpoint and speculation stays on. Without quantizing the KV cache the
  ceiling is about 670k tokens (speculation off).
- **Beyond that:** `ctx-768k` or `ctx-1m`, which store the KV cache in 8-bit
  floats (see [FP8 KV cache](#fp8-kv-cache)).

Long prompts are slow to read the first time. Four of the model's 24 layers
attend over the whole prompt, so prefill speed falls as the prompt grows: a
cold 1M-token prompt takes about 51 minutes before the first output token,
and decoding at that depth runs at about 25 tok/s. Prefix caching is on in
every profile, so follow-up requests that share the prefix skip that cost.

### Other combinations that fit

Each row was booted to confirm it; only the profiles above were benchmarked.
Raw probe output is in `results/profiles/fit_probe.txt`.

| Max context | KV cache | Speculation | Concurrent requests | Prefill chunk | Full-length requests that fit |
|---:|---|---|---:|---:|---:|
| 262,144 | BF16 | off | 8 | 512 | 2.26 |
| 262,144 | FP8 | DSpark | 32 | 1024 | 1.74 |
| 262,144 | FP8 | off | 32 | 1024 | 2.57 |
| 524,288 | BF16 | off | 4 | 512 | 1.25 |
| 524,288 | FP8 | DSpark | 4 | 512 | 1.74 |
| 524,288 | FP8 | off | 8 | 512 | 2.28 |
| 786,432 | BF16 | off | 2 | 512 | does not fit |
| 786,432 | FP8 | DSpark | 2 | 512 | 1.22 |
| 1,048,576 | FP8 | DSpark | 1 | 256 | 1.02 |

To build your own, copy a file in `profiles/` and change it. The variables
that matter are `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `KV_CACHE_DTYPE`,
`SPEC_TOKENS` and `MAX_NUM_BATCHED_TOKENS`. If a combination does not fit,
the server exits at startup with the largest context that would.

Compose reads `.env` only when no `--env-file` is given. To combine your own
`.env` (for example `DOCKER_UID`) with a profile, pass both:
`docker compose --env-file .env --env-file profiles/ctx-1m.env up -d`.

### FP8 KV cache

`KV_CACHE_DTYPE=fp8` stores attention keys and values as 8-bit E4M3 floats
instead of BF16. That halves the memory per cached token, which is what makes
768k and 1M fit. It is the same format vLLM uses for this model on Blackwell
GPUs; keys and values are normalized before they are stored, so no per-tensor
scale is involved.

What it costs:

- **Precision.** Three mantissa bits per stored value instead of seven.
  In the checks run here it made no measurable difference: the correctness
  eval passed 20/20, needles were retrieved up to 995k tokens, and on
  AA-Omniscience (300 closed-book questions) FP8 scored 0.9 against
  −0.2 for BF16, well inside the 3-point spread between two identical runs.
  That is not a guarantee for every workload; use BF16 when the context fits.
- **Speed on long prompts.** On a 3090 the conversion runs in software inside
  the attention kernel. Prefill is about 1.8× slower at 64k tokens and about
  2.5× slower at 250k than BF16. Decode on short prompts is little
  affected: 105 tok/s for one stream, against 111–120 for BF16 with
  speculation off (the default's 125 includes speculation).

vLLM allows FP8 KV for this model on Blackwell only, because both directions
of the conversion use instructions a 3090 lacks. A third build-time patch,
[`inkling_fp8kv_patch.py`](inkling_fp8kv_patch.py), replaces them with
integer arithmetic. `tests/test_e4m3.py` checks that arithmetic against
PyTorch's `float8_e4m3fn`: decoding matches on every finite byte, and
encoding matches on 8 million samples except one exact rounding tie in the
subnormal range. The patch does nothing unless `KV_CACHE_DTYPE=fp8` is set.

## What you get

| Setting | Value | Why |
|---|---|---|
| vLLM | nightly `0.34.0rc1.dev1235+gf2f9edcac` | Inkling 2 is not in a tagged release yet (v0.34.0 lacks it) |
| Weights | the official checkpoint, about 19.5 GB | NVFP4 experts in layers 0–19, MXFP8 in 20–23, everything else BF16. vLLM rejects BF16 experts, so no higher-precision variant can be loaded |
| KV cache | BF16 | unquantized; FP8 is available for the long-context profiles |
| Tensor parallel | 2, NCCL over NVLink | |
| Context | 262,144 tokens | up to 1,048,576 with a [profile](#choosing-a-configuration) |
| Speculation | DSpark, 7 draft tokens | the bundled `mtp.safetensors` head is a DSpark drafter; vLLM rejects `method: "mtp"` and any draft length other than 7 |
| `MAX_NUM_SEQS` | 32 | see [Throughput](#throughput) |
| `MAX_NUM_BATCHED_TOKENS` | 1024 | required for 262,144 to fit; see below |

### Why `MAX_NUM_BATCHED_TOKENS=1024` matters

Twenty of the model's 24 layers use sliding-window attention and keep the
window plus one prefill chunk of KV per request. At vLLM's default chunk of
2048, a single full-length request needs more KV memory than two 24 GB cards
have left after the weights, and the server refuses to start at 262,144.
Halving the chunk makes it fit, with room for 304,385 tokens of KV in total.

That total is shared by all running requests. One request can use the full
262,144, or many smaller requests can run side by side, but two full-length
requests cannot run at once.

## Throughput

512 output tokens per stream, a different prompt per stream, temperature 1.0,
default reasoning effort. Full tables and raw data are in
[`results/`](results/RESULTS.md).

| Concurrent streams | Aggregate tok/s | Per stream tok/s |
|---:|---:|---:|
| 1 | 125 | 134 |
| 4 | 334 | 98 |
| 8 | 516 | 76 |
| 16 | 730 | 55 |
| 32 | 983 | 37 |
| 48 (`MAX_NUM_SEQS=64`) | 1,046 | 26 |
| 64 (`MAX_NUM_SEQS=64`) | 1,161 | 22 |

- **Concurrency.** The default of 32 keeps each stream near 37 tok/s.
  Aggregate throughput was still rising at 64, the highest level tested, so
  set `MAX_NUM_SEQS=64` if total tokens per second matters more than
  per-stream speed. KV capacity drops from 304,385 to 284,610 tokens.
- **Speculation.** DSpark adds 14–19% at 1–4 streams and 6% at 8, and costs
  2% at 16. It also takes KV memory: 329,103 tokens with it versus 433,746
  without, at 16 sequences. For a batch-heavy workload, `SPEC_TOKENS=0` is the
  better setting.
- **Prefill.** See the table below. Prefix caching is on, so a repeated
  prefix is not paid for twice.

### Prefill

One cold request at a time (unique text, so nothing comes from the prefix
cache), default configuration. Time to first token is the whole prefill.

| Prompt tokens | Time to first token | Prefill tok/s |
|---:|---:|---:|
| 2,009 | 0.6 s | 3,507 |
| 7,999 | 1.5 s | 5,187 |
| 15,977 | 2.9 s | 5,437 |
| 31,914 | 6.0 s | 5,288 |
| 63,678 | 13.4 s | 4,748 |
| 127,361 | 33.3 s | 3,825 |
| 248,747 | 90.9 s | 2,735 |

Speed falls with length because four of the 24 layers attend over the whole
prompt. Every run returned the correct needle.

## Configuration

Every tunable is an environment variable with the measured default. Start
from a [profile](#choosing-a-configuration) if one fits; otherwise copy
`.env.example` to `.env` and uncomment what you want to change, then
`docker compose up -d`.

| Variable | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | lower it to trade context for KV headroom |
| `MAX_NUM_SEQS` | `32` | concurrent requests in the engine |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | prefill chunk; do not raise at 262,144. The long-context profiles use 512, which leaves more room for KV |
| `KV_CACHE_DTYPE` | `auto` | `auto` is BF16; `fp8` halves KV memory (see [FP8 KV cache](#fp8-kv-cache)) |
| `SPEC_TOKENS` | `7` | `7` or `0` (off) |
| `GPU_MEM_UTIL` | empty | empty lets vLLM size the KV cache itself |
| `LONG_PREFILL_THRESHOLD` | `32768` | prompts above this are scheduled as long prefills |
| `ENABLE_PREFIX_CACHING` | `1` | |
| `EXTRA_ARGS` | empty | extra `vllm serve` flags, whitespace-split |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | host directory for weights |
| `VLLM_VERSION` | the validated nightly tag | `nightly` tracks upstream |
| `INKLING_SM86_PATCH` | `1` | `0` builds an unpatched image (Hopper or newer) |
| `INKLING_FP8KV_PATCH` | `1` | `0` leaves the attention kernels untouched (no FP8 KV on a 3090) |

Changing `VLLM_VERSION`, `INKLING_SM86_PATCH`, `INKLING_FP8KV_PATCH`, `DOCKER_UID` or `DOCKER_GID`
needs a rebuild (`docker compose up -d --build`).

The compose file uses host networking and the server binds `0.0.0.0:8000`
with no API key. Firewall the port, or add `--api-key <key>` through
`EXTRA_ARGS`, if the machine is reachable from an untrusted network.

## The SM86 patch

The RTX 3090 is compute capability 8.6 (SM86). vLLM's Inkling 2
implementation is gated to SM90 and above, although the code path it uses on
Hopper (Triton attention, Humming expert GEMMs, BF16 activations) needs no
Hopper hardware feature. [`inkling_sm86_patch.py`](inkling_sm86_patch.py)
makes four edits to the installed packages during the image build:

1. **Capability gate.** Lowers the minimum capability in the Inkling 2 quant
   config from 90 to 80.
2. **Vision tower.** Uses FlashAttention 2 instead of FlashAttention 3, whose
   kernels are built for SM90 only. Upstream already takes this branch on
   SM12x.
3. **Programmatic dependent launch.** Disables PDL in Triton below SM90,
   because `ptxas` rejects the `griddepcontrol` instruction there. PDL only
   lets the next kernel start early, so this affects latency, not results.
4. **Fused conv-mix.** Uses the Triton fallback that upstream ships for SM12x
   in place of the native kernel, which is compiled for Hopper and SM10x only.

Each edit asserts that the upstream text it replaces appears exactly once. If
a newer nightly changes any of those lines, the image build fails instead of
producing a half-patched server. That is why the default image is pinned to
the nightly the patch was validated on. To move forward, set
`VLLM_VERSION=nightly`, rebuild, and re-run the checks below; if the build
fails, compare the script against the new source.

What was checked on the patched build:

- a cold needle-in-haystack at 248,773 prompt tokens, answered correctly
- `bench/eval_correctness.py`: 20/20 (five needles at about 111k tokens,
  arithmetic, code generation, JSON output)
- tool calling and image input (the vision tower is one of the patched paths)
- 4 × 60k-token and 32 × 8k-token concurrent requests with no errors

What was not checked: output parity against a Hopper or Blackwell reference,
any GPU other than the RTX 3090, and running without NVLink.

## Reproducing the numbers

```sh
python3 -m venv .venv && .venv/bin/pip install -r bench/requirements.txt

# concurrency sweep
.venv/bin/python bench/bench_inkling.py --mode conc --levels 1 2 4 8 16 24 32

# cold long-context needle (about 90 s to first token)
.venv/bin/python bench/bench_inkling.py --mode needle --ctx 250000 --max-tokens 2048

# N concurrent requests of --ctx tokens each
.venv/bin/python bench/bench_inkling.py --mode ctxconc --ctx 60000 --levels 4

# correctness regression check
.venv/bin/python bench/eval_correctness.py --output eval.json
```

The benchmark sends a different realistic prompt on each stream and counts
tokens from the API `usage` block, because speculation speedup depends on the
text being generated. Expect run-to-run variation of a few percent.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Build fails with `expected exactly one match for ...` | the vLLM source changed under the patch; use the pinned `VLLM_VERSION` |
| `quantization method inkling2 is not supported ... Minimum capability: 90` | the image was built with `INKLING_SM86_PATCH=0` |
| Startup fails saying the KV cache cannot hold `max_model_len` | something else is using GPU memory, `MAX_NUM_BATCHED_TOKENS` was raised, or the combination does not fit; the error gives the largest context that would |
| `Inkling2 KV cache dtype must be one of ('auto', 'bfloat16')` | the image was built with `INKLING_FP8KV_PATCH=0` |
| `Inkling2 DSpark requires num_speculative_tokens = 7` | `SPEC_TOKENS` must be 7 or 0 |
| Permission errors under `/hf-cache` | `DOCKER_UID`/`DOCKER_GID` do not match the owner of `HF_CACHE_DIR` |
| Engine hangs at startup | `ipc: host` was removed; the two workers need a large `/dev/shm` |

## Files

| Path | Purpose |
|---|---|
| `docker-compose.yml` | the service: GPUs, caches, tunables, healthcheck |
| `Dockerfile` | pinned nightly base, host-uid user, patch, entrypoint |
| `inkling_sm86_patch.py` | the four build-time edits, with rationale |
| `inkling_fp8kv_patch.py` | optional FP8 KV cache on pre-Blackwell GPUs |
| `profiles/` | ready-made context/concurrency configurations |
| `tests/test_e4m3.py` | reference check for the FP8 conversion |
| `entrypoint.sh` | turns the environment variables into `vllm serve` flags |
| `scripts/preflight.sh` | host checks before the first start |
| `scripts/smoke.sh` | end-to-end check of a running server |
| `bench/` | throughput benchmark and correctness eval |
| `results/` | measurements from the validated setup |

## License

[MIT](LICENSE). This covers the files in this repository only. vLLM, Triton
and the Inkling-2-Mini weights are under their own licenses.
