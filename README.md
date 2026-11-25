# Inkling-2-Mini on twin RTX 3090s over NVLink

A Docker Compose setup that serves
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with vLLM on two NVLink-bridged RTX 3090s: tensor parallel across both cards,
262,144-token context, speculative decoding from the checkpoint's own MTP
head, and an OpenAI-compatible API on `localhost:8000`.

Measured on this setup: 125 tok/s for a single stream, 983 tok/s aggregate at
32 concurrent streams, and a correct answer from a cold 248,773-token prompt.

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
git clone <this repo> && cd inkling-2-mini-twin-3090-nvlink
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

## What you get

| Setting | Value | Why |
|---|---|---|
| vLLM | nightly `0.34.0rc1.dev1235+gf2f9edcac` | Inkling 2 is not in a tagged release yet (v0.34.0 lacks it) |
| Weights | the official checkpoint, about 19.5 GB | NVFP4 experts in layers 0–19, MXFP8 in 20–23, everything else BF16. vLLM rejects BF16 experts, so no higher-precision variant can be loaded |
| KV cache | BF16 | the only dtype vLLM allows off Blackwell |
| Tensor parallel | 2, NCCL over NVLink | |
| Context | 262,144 tokens | |
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
- **Long prompts.** Cold prefill runs at about 5,200 tok/s for a 30k prompt
  and 2,700 tok/s for a 250k prompt, which is 92 seconds to first token at
  250k. Prefix caching is on, so a repeated prefix is not paid for twice.

## Configuration

Every tunable is an environment variable with the measured default. Copy
`.env.example` to `.env` and uncomment what you want to change, then
`docker compose up -d`.

| Variable | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | lower it to trade context for KV headroom |
| `MAX_NUM_SEQS` | `32` | concurrent requests in the engine |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | prefill chunk; do not raise at 262,144 |
| `SPEC_TOKENS` | `7` | `7` or `0` (off) |
| `GPU_MEM_UTIL` | empty | empty lets vLLM size the KV cache itself |
| `LONG_PREFILL_THRESHOLD` | `32768` | prompts above this are scheduled as long prefills |
| `ENABLE_PREFIX_CACHING` | `1` | |
| `EXTRA_ARGS` | empty | extra `vllm serve` flags, whitespace-split |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | host directory for weights |
| `VLLM_VERSION` | the validated nightly tag | `nightly` tracks upstream |
| `INKLING_SM86_PATCH` | `1` | `0` builds an unpatched image (Hopper or newer) |

Changing `VLLM_VERSION`, `INKLING_SM86_PATCH`, `DOCKER_UID` or `DOCKER_GID`
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
| Startup fails saying the KV cache cannot hold `max_model_len` | something else is using GPU memory, or `MAX_NUM_BATCHED_TOKENS` was raised |
| `Inkling2 DSpark requires num_speculative_tokens = 7` | `SPEC_TOKENS` must be 7 or 0 |
| Permission errors under `/hf-cache` | `DOCKER_UID`/`DOCKER_GID` do not match the owner of `HF_CACHE_DIR` |
| Engine hangs at startup | `ipc: host` was removed; the two workers need a large `/dev/shm` |

## Files

| Path | Purpose |
|---|---|
| `docker-compose.yml` | the service: GPUs, caches, tunables, healthcheck |
| `Dockerfile` | pinned nightly base, host-uid user, patch, entrypoint |
| `inkling_sm86_patch.py` | the four build-time edits, with rationale |
| `entrypoint.sh` | turns the environment variables into `vllm serve` flags |
| `scripts/preflight.sh` | host checks before the first start |
| `scripts/smoke.sh` | end-to-end check of a running server |
| `bench/` | throughput benchmark and correctness eval |
| `results/` | measurements from the validated setup |
