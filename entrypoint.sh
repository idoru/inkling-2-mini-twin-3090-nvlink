#!/usr/bin/env bash
set -euo pipefail

# Inkling-2-Mini on twin RTX 3090 (SM86) over NVLink. Every tunable is an
# env var with a measured default (see results/RESULTS.md).

MODEL="${MODEL:-thinkingmachines/Inkling-2-Mini}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-Inkling-2-Mini inkling-2-mini thinkingmachines/Inkling-2-Mini}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# Measured on 2x3090: aggregate throughput still rises to 64 streams (1160
# tok/s) but per-stream speed halves past 32; 32 is the default balance.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
# Prefill chunk. This is what makes 262144 fit: the sliding-window layers keep
# window + chunk tokens of KV per request, and at vLLM's default (2048) one
# full-length request needs more KV than two 24 GB cards have left.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
# auto = BF16. fp8 (E4M3) halves KV memory; below Blackwell it needs the
# image's FP8 KV patch (inkling_fp8kv_patch.py).
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Empty = let vLLM's extensible KV cache size itself from what is free after
# warmup (the v0.34+ default and recommendation). Set a fraction to pin it.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
# Speculative decoding. The MTP head shipped in the repo (mtp.safetensors) is
# a DSpark block-drafter and only accepts SPEC_TOKENS=7 (its block size - 1);
# SPEC_TOKENS=0 disables speculation entirely.
SPEC_METHOD="${SPEC_METHOD:-dspark}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
SPEC_DRAFT_SAMPLE="${SPEC_DRAFT_SAMPLE:-probabilistic}"
LONG_PREFILL_THRESHOLD="${LONG_PREFILL_THRESHOLD:-32768}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

EXTRA=()
if [[ "$SPEC_TOKENS" != "0" ]]; then
  SPEC="{\"method\":\"${SPEC_METHOD}\",\"num_speculative_tokens\":${SPEC_TOKENS}"
  if [[ -n "$SPEC_DRAFT_SAMPLE" ]]; then
    SPEC+=",\"draft_sample_method\":\"${SPEC_DRAFT_SAMPLE}\""
  fi
  SPEC+="}"
  EXTRA+=(--speculative-config "$SPEC")
fi
if [[ -n "$GPU_MEM_UTIL" ]]; then
  EXTRA+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi
if [[ "$LONG_PREFILL_THRESHOLD" != "0" ]]; then
  EXTRA+=(--long-prefill-token-threshold "$LONG_PREFILL_THRESHOLD")
fi
if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  EXTRA+=(--enable-prefix-caching)
fi

# Free-form extra `vllm serve` flags (whitespace-split, so no spaces inside a
# value). Handy for experiments without editing the compose file.
# shellcheck disable=SC2206
EXTRA+=(${EXTRA_ARGS:-})

# Unquoted on purpose: --served-model-name takes a space-separated list.
# shellcheck disable=SC2206
SERVED_NAMES=(${SERVED_MODEL_NAMES})

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --reasoning-parser inkling \
  --enable-auto-tool-choice \
  --tool-call-parser inkling \
  --served-model-name "${SERVED_NAMES[@]}" \
  "${EXTRA[@]}" \
  "$@"
