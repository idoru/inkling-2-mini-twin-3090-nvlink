#!/usr/bin/env bash
# Host checks before `docker compose up`. Prints one line per check and exits
# non-zero if anything that would stop the server from starting is missing.
set -uo pipefail

fail=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

echo "GPUs"
if ! command -v nvidia-smi >/dev/null; then
  bad "nvidia-smi not found (install the NVIDIA driver)"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
  if (( ${#gpus[@]} < 2 )); then
    bad "found ${#gpus[@]} GPU(s); this setup needs two"
  else
    ok "${#gpus[@]} GPUs"
  fi
  for g in "${gpus[@]}"; do
    IFS=',' read -r name mem drv <<<"$g"
    mem="${mem// /}"; drv="${drv// /}"
    if (( mem < 24000 )); then bad "$name has ${mem} MiB; needs 24 GB per card"; else ok "$name, ${mem} MiB"; fi
  done
  drv_major="${drv%%.*}"
  if [[ -n "${drv_major:-}" ]] && (( drv_major < 580 )); then
    bad "driver $drv; the vLLM nightly image is built for CUDA 13 (driver 580+)"
  else
    ok "driver ${drv:-unknown} (validated on 620.73.04)"
  fi

  links=$(nvidia-smi nvlink -s 2>/dev/null | grep -c 'GB/s' || true)
  if (( links >= 8 )); then
    ok "NVLink up: $links links"
  elif (( links > 0 )); then
    warn "NVLink: only $links active links (expected 8 = 4 per card); check the bridge seating"
  else
    warn "no active NVLink links; it will still run over PCIe but the numbers in results/ will not apply (set NCCL_P2P_LEVEL= in .env)"
  fi

  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if (( used > 1500 )); then
    warn "a GPU already has ${used} MiB in use; 262144 context needs nearly all 24 GB on both cards"
  else
    ok "GPUs are free"
  fi
fi

echo "Docker"
if ! command -v docker >/dev/null; then
  bad "docker not found"
else
  ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '(daemon not reachable)')"
  cv=$(docker compose version --short 2>/dev/null || true)
  if [[ -z "$cv" ]]; then
    bad "docker compose v2 plugin not found"
  else
    cmaj="${cv%%.*}"; crest="${cv#*.}"; cmin="${crest%%.*}"
    if (( cmaj < 2 || (cmaj == 2 && cmin < 30) )); then
      bad "docker compose $cv; the compose file uses 'gpus: all' (needs 2.30+)"
    else
      ok "docker compose $cv"
    fi
  fi
  if docker info 2>/dev/null | grep -qi 'nvidia'; then
    ok "nvidia container runtime registered"
  elif command -v nvidia-ctk >/dev/null || command -v nvidia-container-cli >/dev/null; then
    ok "nvidia-container-toolkit installed"
  else
    bad "nvidia-container-toolkit not found (containers cannot see the GPUs)"
  fi
fi

echo "Storage"
cache="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
if [[ ! -d "$cache" ]]; then
  warn "$cache does not exist; create it first (mkdir -p) or Docker will create it owned by root"
  probe="$HOME"
else
  if [[ -w "$cache" ]]; then ok "$cache is writable"; else bad "$cache is not writable by $(id -un)"; fi
  probe="$cache"
fi
free_gb=$(df -BG --output=avail "$probe" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$free_gb" ]]; then
  if [[ -d "$cache/hub/models--thinkingmachines--Inkling-2-Mini" ]]; then
    ok "checkpoint already in the cache"
  elif (( free_gb < 25 )); then
    bad "${free_gb} GB free under $probe; the checkpoint is ~20 GB"
  else
    ok "${free_gb} GB free for the ~20 GB checkpoint"
  fi
fi
if [[ "$(id -u)" != "${DOCKER_UID:-1000}" ]]; then
  warn "your uid is $(id -u), not ${DOCKER_UID:-1000}; set DOCKER_UID/DOCKER_GID in .env"
fi

echo
if (( fail )); then echo "preflight: FAILED"; exit 1; fi
echo "preflight: ok"
