#!/usr/bin/env python3
"""Build-time patch: FP8 (E4M3) KV cache for Inkling2 below Blackwell.

vLLM's Inkling2 attention can keep its KV cache in 8-bit E4M3 instead of
BF16, which halves KV memory and so roughly doubles the context that fits.
Upstream allows it on SM12x only, because both directions of the conversion
use hardware instructions an RTX 3090 (SM86) does not have:

  * read:  `cvt.rn.f16x2.e4m3x2` in the attention kernel (sm_89+)
  * write: Triton's fp32 -> fp8e4nv cast in the KV conv kernel

This patch replaces both with integer arithmetic that runs anywhere, and
adds fp8 to the dtypes the model accepts. The format in the cache is the
same E4M3 the Blackwell path stores (K is norm-clipped and V RMS-normalized
before the write, so values are O(1) and need no per-tensor scale).

Nothing changes unless the server is started with --kv-cache-dtype fp8
(KV_CACHE_DTYPE=fp8). Applied when the image is built with
INKLING_FP8KV_PATCH=1. Each edit asserts on the exact upstream text.

E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits; subnormals at
exponent 0; largest finite value 448 (0x7E).
"""
import importlib.util
import pathlib


def edit(path: pathlib.Path, old: str, new: str) -> None:
    src = path.read_text()
    assert src.count(old) == 1, f"{path}: expected exactly one match for {old!r}"
    path.write_text(src.replace(old, new))
    print(f"patched {path}")


inkling = pathlib.Path(importlib.util.find_spec("vllm").origin).parent / (
    "models/inkling2/nvidia"
)

# 1. Accept fp8 off Blackwell.
edit(
    inkling / "attention.py",
    "        if current_platform.is_device_capability_family(120):\n"
    '            supported += ("float16", "fp8", "fp8_e4m3")\n',
    "        if current_platform.is_device_capability_family(120):\n"
    '            supported += ("float16", "fp8", "fp8_e4m3")\n'
    "        else:\n"
    "            # FP8 KV patch: software E4M3 conversions (inkling_fp8kv_patch.py).\n"
    '            supported += ("fp8", "fp8_e4m3")\n',
)

# 1b. Hand the cache to the kernels as raw bytes. Triton rejects fp8e4nv
#     pointers below sm_89; the kernels only test for an 8-bit element type.
edit(
    inkling / "attention.py",
    "            kv_cache = kv_cache.view(torch.float8_e4m3fn)\n",
    "            kv_cache = kv_cache.view(torch.uint8)  # FP8 KV patch\n",
)

# 2. Read side: decode E4M3 bytes with integer ops instead of the sm_89+
#    `cvt` instruction. Exact: every E4M3 value is representable in BF16.
edit(
    inkling / "ops/attention_sm90.py",
    "@triton.jit\n"
    "def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):\n",
    "@triton.jit\n"
    "def _e4m3_to_f32(raw):\n"
    '    """FP8 KV patch: uint32 E4M3 codes -> fp32, integer ops only."""\n'
    "    exponent = (raw >> 3) & 15\n"
    "    mantissa = raw & 7\n"
    "    # Normal: (-1)^s * 2^(e-7) * (1 + m/8), built as fp32 bits.\n"
    "    bits = ((raw >> 7) << 31) | ((exponent + 120) << 23) | (mantissa << 20)\n"
    "    normal = bits.to(tl.float32, bitcast=True)\n"
    "    # Subnormal (e == 0): (-1)^s * m * 2^-9.\n"
    "    subnormal = mantissa.to(tl.float32) * 0.001953125\n"
    "    subnormal = tl.where(raw >= 128, -subnormal, subnormal)\n"
    "    return tl.where(exponent == 0, subnormal, normal)\n"
    "\n"
    "\n"
    "@triton.jit\n"
    "def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):\n",
)
edit(
    inkling / "ops/attention_sm90.py",
    "    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.\n"
    "    half = tl.arange(0, D // 2)\n"
    "    pair = tl.load(\n"
    "        ptr.to(tl.pointer_type(tl.uint16))\n"
    "        + tl.multiple_of(offsets // 2, D // 2)[:, None]\n"
    "        + half[None, :],\n"
    "        mask=mask[:, None],\n"
    "        other=0,\n"
    "    )\n"
    "    lo, hi = tl.inline_asm_elementwise(\n"
    '        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",\n'
    '        "=h,=h,h",\n'
    "        [pair],\n"
    "        dtype=(tl.float16, tl.float16),\n"
    "        is_pure=True,\n"
    "        pack=1,\n"
    "    )\n"
    "    return tl.interleave(lo, hi).to(dtype)\n",
    "    # FP8 KV patch: software E4M3 decode (no sm_89+ `cvt`). Same 16-bit\n"
    "    # pair loads as upstream, so the memory access pattern is unchanged.\n"
    "    half = tl.arange(0, D // 2)\n"
    "    pair = tl.load(\n"
    "        ptr.to(tl.pointer_type(tl.uint16))\n"
    "        + tl.multiple_of(offsets // 2, D // 2)[:, None]\n"
    "        + half[None, :],\n"
    "        mask=mask[:, None],\n"
    "        other=0,\n"
    "    ).to(tl.uint32)\n"
    "    lo = _e4m3_to_f32(pair & 255).to(dtype)\n"
    "    hi = _e4m3_to_f32(pair >> 8).to(dtype)\n"
    "    return tl.interleave(lo, hi)\n",
)

# 3. Write side: encode fp32 to E4M3 (round to nearest even, saturating at
#    448) with integer ops instead of Triton's fp32 -> fp8e4nv cast.
sconv = inkling / "ops/sconv.py"
edit(
    sconv,
    "    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n",
    "    if key_cache_ptr.dtype.element_ty.primitive_bitwidth == 8:\n"
    "        # FP8 KV patch: software E4M3 encode.\n"
    "        magnitude = tl.minimum(tl.abs(y), 448.0)\n"
    "        bits = magnitude.to(tl.uint32, bitcast=True)\n"
    "        # Keep 3 mantissa bits, round to nearest even; a mantissa carry\n"
    "        # runs into the exponent field, which is what E4M3 wants.\n"
    "        rounded = (bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20\n"
    "        normal = rounded - 960  # rebias the exponent: (127 - 7) << 3\n"
    "        subnormal = (magnitude * 512.0 + 0.5).to(tl.uint32)\n"
    "        code = tl.where(magnitude < 0.015625, subnormal, normal)\n"
    "        code = tl.minimum(code, 126)\n"
    "        code = tl.where(y < 0, code | 128, code)\n"
    "        tl.store(\n"
    "            dst.to(tl.pointer_type(tl.uint8)),\n"
    "            code.to(tl.uint8),\n"
    "            mask=(kv_slot >= 0)[:, None],\n"
    "        )\n"
    "    else:\n"
    "        tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n",
)
