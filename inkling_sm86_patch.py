#!/usr/bin/env python3
"""Build-time patch: let vLLM's Inkling2 implementation run on Ampere (SM86).

Upstream supports Inkling2 on Hopper (SM90) and Blackwell only. Its "Hopper"
code path is Triton attention + Humming expert GEMMs with BF16 activations,
which needs no SM90 hardware feature -- but four things assume SM90 anyway.
Each edit below asserts on the exact upstream text, so a nightly that moves
any of them fails the image build instead of producing a half-patched server.

Applied only when the image is built with INKLING_SM86_PATCH=1.
"""
import importlib.util
import pathlib


def path_of(module: str) -> pathlib.Path:
    return pathlib.Path(importlib.util.find_spec(module).origin)


def edit(path: pathlib.Path, old: str, new: str) -> None:
    src = path.read_text()
    assert src.count(old) == 1, f"{path}: expected exactly one match for {old!r}"
    path.write_text(src.replace(old, new))
    print(f"patched {path}")


vllm_root = path_of("vllm").parent
triton_root = path_of("triton").parent

# 1. Capability gate. Inkling2QuantConfig refuses anything below SM90.
edit(
    vllm_root / "models/inkling2/quant_config.py",
    "        return 90\n",
    "        return 80\n",
)

# 2. Vision tower. It calls FlashAttention-3 (built for sm90 only: "no kernel
#    image is available") on everything that is not SM100/SM12x. Send non-Hopper
#    devices down the FA2 branch upstream already uses for SM12x.
edit(
    vllm_root / "models/inkling2/nvidia/vision.py",
    "version = 2 if current_platform.is_device_capability_family(120) else 3",
    "version = 3 if current_platform.is_device_capability_family(90) else 2",
)

# 3. Programmatic Dependent Launch. The Inkling2 Triton ops launch with
#    launch_pdl=True and emit `griddepcontrol` unconditionally; ptxas rejects
#    that below sm_90. PDL only lets the next kernel *start early*; without it
#    kernels on a stream simply run back to back, so dropping the instruction
#    and the launch attribute below SM90 changes latency, not results.
edit(
    triton_root / "backends/nvidia/compiler.py",
    '        capability = int(self._parse_arch(args["arch"]))\n',
    '        capability = int(self._parse_arch(args["arch"]))\n'
    "        if capability < 90:\n"
    '            args["launch_pdl"] = False  # SM86 patch: no PDL below Hopper\n',
)
GUARD = (
    "    from triton.runtime import driver as _driver\n"
    "    if int(_driver.active.get_current_target().arch) < 90:\n"
    "        return  # SM86 patch: griddepcontrol needs sm_90\n"
)
for asm in ("griddepcontrol.wait; // dummy $0", "griddepcontrol.launch_dependents; // dummy $0"):
    edit(
        triton_root / "language/extra/cuda/gdc.py",
        f'    core.inline_asm_elementwise("{asm}"',
        GUARD + f'    core.inline_asm_elementwise("{asm}"',
    )

# 4. Fused conv-mix. The native kernel (csrc/inkling2/conv_mix.cu) is compiled
#    for Hopper/SM10x only. Upstream already has a Triton fallback
#    (short_conv + attn_res) that it uses on SM12x; use it everywhere the
#    native kernel does not exist. (The TP-fused variants are never reached on
#    SM86: Inkling2AllReduceWorkspace.create() returns None off SM9x/SM10x and
#    the model falls back to a plain NCCL all-reduce.)
edit(
    vllm_root / "models/inkling2/nvidia/ops/conv_mix.py",
    "        if current_platform.is_device_capability_family(120):\n",
    "        if not (\n"
    "            current_platform.is_device_capability_family(90)\n"
    "            or current_platform.is_device_capability_family(100)\n"
    "        ):\n",
)
