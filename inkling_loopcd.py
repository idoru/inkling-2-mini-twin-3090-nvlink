"""LoopCD-Logits for Inkling 2 (arXiv:2610.02185), as a per-request option.

Inkling-2-Mini runs its middle layers three times. LoopCD contrasts the
logits of the full model, z_R, with the logits the model would produce if it
stopped looping after the first pass, z_1:

    z' = z_R + w * (z_R - z_1)

z_1 is not read straight off the first pass: the first pass's state goes
through the remaining (post-loop) layers, the final norm and the LM head,
with its own KV cache. inkling_loopcd_patch.py adds that second run to the
model when the server starts with INKLING_LOOPCD=1; this module is the
sampler-side half, which applies the contrast to the rows of requests that
asked for it:

    "vllm_xargs": {"loopcd": 0.5}                         # adaptive (default)
    "vllm_xargs": {"loopcd": 0.3, "loopcd_adaptive": 0}   # fixed strength

Adaptive strength (the paper's Eq. 4) scales w per token by how undecided the
unguided model is: w = loopcd * (1 - (p1 - p2)), with p1 and p2 the two
largest probabilities of softmax(z_R). Requests without the parameter are
sampled from z_R unchanged.

Import-light on purpose: vLLM also loads this class in the API server
process to validate request parameters.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

if TYPE_CHECKING:
    import torch

ENABLED = os.environ.get("INKLING_LOOPCD", "0") == "1"
MAX_STRENGTH = 4.0
# Negative strengths move toward the first pass instead of away from it; with
# fixed strength, -1 decodes from the first-pass logits alone. Diagnostic only.
MIN_STRENGTH = -1.0
# Token ids from here up are Inkling's control tokens (message, block and
# tool-call markers). They keep the unguided logits: contrasting them breaks
# the chat structure at higher strengths.
CONTROL_START = int(os.environ.get("INKLING_LOOPCD_CONTROL_START", "199998"))

# Logits of the first loop pass for the rows of the current step, set by the
# patched Inkling2ForConditionalGeneration.compute_logits just before the
# sampler runs. Row-aligned with the logits the sampler receives.
weak_logits: "torch.Tensor | None" = None


def request_params(sampling_params) -> tuple[float, bool]:
    """(strength, adaptive) of a request; strength 0 means off."""
    args = sampling_params.extra_args or {}
    raw = args.get("loopcd", 0)
    try:
        strength = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"loopcd must be a number, got {raw!r}") from None
    if not MIN_STRENGTH <= strength <= MAX_STRENGTH:
        raise ValueError(
            f"loopcd must be between {MIN_STRENGTH} and {MAX_STRENGTH}, got {raw!r}"
        )
    raw_adaptive = args.get("loopcd_adaptive", 1)
    if str(raw_adaptive).lower() in ("1", "1.0", "true"):
        adaptive = True
    elif str(raw_adaptive).lower() in ("0", "0.0", "false"):
        adaptive = False
    else:
        raise ValueError(f"loopcd_adaptive must be 0 or 1, got {raw_adaptive!r}")
    return strength, adaptive


class LoopCDLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states) -> None:
        import numpy as np
        import torch

        if not ENABLED:
            raise RuntimeError(
                "LoopCDLogitsProcessor needs INKLING_LOOPCD=1 (and an image "
                "built with INKLING_LOOPCD_PATCH=1)"
            )
        self.device = req_states.device
        # Host copies decide whether a step needs any work at all.
        self.strength_np = np.zeros(req_states.max_num_reqs, dtype=np.float32)
        self.strength = torch.zeros(req_states.max_num_reqs, device=self.device)
        self.adaptive = torch.zeros(
            req_states.max_num_reqs, dtype=torch.bool, device=self.device
        )
        self.dirty = False
        self.adaptive_np = np.zeros(req_states.max_num_reqs, dtype=bool)

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        request_params(sampling_params)

    def add_request(self, req_idx: int, sampling_params) -> bool:
        strength, adaptive = request_params(sampling_params)
        self.strength_np[req_idx] = strength
        self.adaptive_np[req_idx] = adaptive
        self.dirty = True
        return strength != 0.0

    def apply_staged_writes(self) -> None:
        import torch

        if self.dirty:
            self.strength.copy_(torch.from_numpy(self.strength_np))
            self.adaptive.copy_(torch.from_numpy(self.adaptive_np))
            self.dirty = False

    def apply(self, logits, ctx):
        import torch

        if not self.strength_np[ctx.idx_mapping_np].any():
            return logits
        weak = weak_logits
        if weak is None or weak.shape != logits.shape:
            raise RuntimeError(
                "LoopCD was requested but the model produced no first-pass "
                "logits for this step"
            )
        strength = self.strength[ctx.expanded_idx_mapping]
        rows = strength.nonzero().squeeze(-1)
        strong = logits[rows]
        scale = strength[rows]
        adaptive = self.adaptive[ctx.expanded_idx_mapping][rows]
        top2 = torch.softmax(strong, dim=-1).topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        scale = torch.where(adaptive, scale * (1.0 - margin), scale)
        # Tokens an earlier stage masked to -inf stay masked (and a zero
        # scale must not turn them into NaN).
        delta = torch.where(
            torch.isfinite(strong), strong - weak[rows].to(strong.dtype), 0.0
        )
        delta[:, CONTROL_START:] = 0.0
        logits[rows] = strong + scale.unsqueeze(-1) * delta
        return logits
