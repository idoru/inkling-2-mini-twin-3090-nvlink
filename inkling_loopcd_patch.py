#!/usr/bin/env python3
"""Build-time patch: LoopCD-Logits for vLLM's looped Inkling2 model.

Adds the model-side half of LoopCD (arXiv:2610.02185; see inkling_loopcd.py
for the method and the request parameter). Everything here is inert unless
the server process runs with INKLING_LOOPCD=1:

1. The post-loop ("coda") layers get a second set of KV and conv-state
   caches, the same mechanism the loop core layers use for their passes.
2. After the first loop pass, the coda runs once more on that pass's state
   using the second cache set. Its final-norm output is the "weak" hidden
   state h_1'.
3. The model returns [h_R | h_1'] side by side, and compute_logits splits
   them, runs the LM head on both and hands the weak logits to
   inkling_loopcd.LoopCDLogitsProcessor. Returning them through the forward
   output (rather than stashing a tensor on the module) keeps this correct
   under CUDA graph replay.

Cost when enabled: 6 extra layer visits per token on top of 48 (about 12%
more compute) and one more full-attention KV cache on top of 8, for every
request, whether or not it uses the parameter. With INKLING_LOOPCD unset the
patched code takes the original path.

Each edit asserts on the exact upstream text, like inkling_sm86_patch.py.
"""
import importlib.util
import pathlib
import shutil
import sysconfig


def path_of(module: str) -> pathlib.Path:
    return pathlib.Path(importlib.util.find_spec(module).origin)


def edit(path: pathlib.Path, old: str, new: str, count: int = 1) -> None:
    src = path.read_text()
    assert src.count(old) == count, f"{path}: expected {count} match(es) for {old!r}"
    path.write_text(src.replace(old, new))
    print(f"patched {path}")


inkling = path_of("vllm").parent / "models/inkling2/nvidia"
loop = inkling / "loop.py"
model = inkling / "model.py"

# The sampler-side module, importable by both the workers and the API server.
here = pathlib.Path(__file__).resolve().parent
shutil.copy(here / "inkling_loopcd.py", sysconfig.get_paths()["purelib"])

edit(
    loop,
    "from collections.abc import Sequence\n",
    "import os\nfrom collections.abc import Sequence\n",
)
edit(
    loop,
    'LOOP_STREAMS = ("attn", "k", "v", "mlp")\n',
    'LOOP_STREAMS = ("attn", "k", "v", "mlp")\n'
    'LOOPCD = os.environ.get("INKLING_LOOPCD", "0") == "1"\n',
)

# 1. A second cache set for the coda layers.
edit(
    loop,
    "            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)\n"
    "        return super()._make_layer(config, layer_id, quant_config, prefix)\n",
    "            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)\n"
    "        if LOOPCD and layer_id >= config.loop_core_layers.stop:\n"
    "            return Inkling2DecoderLayer(\n"
    "                config, layer_id, quant_config, prefix, num_passes=2\n"
    "            )\n"
    "        return super()._make_layer(config, layer_id, quant_config, prefix)\n",
)
# The coda's MoE has one routed-experts capture row; the weak run shares it.
edit(
    loop,
    "layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx)",
    "layer.mlp(\n"
    "                    mlp_in[0],\n"
    "                    reduce=False,\n"
    "                    pass_idx=min(pass_idx, len(layer.mlp.routing_captures) - 1),\n"
    "                )",
)
edit(
    loop,
    "                layer.mlp(mlp_in[0], pass_idx=pass_idx),\n",
    "                layer.mlp(\n"
    "                    mlp_in[0],\n"
    "                    pass_idx=min(pass_idx, len(layer.mlp.routing_captures) - 1),\n"
    "                ),\n",
)

# 2. The weak coda run, right after the first pass's last core layer.
edit(
    loop,
    "    def forward(\n"
    "        self, positions: torch.Tensor, inputs_embeds: torch.Tensor\n"
    "    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:\n"
    '        """Prelude, `num_passes` core passes, then the coda.',
    "    def _loopcd_weak(\n"
    "        self,\n"
    "        stack: torch.Tensor,\n"
    "        num_blocks: int,\n"
    "        positions: torch.Tensor,\n"
    "        tau: torch.Tensor | None,\n"
    "    ) -> torch.Tensor:\n"
    '        """LoopCD: the coda and final norm on the state after the first\n'
    "        loop pass (`stack[:num_blocks]` completed blocks and the partial\n"
    "        `stack[num_blocks]`), on a copy of the stack and with the coda\n"
    '        layers\' second cache set. Returns the final-norm output."""\n'
    "        start, end = self.loop_core.stop, len(self.layers)\n"
    "        stack = stack.clone()\n"
    "        inputs = self._mix(\n"
    "            stack[:num_blocks],\n"
    "            stack[num_blocks],\n"
    "            self.attn_res.attn_queries[start - 1],\n"
    "            self.layers[start].attn_norm,\n"
    "        )\n"
    "        plan = [(False, False)] * 2\n"
    "        for i in range(start, end):\n"
    "            if i + 1 == end:\n"
    "                next_query, next_norm = self.attn_res.output_query, self.norm\n"
    "            else:\n"
    "                next_query = self.attn_res.attn_queries[i]\n"
    "                next_norm = self.layers[i + 1].attn_norm\n"
    "            num_blocks, inputs = self._run_layer(\n"
    "                i,\n"
    "                1,\n"
    "                stack,\n"
    "                num_blocks,\n"
    "                inputs,\n"
    "                None,\n"
    "                next_query,\n"
    "                next_norm,\n"
    "                positions,\n"
    "                tau,\n"
    "                plan,\n"
    "            )\n"
    "        return inputs[0]\n"
    "\n"
    "    def forward(\n"
    "        self, positions: torch.Tensor, inputs_embeds: torch.Tensor\n"
    "    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:\n"
    '        """Prelude, `num_passes` core passes, then the coda.',
)
edit(
    loop,
    "        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None\n",
    "        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None\n"
    "        loopcd_weak = None\n",
)
edit(
    loop,
    "                tau,\n"
    "                plan,\n"
    "            )\n"
    "        assert inputs is not None\n"
    "        out = inputs[0]\n"
    "        if self.aux_hidden_state_layers:\n"
    "            return out, [*aux_hidden_states, out]\n"
    "        return out\n",
    "                tau,\n"
    "                plan,\n"
    "            )\n"
    "            if LOOPCD and pass_idx == 0 and i + 1 == core.stop:\n"
    "                loopcd_weak = self._loopcd_weak(stack, num_blocks, positions, tau)\n"
    "        assert inputs is not None\n"
    "        out = inputs[0]\n"
    "        # LoopCD: [h_R | h_1'] side by side; compute_logits splits them.\n"
    "        hidden_states = (\n"
    "            out if loopcd_weak is None else torch.cat((out, loopcd_weak), dim=-1)\n"
    "        )\n"
    "        if self.aux_hidden_state_layers:\n"
    "            return hidden_states, [*aux_hidden_states, out]\n"
    "        return hidden_states\n",
)

# 3. Split in compute_logits; one LM-head call covers both halves.
edit(
    model,
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n",
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        width = self.text_config.hidden_size\n"
    "        if hidden_states.shape[-1] == 2 * width:\n"
    "            # LoopCD: rows of [h_R | h_1'] (see loop.py).\n"
    "            import inkling_loopcd\n"
    "\n"
    "            num_rows = hidden_states.shape[0]\n"
    "            both = torch.cat((hidden_states[:, :width], hidden_states[:, width:]))\n"
    "            logits = self.logits_processor(self.lm_head, both)\n"
    "            if logits is None:\n"
    "                return None\n"
    "            inkling_loopcd.weak_logits = logits[num_rows:]\n"
    "            return logits[:num_rows]\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n",
)
