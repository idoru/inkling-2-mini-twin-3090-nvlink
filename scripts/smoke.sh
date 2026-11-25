#!/usr/bin/env bash
# Quick end-to-end check of a running server: model list, the KV-cache line
# from startup, one chat completion and one tool call.
# Usage: scripts/smoke.sh [base-url]   (default http://localhost:8000)
set -euo pipefail

BASE="${1:-http://localhost:8000}"
MODEL="${MODEL:-inkling-2-mini}"

echo "== models"
curl -fsS "$BASE/v1/models" | python3 -c '
import json, sys
for m in json.load(sys.stdin)["data"]:
    print("  ", m["id"], "max_model_len =", m.get("max_model_len"))'

echo "== chat"
curl -fsS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "reasoning_effort": "low",
  "max_tokens": 512,
  "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]
}' | python3 -c '
import json, sys
r = json.load(sys.stdin)
a = (r["choices"][0]["message"]["content"] or "").strip()
print("   answer:", a, "| tokens:", r["usage"]["completion_tokens"])
sys.exit(0 if "391" in a else 1)'

echo "== tool call"
curl -fsS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "reasoning_effort": "low",
  "max_tokens": 512,
  "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}]
}' | python3 -c '
import json, sys
calls = json.load(sys.stdin)["choices"][0]["message"].get("tool_calls") or []
for c in calls:
    print("   call:", c["function"]["name"], c["function"]["arguments"])
sys.exit(0 if calls and calls[0]["function"]["name"] == "get_weather" else 1)'

echo "smoke: ok"
