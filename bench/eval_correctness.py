"""
Correctness eval for the Inkling-2-Mini vLLM server (prefix caching on, DSpark).

Tasks span four categories that stress the bits most likely to regress under
config tuning:
  - needle: long-context retrieval at 5 positions (sliding-window + global attention)
  - math:   multi-step arithmetic / algebra (stresses reasoning under speculation)
  - code:   simple Python function generation (structural correctness)
  - json:   JSON-only output (format compliance, used by tool-calling agents)

Run before/after a config change (e.g. SPEC_TOKENS 7 -> 0, prefix caching
on/off, --max-num-batched-tokens) to detect correctness regressions:

  python eval_correctness.py --output /tmp/eval_baseline.json
  # ... change config, restart vLLM ...
  python eval_correctness.py --compare /tmp/eval_baseline.json
"""
import asyncio, time, json, re, argparse, sys
from openai import AsyncOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000/v1")
parser.add_argument("--model", default="inkling-2-mini")
parser.add_argument("--output", default="/tmp/eval_results.json")
parser.add_argument("--compare", default=None,
                    help="diff pass/fail outcomes against a saved JSON file")
parser.add_argument("--needle-context-tokens", type=int, default=100000)
parser.add_argument("--max-tokens", type=int, default=2048)
parser.add_argument("--skip-long-context", action="store_true")
args = parser.parse_args()

client = AsyncOpenAI(base_url=args.base_url, api_key="x")


def filler(target_tokens):
    unit = "Filler text. The quick brown fox jumps over the lazy dog. " * 4 + "\n"
    chars = target_tokens * 4
    return (unit * (chars // len(unit) + 1))[:chars]


def haystack_with_needle(needle, position_frac, total_tokens):
    body = filler(total_tokens)
    pos = int(len(body) * position_frac)
    pos = body.rfind("\n", 0, pos) + 1
    return body[:pos] + needle + "\n" + body[pos:]


TASKS = []
def task(category, id_, build, validate):
    TASKS.append({"category": category, "id": id_, "build": build, "validate": validate})


NEEDLES = [
    ("REMEMBER: the magic phrase is 'cobalt-meridian-42'.",   "cobalt-meridian-42"),
    ("REMEMBER: the secret code is 'azure-helix-789'.",       "azure-helix-789"),
    ("REMEMBER: the access token is 'crimson-eta-15'.",       "crimson-eta-15"),
    ("REMEMBER: the cipher key is 'jade-vortex-3'.",          "jade-vortex-3"),
    ("REMEMBER: the activation phrase is 'amber-quantum-77'.","amber-quantum-77"),
]
POSITIONS = [0.05, 0.25, 0.50, 0.75, 0.95]
for frac, (needle, answer) in zip(POSITIONS, NEEDLES):
    def make(needle=needle, frac=frac):
        h = haystack_with_needle(needle, frac, args.needle_context_tokens)
        return [{"role": "user", "content":
            "Below is a long document. Read it and answer the question at the end.\n\n"
            + h + "\n\nQuestion: A REMEMBER instruction inside the document defines "
            "a phrase, code, token, key, or password. Reply with ONLY that exact value, no quotes."
        }]
    def make_v(answer=answer):
        def v(text, _obj):
            ok = answer.lower() in text.lower()
            return ok, "" if ok else f"answer '{answer}' not in response"
        return v
    task("needle", f"needle_pos_{int(frac*100):02d}", make, make_v())


MATH = [
    ("A bookstore has 320 books. They sell 35% on Monday and 40 more on Tuesday. "
     "How many books are left? Answer with just the integer.",
     re.compile(r"\b168\b")),
    ("Train A leaves a station at 8am traveling at 60 mph. Train B leaves the same station "
     "at 9am traveling at 80 mph in the same direction. At what time does B catch A? "
     "Answer with just the hour, e.g. '12pm'.",
     re.compile(r"\b12\s*pm\b", re.IGNORECASE)),
    ("If a recipe calls for 3/4 cup of sugar and you want to make 5x the recipe, "
     "how many cups of sugar are needed? Answer with just a number or fraction.",
     re.compile(r"(?:15/4|3\.75|3\s*3/4)")),
    ("A square has perimeter 36 cm. What is its area in cm^2? Answer with just the integer.",
     re.compile(r"\b81\b")),
    ("If x + 2y = 10 and 3x - y = 9, what is x + y? Answer with just the integer.",
     re.compile(r"\b7\b")),
]
for i, (prompt, regex) in enumerate(MATH):
    def make(p=prompt): return [{"role": "user", "content": p}]
    def make_v(rgx=regex):
        def v(text, _obj):
            ok = bool(rgx.search(text))
            return ok, "" if ok else f"no match for /{rgx.pattern}/"
        return v
    task("math", f"math_{i+1}", make, make_v())


CODE = [
    ("Write a Python function `is_palindrome(s: str) -> bool` that returns True iff s reads the "
     "same forwards and backwards (case-insensitive, ignoring non-alphanumeric). Reply with ONLY "
     "the function code, no markdown.",
     [r"def\s+is_palindrome", r"return"]),
    ("Write a Python function `fizzbuzz(n: int) -> list[str]` that returns the FizzBuzz output for "
     "1..n. Reply with ONLY the function code, no markdown.",
     [r"def\s+fizzbuzz", r"Fizz", r"Buzz"]),
    ("Write a Python function `reverse_words(s: str) -> str` that reverses the order of "
     "space-separated words. Reply with ONLY the function code, no markdown.",
     [r"def\s+reverse_words", r"split", r"join"]),
    ("Write a Python function `is_prime(n: int) -> bool`. Reply with ONLY the function code, no markdown.",
     [r"def\s+is_prime", r"%"]),
    ("Write a Python function `nth_fibonacci(n: int) -> int` returning the n-th Fibonacci number "
     "(0-indexed). Reply with ONLY the function code, no markdown.",
     [r"def\s+nth_fibonacci", r"return"]),
]
for i, (prompt, needles) in enumerate(CODE):
    def make(p=prompt): return [{"role": "user", "content": p}]
    def make_v(ns=needles):
        def v(text, _obj):
            for n in ns:
                if not re.search(n, text):
                    return False, f"missing /{n}/"
            return True, ""
        return v
    task("code", f"code_{i+1}", make, make_v())


JSON_TASKS = [
    ("Return ONLY a JSON object with keys 'capital' (string) and 'population_millions' (number) "
     "for France. No markdown.",
     ["capital", "population_millions"]),
    ("Return ONLY a JSON object with keys 'language', 'paradigm' (array of strings), "
     "'first_appeared' (integer year) for Python. No markdown.",
     ["language", "paradigm", "first_appeared"]),
    ("Return ONLY a JSON array of 3 objects, each with 'name' and 'symbol' for the first 3 "
     "elements of the periodic table. No markdown.",
     ["name", "symbol"]),
    ("Return ONLY a JSON object with keys 'sum' and 'product' computed from the integers 7 and 8. "
     "No markdown.",
     ["sum", "product"]),
    ("Return ONLY a JSON object with key 'planets' (array of 8 strings) for the planets in order "
     "from the Sun. No markdown.",
     ["planets"]),
]
for i, (prompt, keys) in enumerate(JSON_TASKS):
    def make(p=prompt): return [{"role": "user", "content": p}]
    def make_v(needles=keys):
        def v(text, _obj):
            s = text.strip()
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```\s*$", "", s)
            try:
                obj = json.loads(s)
            except Exception as e:
                return False, f"json parse fail: {e}"
            blob = json.dumps(obj)
            for n in needles:
                if f'"{n}"' not in blob:
                    return False, f"missing key '{n}'"
            return True, ""
        return v
    task("json", f"json_{i+1}", make, make_v())


async def run_one(t):
    msgs = t["build"]()
    t0 = time.perf_counter()
    try:
        r = await client.chat.completions.create(
            model=args.model, messages=msgs,
            max_tokens=args.max_tokens, temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:
        return {"id": t["id"], "category": t["category"], "pass": False,
                "reason": f"api error: {e}", "latency_s": time.perf_counter()-t0,
                "response": None, "prompt_tokens": 0, "completion_tokens": 0}
    dt = time.perf_counter() - t0
    text = r.choices[0].message.content or ""
    ok, reason = t["validate"](text, r)
    return {"id": t["id"], "category": t["category"], "pass": ok, "reason": reason,
            "latency_s": dt, "response": text[:1500],
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens}


async def main():
    tasks = [t for t in TASKS if not (args.skip_long_context and t["category"] == "needle")]
    print(f"Running {len(tasks)} tasks against {args.model} (concurrency=1, temp=0)")
    print()
    results = []
    t0 = time.perf_counter()
    for t in tasks:
        r = await run_one(t)
        results.append(r)
        mark = "PASS" if r["pass"] else "FAIL"
        suffix = "" if r["pass"] else f"  -- {r['reason']}"
        print(f"  [{mark}] {r['category']:>6}/{r['id']:<18} "
              f"{r['latency_s']:>6.2f}s  "
              f"({r['prompt_tokens']:>6,} prompt / {r['completion_tokens']:>4} gen){suffix}")
    wall = time.perf_counter() - t0

    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, [0, 0])
        by_cat[c][1] += 1
        if r["pass"]: by_cat[c][0] += 1
    passed = sum(p for p, _ in by_cat.values())
    total = sum(n for _, n in by_cat.values())

    print()
    print(f"--- summary ({wall:.1f}s wall) ---")
    for cat, (p, n) in sorted(by_cat.items()):
        bar = "█" * p + "·" * (n - p)
        print(f"  {cat:>6}: {p}/{n}  {bar}")
    print(f"  {'total':>6}: {passed}/{total}  ({100*passed/total:.1f}%)")

    out = {"model": args.model, "wall_s": wall, "passed": passed, "total": total,
           "by_category": by_cat, "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved: {args.output}")

    if args.compare:
        try:
            ref = json.load(open(args.compare))
        except Exception as e:
            print(f"!! could not load --compare {args.compare}: {e}", file=sys.stderr)
            return
        print()
        print(f"--- diff vs {args.compare} ---")
        ref_by_id = {r["id"]: r for r in ref["results"]}
        outcome_changes = 0
        text_changes = 0
        for r in results:
            ref_r = ref_by_id.get(r["id"])
            if not ref_r: continue
            if r["pass"] != ref_r["pass"]:
                outcome_changes += 1
                print(f"  OUTCOME: {r['id']:<22} {ref_r['pass']} -> {r['pass']}  "
                      f"(was: {ref_r.get('reason','')!r}, now: {r.get('reason','')!r})")
            elif r["response"] != ref_r["response"]:
                text_changes += 1
        print(f"  outcome changes: {outcome_changes}")
        print(f"  text-only drift (same pass/fail): {text_changes}")
        if outcome_changes == 0:
            print(f"  OK: pass/fail identical ({passed}/{total} both runs)")


asyncio.run(main())
