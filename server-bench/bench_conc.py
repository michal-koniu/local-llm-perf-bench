#!/usr/bin/env python3
"""bench_conc.py MODEL OUT.json  -- N parallel agent-like streams (N=1,2,4), temp 0.2, 300 tokens each, 2 rounds.
Prints per-stream tok/s (server timings) and aggregate tok/s (sum of tokens / wall)."""
import json, sys, time, threading, urllib.request, statistics
sys.argv += [None] * 3
MODEL, OUT = sys.argv[1], sys.argv[2]
URL = "http://localhost:8080/v1/chat/completions"
PROMPTS = [
 "Translate to English, output only the translation: Modele typu Mixture-of-Experts aktywują w każdym kroku tylko część ekspertów, więc o szybkości generowania decyduje głównie przepustowość pamięci karty graficznej, a nie jej moc obliczeniowa. Kwantyzacja zmniejsza liczbę bajtów do przeczytania w każdym kroku.",
 "Write a Python function that parses an nginx access log line with a regex and returns a dict. Include type hints and a short docstring. Code only.",
 "You are a coding agent. Explain step by step how you would find and fix a failing unit test in a Python project, then list the shell commands you would run.",
 "Summarize the trade-offs between layer-split and tensor-split multi-GPU inference for a MoE model, as a bullet list.",
]
def one(i, res):
    body = dict(model=MODEL, stream=False, temperature=0.2, max_tokens=300, seed=7 + i,
                messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900)); tm = r["timings"]
    res.append((tm["predicted_n"], tm["predicted_per_second"], tm.get("draft_n"), tm.get("draft_n_accepted")))
urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(dict(model=MODEL, max_tokens=4, messages=[{"role":"user","content":"hi"}])).encode(), headers={"Content-Type": "application/json"}), timeout=900).read()
print(f"[{MODEL}] ready", flush=True); out = []
for N in (1, 2, 4):
    for rnd in range(2):
        res = []; ths = [threading.Thread(target=one, args=(i, res)) for i in range(N)]
        t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
        tok = sum(r[0] for r in res); per = statistics.mean(r[1] for r in res)
        d = sum(r[2] or 0 for r in res); a = sum(r[3] or 0 for r in res)
        row = dict(N=N, rnd=rnd, per_stream_tps=round(per, 1), aggregate_tps=round(tok / wall, 1), accept=round(a / d, 2) if d else None, wall=round(wall, 1))
        out.append(row); print(row, flush=True)
print("\n=== SUMMARY", MODEL)
for N in (1, 2, 4):
    s = [r for r in out if r["N"] == N]
    print(f"N={N}: per-stream {statistics.mean(r['per_stream_tps'] for r in s):5.1f} tok/s | aggregate {statistics.mean(r['aggregate_tps'] for r in s):5.1f} tok/s")
if OUT: json.dump(out, open(OUT, "w"), indent=1)
