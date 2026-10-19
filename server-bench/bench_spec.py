#!/usr/bin/env python3
"""bench_spec.py MODEL_NAME [OUT.json]  -- single-stream decode benchmark through llama-swap (:8080).
Reads llama-server's own `timings` (predicted_per_second, draft_n, draft_n_accepted) from every response.
Workloads: translation PL->EN, agentic tool call, code, free prose; temperature 0 and 1.0; 2 reps each."""
import json, sys, time, urllib.request, statistics
MODEL = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
URL = "http://localhost:8080/v1/chat/completions"
PL = ("Kwantyzacja polega na zapisaniu wag modelu na mniejszej liczbie bitów. Dzięki temu model zajmuje mniej pamięci "
      "karty graficznej, a generowanie tokenów jest szybsze, bo w każdym kroku trzeba przeczytać z pamięci mniej bajtów. "
      "Kosztem jest niewielki spadek jakości, który zależy od metody kwantyzacji i od tego, które warstwy zostały "
      "zachowane w wyższej precyzji. W modelach typu Mixture-of-Experts w każdym kroku aktywna jest tylko część ekspertów, "
      "więc o szybkości decyduje głównie przepustowość pamięci, a nie moc obliczeniowa.")
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from disk",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
          "end_line": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "grep_search", "description": "Search files with a regex",
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}}]
WORK = {
 "translate": dict(messages=[{"role": "user", "content": "Translate to English. Output only the translation.\n\n" + PL}], max_tokens=300),
 "toolcall":  dict(messages=[{"role": "user", "content": "Find where the function `load_config` is defined in /srv/app, then read the first 40 lines of that file."}],
                   tools=TOOLS, max_tokens=300),
 "code":      dict(messages=[{"role": "user", "content": "Write a Python function that parses an nginx access log line with a regex and returns a dict. Include type hints and a short docstring. Code only."}], max_tokens=300),
 "prose":     dict(messages=[{"role": "user", "content": "Write a short, original story about a lighthouse keeper who finds a message in a bottle."}], max_tokens=300),
}
def call(body, timeout=900):
    body = dict(body, model=MODEL, stream=False)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=timeout)); return r, time.time() - t0
t0 = time.time(); r, _ = call(dict(messages=[{"role": "user", "content": "Say hi."}], max_tokens=8, temperature=0))
print(f"[{MODEL}] model ready after {time.time()-t0:.0f}s (incl. load)", flush=True)
rows = []
for temp in (0.0, 1.0):
    for name, w in WORK.items():
        for rep in range(int(__import__("os").environ.get("BENCH_REPS", "2"))):
            r, dt = call(dict(w, temperature=temp, seed=1000 + rep))
            tm = r.get("timings", {}); u = r.get("usage", {})
            row = dict(work=name, temp=temp, rep=rep, tokens=tm.get("predicted_n", u.get("completion_tokens")),
                       tps=tm.get("predicted_per_second"), draft_n=tm.get("draft_n"), draft_acc=tm.get("draft_n_accepted"),
                       wall=round(dt, 2))
            rows.append(row); print(row, flush=True)
print("\n=== SUMMARY", MODEL)
print(f"{'work':10} {'temp':>4} {'tok/s':>7} {'accept':>7}")
for temp in (0.0, 1.0):
    for name in WORK:
        sel = [x for x in rows if x["work"] == name and x["temp"] == temp and x["tps"]]
        d = sum(x["draft_n"] or 0 for x in sel); a = sum(x["draft_acc"] or 0 for x in sel)
        print(f"{name:10} {temp:4.1f} {statistics.mean(x['tps'] for x in sel):7.1f} {(a/d if d else float('nan')):7.2f}")
allt = [x["tps"] for x in rows if x["tps"]]; print(f"{'ALL':10} {'':>4} {statistics.mean(allt):7.1f}")
if OUT: json.dump(rows, open(OUT, "w"), indent=1)
