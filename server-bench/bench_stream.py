#!/usr/bin/env python3
"""bench_stream.py MODEL TEMP TOP_P TOP_K [OUT.json]  -- engine-neutral decode benchmark.

Works with any OpenAI-compatible server (llama-server, vLLM, radiance, llama-swap in front of them),
because it measures on the client side from the stream instead of reading engine-specific timings:
decode tok/s = (completion_tokens - 1) / (time of last token - time of first token).

Two prompts, each run twice (the first run after a model start is usually slower - warm-up):
  chat     short question, ~700 tokens of prose (thinking models: mostly reasoning)
  code16k  ~14k-token source file + "rewrite the longest function" (agent-like long context)
Concurrency levels: BENCH_N (default "1,4"); every stream gets the same prompt.
Use the model vendor's sampling (TEMP/TOP_P/TOP_K): speculative decoding accepts far more draft
tokens at temperature 0, so greedy numbers overstate what real chats get.
Endpoint: BENCH_URL (default http://localhost:8080). Code context: BENCH_CODE_FILE (default:
Python's own argparse.py)."""
import argparse, json, os, sys, threading, time, urllib.request
model, temp, top_p, top_k = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
out_path = sys.argv[5] if len(sys.argv) > 5 else None
url = os.environ.get("BENCH_URL", "http://localhost:8080")
levels = [int(x) for x in os.environ.get("BENCH_N", "1,4").split(",")]
code = open(os.environ.get("BENCH_CODE_FILE", argparse.__file__)).read()[:52000]
PROMPTS = {
    "chat": [{"role": "user", "content": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, CUBIC vs BBR). Be thorough, use sections."}],
    "code16k": [{"role": "system", "content": "You are a senior reviewer."},
                {"role": "user", "content": "Here is a source file:\n\n```\n" + code + "\n```\n\nPick the longest function in this file and rewrite it fully with better structure and comments. Output only code."}],
}

def stream(msgs, out):
    body = dict(model=model, messages=msgs, max_tokens=700, temperature=temp, top_p=top_p, top_k=top_k,
                stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, last, usage = time.time(), None, None, None
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        j = json.loads(line[5:])
        usage = j.get("usage") or usage
        d = (j.get("choices") or [{}])[0].get("delta", {})
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
            last = time.time(); first = first or last
    n = usage["completion_tokens"]
    out.append(dict(tps=(n - 1) / (last - first), ttft=first - t0, gen=n, prompt=usage["prompt_tokens"]))

results = []
for conc in levels:
    for name, msgs in PROMPTS.items():
        for rep in range(2):
            out = []
            th = [threading.Thread(target=stream, args=(msgs, out)) for _ in range(conc)]
            [t.start() for t in th]; [t.join() for t in th]
            per = sum(o["tps"] for o in out) / len(out)
            r = dict(model=model, prompt=name, conc=conc, rep=rep, per_stream=round(per, 1), total=round(per * conc, 1),
                     ttft=round(out[0]["ttft"], 2), gen=out[0]["gen"], prompt_tokens=out[0]["prompt"])
            results.append(r)
            print(f"{model:32s} {name:8s} conc {conc:2d} rep {rep}: {per:6.1f} tok/s per stream, {per*conc:6.1f} total, "
                  f"ttft {r['ttft']:5.2f}s, prompt {r['prompt_tokens']}", flush=True)
if out_path:
    json.dump(dict(model=model, sampling=dict(temperature=temp, top_p=top_p, top_k=top_k), url=url,
                   date=time.strftime("%Y-%m-%d %H:%M"), results=results), open(out_path, "w"), indent=1)
