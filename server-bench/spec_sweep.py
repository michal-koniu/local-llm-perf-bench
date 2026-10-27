#!/usr/bin/env python3
"""spec_sweep.py MODEL '{"label": "extra llama-server flags", ...}'

Starts one llama-server per variant from MODEL's command in ~/.config/llama-swap/config.yaml (spec flags removed,
the variant's flags appended), runs a chat and a ~14k-token code prompt twice each with the server's default
(= vendor) sampling, prints tok/s and draft acceptance, stops the server.
In a variant: MD = the entry's -md draft model, BIN=/path/llama-server switches the binary,
ENV:NAME=value sets an environment variable (e.g. ENV:GGML_VK_VISIBLE_DEVICES=0).
Unload llama-swap models first (port 18999, the GPUs must be free)."""
import glob, json, os, re, shlex, subprocess, sys, time, urllib.request
import yaml

CFG = yaml.safe_load(open(os.path.expanduser("~/.config/llama-swap/config.yaml")))
PORT = 18999
which = sys.argv[1]            # model name in config
variants = json.loads(sys.argv[2])   # {"label": "extra flags"}

m = CFG["models"][which]
cmd = m["cmd"]
for k, v in (CFG.get("macros") or {}).items():
    cmd = cmd.replace("${%s}" % k, v)
cmd = cmd.replace("${PORT}", str(PORT))
base = shlex.split(cmd)
# strip spec flags; variants re-add them
out, skip = [], 0
for i, a in enumerate(base):
    if skip:
        skip -= 1; continue
    if a in ("--spec-type", "--spec-draft-n-max", "--spec-draft-p-min", "-md", "--model-draft"):
        skip = 1; continue
    out.append(a)
base = out
md = re.search(r"-md\s+(\S+)", cmd)
env = dict(os.environ)
for e in m.get("env", []) or []:
    k, v = e.split("=", 1); env[k] = v

src = sorted(glob.glob(os.path.expanduser("~/llama.cpp-e86f5fd/tools/server/*.cpp")), key=os.path.getsize)[-1]
code = open(src).read()[:52000]
PROMPTS = {
    "chat": [{"role": "user", "content": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, CUBIC vs BBR). Be thorough, use sections."}],
    "code16k": [{"role": "system", "content": "You are a senior C++ reviewer."},
                {"role": "user", "content": "Here is a source file:\n\n```cpp\n" + code + "\n```\n\nPick the longest function in this file and rewrite it fully with better structure and comments. Output only code."}],
}

def req(msgs):
    body = json.dumps({"messages": msgs, "max_tokens": 700}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                                      {"Content-Type": "application/json"}), timeout=900)
    t = json.loads(r.read())["timings"]
    return t["predicted_per_second"], t.get("draft_n", 0), t.get("draft_n_accepted", 0), t["prompt_n"]

for label, extra in variants.items():
    flags = extra.replace("MD", md.group(1) if md else "")
    toks, venv, full = shlex.split(flags), dict(env), list(base)
    for t in [t for t in toks if t.startswith(("BIN=", "ENV:"))]:
        if t.startswith("BIN="):
            full[0] = os.path.expanduser(t[4:])
        else:
            k, v = t[4:].split("=", 1); venv[k] = v
    full += [t for t in toks if not t.startswith(("BIN=", "ENV:"))]
    p = subprocess.Popen(full, env=venv, stdout=open(f"/tmp/sweep-{label}.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 900 and p.poll() is None:
        try:
            ok = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200
        except Exception:
            ok = False
        if ok:
            break
        time.sleep(3)
    if not ok:
        print(f"{label}: failed to start", flush=True); p.kill(); p.wait(); continue
    res = []
    for name, msgs in PROMPTS.items():
        for _ in range(2):
            res.append((name,) + req(msgs))
    p.terminate(); p.wait()
    line = " | ".join(f"{n} {tps:6.1f} t/s acc {(a / d if d else 0):.2f} ctx {pn}" for n, tps, d, a, pn in res)
    print(f"{label:22s} {line}", flush=True)
    time.sleep(5)
