# server-bench: speed as an agent sees it (llama-server / llama-swap)

`bench.py` drives `llama-bench`, which cannot run **speculative decoding**, serve **several requests at once**, or use the server's real
scheduler. These two small scripts measure through the OpenAI-compatible endpoint instead (llama-swap on `localhost:8080`, or any llama-server),
and read the speed from the **server's own counters** (`timings.predicted_per_second`, `draft_n`, `draft_n_accepted`) rather than from a stopwatch.
Standard library only.

| Script | Measures | Example |
|---|---|---|
| `bench_spec.py MODEL [OUT.json]` | **one agent**: four kinds of task (translation PL→EN, tool call, code, free prose) × temperature 0 and 1.0, 300 tokens each; tok/s and draft acceptance per task | `BENCH_REPS=2 python3 bench_spec.py deepseek out.json` |
| `bench_conc.py MODEL OUT.json` | **1, 2 and 4 agents at once** (temperature 0.2, 300 tokens each, two rounds): tok/s per stream and in total | `python3 bench_conc.py deepseek out.json` |

Both print a summary table at the end and (if given) write every request to JSON.

## Read the numbers with these caveats

- **Thinking tokens are counted.** A reasoning model writes its reasoning first; for 300 tokens that is most of the output. This is the realistic picture of what an agent waits for,
  but it is not the same as the length of the final answer.
- **One repetition per cell by default** (`BENCH_REPS=1`): differences below ~2 tok/s are noise. Use `BENCH_REPS=2` or more before drawing conclusions.
- **`bench_conc.py` totals include prompt processing time**, so the single-agent total is lower than the per-stream speed.
- **No telemetry.** Unlike `bench.py`, these scripts do not record power draw, temperatures or PCIe state. Record the server command line yourself
  (llama-swap: `curl localhost:8080/running`; the quality bench's manifest does this automatically) and the power cap.
- They do **not** measure time to first token or prefill speed separately (the first is visible in a chat client; for prefill send one long prompt and read `timings.prompt_per_second`).

Used for the speculative-decoding and tensor-split measurements in
[local-llm-bench-results/SPEC-DECODING.md](https://github.com/michal-koniu/local-llm-bench-results/blob/main/SPEC-DECODING.md); how they fit into a tuning procedure:
[llm-education: speeding up generation](https://github.com/michal-koniu/llm-education/tree/main/en/speeding-up-generation).
