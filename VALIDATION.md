# Test results

## Spark run — 2026-09-08

DGX Spark GB10; the pinned dealignai checkpoint and SGLang image from
[model.lock.json](model.lock.json) and [common.sh](common.sh).
`MODE=1`, `CTX=200000`, `MEM_FRACTION=0.90`, medium thinking, FP8 KV,
BF16 recurrent state and NEXTN (3 steps, 4 draft tokens).

| Test | Result |
| --- | --- |
| Authentication | 401 without the key |
| Arithmetic | Correct |
| JSON output | Correct |
| Tool-call parser | Correct function and arguments |
| Reasoning parser | Separate reasoning and final answer |
| Long-input retrieval | Correct code from 22,931 input tokens; 11 output tokens; 10.686 s E2E |

**6/6 passed.** The server allocated 32,896 KV tokens for this run.
The earlier run passed five checks but rejected the long input with HTTP 400;
its reported KV capacity was 5,632 tokens. Capacity varies between launches.

Files:

- [Successful smoke run](validation/2026-09-08/smoke-long-rerun-2026-09-08.json)
- [Earlier smoke run, including the failure](validation/2026-09-08/smoke-before.json)
- [Server memory log excerpt](validation/2026-09-08/server-memory-excerpt.txt)
- [Source and file hashes](validation/2026-09-08/provenance.json)

Both smoke files include image and launch receipts; their 18 runtime-source
fingerprints match the code. JSON results are unchanged. The log excerpt
retains original line numbers and omits full server arguments.

### Memory

The log confirms HashK R=4 and FP8 conversion of the output head and 255 modules.
The main model, including HashK, accounts for 94.52 in the log's GB units;
the draft model adds 1.17 and Mamba state about 2.44. Main and draft K/V buffers
add about 0.42, with QSA index storage additional.

Final `available_gpu_mem=10.58 GB` is free memory, **not KV allocation**.
Use `max_total_num_tokens` to check capacity and record initial
`Load weight begin. avail mem=` when comparing launches.

### Coverage

This is one successful retrieval probe. Full 200k/256k context, MODE=2,
concurrent traffic and sustained reliability have not been validated.
No standalone throughput benchmark is included.

## Local tests

35 tests passed on macOS/arm64 with Python 3.9.6, PyTorch 2.7.1, SymPy 1.14.0,
safetensors 0.6.2, NumPy 2.0.2 and packaging 24.2.

Run `./check.sh`. It covers checkpoint/artifact integrity, patch generation,
launcher behavior, API validation and timing calculations. Six numerical tests
are skipped when their optional dependencies are absent. Docker/GPU behavior
is simulated in the local tests.

For a hardware repeat, use the README quickstart: long smoke, timing, then
long smoke again. Retain the JSON, runtime receipts and server logs.
