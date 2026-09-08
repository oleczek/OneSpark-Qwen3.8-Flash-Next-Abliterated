# OneSpark — Qwen3.8-Flash-Next Abliterated on one DGX Spark

Run [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
on a single DGX Spark (GB10, 128 GB) through SGLang's OpenAI-compatible API.
Text only. Based on the work of
[azampatti](https://github.com/azampatti/GB10-3.8-Flash-Next) and
[deathbyorderfill](https://github.com/deathbyorderfill/flashnext-one-spark).

**Tested on Spark:** 6/6 smoke checks passed, including retrieval from a
22,931-token input. [Results and logs](VALIDATION.md).

Experimental, as-is. No support or commitment to updates.

## Quickstart

Requirements: Linux/aarch64, NVIDIA driver, Docker with NVIDIA Container
Toolkit, Git, Python 3.9+, curl and the Hugging Face CLI. Allow roughly
180–200 GiB of free disk for a fresh setup. The HashK builder requires 40 GiB
free on the checkout volume.

Stop other memory-heavy workloads first. Setup and launch require at least
105 GiB available RAM; less than 16 GiB free swap produces a warning.

```bash
git clone https://github.com/oleczek/OneSpark-Qwen3.8-Flash-Next-Abliterated.git
cd OneSpark-Qwen3.8-Flash-Next-Abliterated

./check.sh

hf download dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --revision be794b990578ef3031eccf9f28e675a289a09ee9

export API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MODE=1

./install.sh
./launch.sh

mkdir -p results
./smoke.sh --long > results/smoke-before.json
./sgbench.sh 1 --runs 1 > results/timing-c1.json
./smoke.sh --long > results/smoke-after.json
./stop.sh
```

`install.sh` verifies all 206 weight files against [model.lock.json](model.lock.json),
pulls the pinned runtime image and builds HashK. Downloads, builds and startup
can take tens of minutes. Follow startup with `docker logs -f onespark-abliterated`.
Keep the same `API_KEY` for clients and repeat runs.

## How it fits

The checkpoint contains about 135.2 GB of tensors, including a 51.2 GB FP8
n-gram embedding table (PLE). The default mode compresses PLE to about
12.8 GB with HashK R=4. The stack uses NVFP4 experts, FP8 dense layers and KV
cache, BF16 recurrent state, and NEXTN speculative decoding (3 steps, 4 draft tokens).

| Mode | PLE | Configured context |
| --- | --- | --- |
| `MODE=1` (default) | HashK R=4, lossy compression | 200,000 |
| `MODE=2` | PLE disabled; untested on Spark | 256,000 |

**Usable context depends on free memory at startup.** The successful test
allocated 32,896 KV tokens; the earlier launch reportedly allocated 5,632.
Check `max_total_num_tokens` in the server log and leave room for output and
other requests. Neither configured context above has been validated in full.
HashK, PLE-off and quantization can affect answer quality.

## API and configuration

API: `http://127.0.0.1:30000/v1`. Use the checkpoint ID as the model name and
`API_KEY` as the Bearer token. For remote access, use an SSH tunnel; the server
uses plain HTTP. The key is visible to host/Docker administrators in process
configuration, so keep full server dumps private.

| Variable | Default / behavior |
| --- | --- |
| `MODE` | `1`; use the same value for install and launch |
| `API_KEY` | Required, at least 16 characters |
| `PORT` | `30000` |
| `LISTEN_HOST` | `127.0.0.1` |
| `CONTAINER_NAME` | `onespark-abliterated` |
| `CTX` | Mode default above; maximum 262,144 |
| `MEM_FRACTION` | `0.90`; accepts lower values |
| `THINKING` | `medium`; also `off`, `low`, `xhigh` |
| `HF_HOME`, `HF_HUB_CACHE` | Hugging Face cache overrides |
| `MODEL_SNAPSHOT` | Complete local copy of the pinned revision |
| `BASE_URL` | Clients: `http://127.0.0.1:$PORT` |
| `BENCH_THINKING` | Timing client: `medium` |
| `IMAGE` | Runtime override; requires reinstall, untested |

The default linux/arm64 image is pinned in [common.sh](common.sh).
After changing runtime inputs, rerun `install.sh`. HashK artifacts require a
matching provenance sidecar; rebuild old artifacts in a fresh checkout.

`launch.sh` refuses to replace an existing container. `stop.sh` removes only
the container owned by this checkout and keeps the weights and HashK files.
Keep the checkout path unchanged while its container is running.

## Checks and timing

`check.sh` runs offline code tests. `smoke.sh --long` checks authentication,
arithmetic, JSON, tool-call parsing, separated reasoning and long-input retrieval.

`sgbench.sh [-v] [1–8] [--runs N]` saves responses and **end-to-end** timing,
including prefill, decode and HTTP. Completion counts include reasoning tokens;
parallel runs use one shared batch duration.

The upstream reports output corruption after some traffic while health checks
still pass. Repeat the long smoke test after workloads. This port has no
watchdog; sustained reliability and concurrent traffic have not been validated.

## Credits and license

Deployment port by [Aleksander Fafuła](https://fafula.com) · [alex@fafula.com](mailto:alex@fafula.com).

Model, abliteration, HashK and kernel credits, source revisions and licenses: [NOTICE.md](NOTICE.md).
Weights and generated HashK files are downloaded or built locally.
