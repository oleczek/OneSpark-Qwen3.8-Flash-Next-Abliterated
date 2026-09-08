#!/usr/bin/env python3
"""Authenticated smoke checks and end-to-end timing, not a decode/quality benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import secrets
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from checkpoint import ROOT


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never forward the Bearer key to a redirect target


class Client:
    def __init__(self, base, key, timeout=180):
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout
        self.opener = build_opener(NoRedirect())

    def request(self, path, payload=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.key
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=data, headers=headers)
        start = time.perf_counter()
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        return result, time.perf_counter() - start

    def chat(self, model, prompt, max_tokens=512, **extra):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        payload.update(extra)
        response, elapsed = self.request("/v1/chat/completions", payload)
        if response.get("error"):
            raise ValueError("API returned an error payload")
        message = response["choices"][0]["message"]
        tokens = response["usage"]["completion_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("Missing/invalid completion token count")
        if not (message.get("content") or message.get("reasoning_content") or message.get("tool_calls")):
            raise ValueError("Empty assistant response")
        return {"prompt": prompt, "elapsed_seconds": elapsed, "completion_tokens": tokens,
                "e2e_output_tokens_per_second": tokens / elapsed, "response": response}


def validate_math(result):
    content = result["response"]["choices"][0]["message"].get("content") or ""
    if content.strip() != "4":
        raise ValueError("Expected exactly 4")


def validate_json(result):
    content = result["response"]["choices"][0]["message"].get("content") or ""
    if json.loads(content) != {"ok": True, "value": 7}:
        raise ValueError("JSON payload differs from requested object")


def validate_tool(result):
    calls = result["response"]["choices"][0]["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "get_weather":
        raise ValueError("Expected one parsed get_weather tool call")
    arguments = calls[0]["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if arguments != {"city": "Warsaw"}:
        raise ValueError("Incorrect tool arguments")


def validate_reasoning(result):
    message = result["response"]["choices"][0]["message"]
    if not message.get("reasoning_content") or "323" not in (message.get("content") or ""):
        raise ValueError("Expected separated reasoning_content and final answer 323")


def smoke(client, model, long_probe=False):
    checks = []
    try:
        client.request("/v1/models", authenticated=False)
        checks.append({"name": "auth", "ok": False, "error": "Unauthenticated model listing was accepted"})
    except HTTPError as exc:
        checks.append({"name": "auth", "ok": exc.code in (401, 403), "http_status": exc.code})
    tool = {"type": "function", "function": {"name": "get_weather",
            "description": "Look up current weather for a city.", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"]}}}
    cases = [
        ("math", "What is 2 + 2? Reply with the digit only.", {}, validate_math),
        ("json", 'Return only this JSON object, no Markdown: {"ok":true,"value":7}', {}, validate_json),
        ("tool_parser", "Use get_weather to look up the current weather in Warsaw. Do not invent the weather.",
         {"tools": [tool], "tool_choice": "auto"}, validate_tool),
        ("reasoning_parser", "Compute 17 * 19, verify your arithmetic, then give the result.",
         {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}, validate_reasoning),
    ]
    for name, prompt, extra, validate in cases:
        check = {"name": name, "ok": False}
        try:
            check["result"] = client.chat(model, prompt, max_tokens=1024, **extra)
            validate(check["result"])
            check["ok"] = True
        except Exception as exc:
            check["error"] = str(exc)
        checks.append(check)
    if long_probe:
        # A content canary beyond the upstream's reported QSA sparse-path threshold.
        # This is NOT a full-window/context-quality evaluation or a watchdog.
        code = secrets.token_hex(6)
        rows = [f'Reference entry {i}: ordinary filler without a retrieval code.' for i in range(1500)]
        rows.insert(750, f'The unique retrieval code is {code}.')
        prompt = '\n'.join(rows) + '\nReturn only the unique retrieval code, with no explanation.'
        check = {'name': 'long_context_canary', 'ok': False, 'expected': code}
        try:
            result = client.chat(model, prompt, max_tokens=128)
            check['result'] = result
            response = result['response']
            if response['usage']['prompt_tokens'] < 8192:
                raise ValueError('Canary too short to exercise intended sparse path')
            if (response['choices'][0]['message'].get('content') or '').strip() != code:
                raise ValueError('Long-context retrieval failed; inspect raw output, do not count speed as success')
            check['ok'] = True
        except Exception as exc:
            check['error'] = str(exc)
        checks.append(check)
    return checks


def benchmark(client, model, streams, runs, thinking="medium"):
    if thinking not in ('off', 'low', 'medium', 'xhigh'):
        raise ValueError('BENCH_THINKING must be off, low, medium or xhigh')
    kwargs = {'enable_thinking': thinking != 'off'}
    if thinking != 'off':
        kwargs['reasoning_effort'] = thinking
    pools = json.loads((ROOT / "tools/bench_prompts.json").read_text())
    batches = []
    for run in range(1, runs + 1):
        for pool in pools:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=streams) as executor:
                futures = [executor.submit(client.chat, model, prompt, pool["max_tokens"], chat_template_kwargs=kwargs)
                           for prompt in pool["prompts"][:streams]]
                results = []
                for future in futures:
                    try:
                        results.append(dict(ok=True, **future.result()))
                    except Exception as exc:
                        results.append({"ok": False, "error": str(exc)})
            elapsed = time.perf_counter() - start
            tokens = sum(result.get("completion_tokens", 0) for result in results)
            # One shared wall-clock window, not a sum of independent stream rates.
            batches.append({"name": pool["name"], "run": run, "streams": streams,
                            "ok": all(result["ok"] for result in results),
                            "elapsed_seconds": elapsed, "completion_tokens": tokens,
                            "aggregate_e2e_output_tokens_per_second": tokens / elapsed,
                            "results": results})
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "bench"])
    parser.add_argument("streams", nargs="?", type=int, default=1, choices=range(1, 9))
    parser.add_argument("-v", action="store_true", help="Accepted for compatibility; JSON always includes responses")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--long", action="store_true", help="Smoke only: add an >8k-token content canary (requires sufficient CTX)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    key = os.environ.get("API_KEY", "")
    if not key:
        parser.error("Set API_KEY to the key used by launch.sh")
    base = os.environ.get("BASE_URL", "http://127.0.0.1:" + os.environ.get("PORT", "30000"))
    client = Client(base, key)
    models, _ = client.request("/v1/models")
    model = models["data"][0]["id"]
    expected = json.loads((ROOT / "model.lock.json").read_text())["model_id"]
    if model != expected:
        raise ValueError(f"Unexpected served model {model!r}; expected {expected!r}")
    thinking = os.environ.get('BENCH_THINKING', 'medium')
    results = smoke(client, model, args.long) if args.mode == "smoke" else benchmark(client, model, args.streams, args.runs, thinking)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "base_url": base,
              "model": model, "mode": args.mode, "ok": all(row["ok"] for row in results),
              "measurement": "non-streaming end-to-end; includes prefill, decode and HTTP; NOT decode-only",
              "results": results}
    if args.mode == 'bench':
        report['benchmark_thinking'] = thinking
    receipt = ROOT / ".state/runtime.json"
    if receipt.exists():
        report["local_runtime_receipt"] = json.loads(receipt.read_text())
    launch = ROOT / '.state/last_launch.json'
    if launch.exists():
        report['local_launch_receipt'] = json.loads(launch.read_text())
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(f"API probe failed: {exc}")
