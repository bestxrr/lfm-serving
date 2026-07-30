#!/usr/bin/env python3
"""Start vLLM, warm representative prefill/decode/sampler paths, then mark ready."""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SENTINEL = Path("/tmp/lfm-warmup-ready")


def cli_value(name: str, default: str) -> str:
    prefix = f"--{name}="
    for index, arg in enumerate(sys.argv[1:]):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == f"--{name}" and index + 2 < len(sys.argv):
            return sys.argv[index + 2]
    return default


def request(url: str, payload: dict | None = None, timeout: int = 180) -> None:
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}")


def main() -> int:
    SENTINEL.unlink(missing_ok=True)
    port = cli_value("port", "8000")
    model = cli_value("served-model-name", "LFM2.5-1.2B-Instruct")
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    )

    def stop(signum: int, _frame: object) -> None:
        if server.poll() is None:
            server.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    health = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + int(os.environ.get("LFM_WARMUP_START_TIMEOUT", "180"))
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return server.returncode or 1
        try:
            request(health, timeout=2)
            break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        server.terminate()
        raise TimeoutError("vLLM did not become healthy before warmup timeout")

    rng = random.Random(20260729)
    warmups = [
        (64, 128, 0.0, 1.0),
        (64, 128, 0.7, 0.9),
        (4400, 128, 0.0, 1.0),
        (4400, 128, 0.7, 0.9),
    ]
    endpoint = f"http://127.0.0.1:{port}/v1/completions"
    warmed_prompts: list[list[int]] = []
    for prompt_len, output_len, temperature, top_p in warmups:
        # Synthetic token IDs avoid polluting natural-language prefix entries.
        prompt_ids = [rng.randrange(1000, 60000) for _ in range(prompt_len)]
        warmed_prompts.append(prompt_ids)
        request(
            endpoint,
            {
                "model": model,
                "prompt": prompt_ids,
                "max_tokens": output_len,
                "temperature": temperature,
                "top_p": top_p,
                "ignore_eos": True,
            },
        )

    if os.environ.get("LFM_DRAFT_WARMUP_BURST", "0") == "1":
        # Exercise full-prefix-hit rows mixed with uncached rows before the
        # container is declared ready. This is the production regression cell
        # for the generic draft metadata request-set bug.
        burst_prompts = [
            warmed_prompts[-1],
            warmed_prompts[-1],
            [rng.randrange(1000, 60000) for _ in range(96)],
            [rng.randrange(1000, 60000) for _ in range(128)],
        ]

        def burst_request(prompt_ids: list[int]) -> None:
            request(
                endpoint,
                {
                    "model": model,
                    "prompt": prompt_ids,
                    "max_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(burst_request, burst_prompts))
        print("Draft warmup burst complete: 4 concurrent requests", flush=True)

    # This endpoint is version-dependent; cache reset is hygiene, not readiness.
    try:
        request(f"http://127.0.0.1:{port}/reset_prefix_cache", {}, timeout=10)
        print("Self-warmup complete; prefix cache reset", flush=True)
    except Exception as exc:
        print(f"Self-warmup complete; prefix cache reset unavailable: {exc}", flush=True)
    SENTINEL.touch()
    return server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
