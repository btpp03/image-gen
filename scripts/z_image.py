#!/usr/bin/env python3
"""Z-Image-Turbo (mrfakename) free image generator via HuggingFace Space.

The qwen-image-2-1 Space had its Gradio queue stopped by the owner (503
"Queue is stopped"), so this script targets the mrfakename Z-Image-Turbo
Space instead, which uses a different, single-call API:

    POST /gradio_api/call/generate_image  [prompt, height, width, steps, seed, randomize]

Stdlib only. Proxy pool support matches qwen_image.py usage in the workflow:
    python z_image.py "prompt" --proxy-pool pool.txt --proxy-policy round-robin --attempts 40
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

BASE = "https://mrfakename-z-image-turbo.hf.space"
UA = "Mozilla/5.0 (compatible; z-image-cli/1.1)"
OPENER = urllib.request.build_opener()

QUOTA_MARKERS = ("zerogpu quota", "zerogpu runs limit", "exceeded your zerogpu", "too many requests")

# Buckets (multiple of 8, area ~= res^2)
RATIO_BUCKETS = {
    "1:1": (1024, 1024), "4:3": (1152, 896), "3:4": (896, 1152),
    "3:2": (1248, 832), "2:3": (832, 1248), "16:9": (1280, 720), "9:16": (720, 1280),
}


class QuotaError(RuntimeError):
    pass


class QueueStopped(RuntimeError):
    pass


def use_proxy(proxy=None):
    global OPENER
    if not proxy:
        OPENER = urllib.request.build_opener()
        return
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})]
    try:
        import socks  # noqa: F401
        if proxy.startswith(("socks5://", "socks5h://")):
            handlers.insert(0, urllib.request.ProxyHandler({}))
    except ImportError:
        pass
    OPENER = urllib.request.build_opener(*handlers)


def submit(prompt, height, width, steps, seed, token=None, timeout=60):
    body = json.dumps({"data": [prompt, height, width, steps, seed, True]}).encode()
    headers = {"User-Agent": UA, "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(f"{BASE}/gradio_api/call/generate_image", data=body, headers=headers)
    with OPENER.open(req, timeout=timeout) as r:
        d = json.loads(r.read())
    eid = d.get("event_id")
    if not eid:
        raise RuntimeError(f"no event_id: {d!r}")
    return eid


def parse_sse(lines):
    # Parse an SSE stream. IMPORTANT: data buffer must reset on every
    # "event:" line — Gradio sends heartbeats ("data: null") before
    # complete, and concatenating them breaks json.loads.
    event, data = None, []
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if line.startswith("event:"):
            event = line[6:].strip()
            data = []
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and event in ("complete", "error"):
            blob = " ".join(data)
            low = blob.lower()
            if event == "error" and any(m in low for m in QUOTA_MARKERS):
                raise QuotaError(blob[:200])
            if event == "error":
                raise RuntimeError(blob[:200])
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                raise RuntimeError(f"bad payload: {blob[:200]}")
            return payload
    raise TimeoutError("SSE stream ended without complete/error")


def watch(event_id, token=None, timeout=600):
    headers = {"User-Agent": UA, "Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(f"{BASE}/gradio_api/call/generate_image/{event_id}", headers=headers)
    with OPENER.open(req, timeout=timeout) as r:
        return parse_sse(r)


def download(url, dest, token=None, timeout=300):
    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with OPENER.open(req, timeout=timeout) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    with open(dest, "rb") as fh:
        if fh.read(4) != b"\x89PNG":
            raise RuntimeError(f"not a PNG: {dest}")
    return dest


def extract_url(payload):
    # Recursively find the image URL — payload shape varies between
    # Gradio versions: "url", [[{...}]], [{...}], or a bare URL string.
    if isinstance(payload, str):
        return payload if payload.startswith("http") else None
    if isinstance(payload, dict):
        u = payload.get("url")
        return u if isinstance(u, str) and u.startswith("http") else None
    if isinstance(payload, (list, tuple)):
        for item in payload:
            u = extract_url(item)
            if u:
                return u
    return None


def generate_once(args, token, prompt, proxy):
    use_proxy(proxy)
    tag = proxy or "direct"
    scale = args.resolution / 1024.0
    w0, h0 = RATIO_BUCKETS.get(args.ratio, (1024, 1024))
    width = int(round(w0 * scale / 8.0)) * 8
    height = int(round(h0 * scale / 8.0)) * 8
    seed = random.randint(0, 2**31 - 1) if args.seed is None else args.seed

    t0 = time.time()
    print(f"  [{tag}] submitting {width}x{height} steps={args.steps} seed={seed} ...", file=sys.stderr, flush=True)
    eid = submit(prompt, height, width, args.steps, seed, token)
    payload = watch(eid, token, args.timeout)

    url = extract_url(payload)
    if not url:
        raise RuntimeError(f"no image url in payload: {str(payload)[:200]}")

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"zimage-{time.strftime('%Y%m%d-%H%M%S')}-{seed}.png")
    download(url if url.startswith("http") else BASE + "/gradio_api/file=" + url, dest, token)
    print(f"  [{tag}] done in {time.time()-t0:.1f}s -> {dest}", file=sys.stderr, flush=True)
    return dest, seed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Z-Image-Turbo (mrfakename) free gen")
    ap.add_argument("prompt")
    ap.add_argument("--ratio", default="Auto", choices=["Auto"] + list(RATIO_BUCKETS))
    ap.add_argument("--resolution", type=int, default=1024, choices=[1024, 1536, 2048])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--proxy-pool", default=None, help="file (one per line) or comma list")
    ap.add_argument("--proxy-policy", default="random", choices=["random", "round-robin"])
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default="./z_output")
    ap.add_argument("--token", default=None)
    args = ap.parse_args(argv)

    token = args.token or os.environ.get("HF_TOKEN", "").strip() or None
    proxies = []
    if args.proxy_pool:
        spec = args.proxy_pool
        if os.path.isfile(spec):
            proxies = [l.strip() for l in open(spec) if l.strip() and not l.startswith("#")]
        else:
            proxies = [p.strip() for p in spec.split(",") if p.strip()]

    n_attempts = args.attempts or (len(proxies) + 1)
    queue_strikes = 0
    last_err = None
    for i in range(n_attempts):
        if not proxies:
            proxy = None if i == 0 else None
        else:
            proxy = random.choice(proxies) if args.proxy_policy == "random" else proxies[i % len(proxies)]
        try:
            dest, seed = generate_once(args, token, args.prompt, proxy)
            print(dest)
            return 0
        except QuotaError as e:
            last_err = e
            print(f"  attempt {i+1} quota on {proxy or 'direct'}: {e} — rotating", file=sys.stderr, flush=True)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 503:
                queue_strikes += 1
                print(f"  attempt {i+1} 503 (queue stopped?) on {proxy or 'direct'} — backing off 45s "
                      f"(strike {queue_strikes})", file=sys.stderr, flush=True)
                if queue_strikes >= 3:
                    print("ERROR: queue stopped 3x in a row — Space likely frozen, aborting", file=sys.stderr)
                    return 2
                time.sleep(45)
            else:
                print(f"  attempt {i+1} HTTP {e.code} on {proxy or 'direct'} — rotating", file=sys.stderr, flush=True)
        except Exception as e:
            last_err = e
            print(f"  attempt {i+1} network error on {proxy or 'direct'}: {e} — rotating", file=sys.stderr, flush=True)
    print(f"ERROR: all {n_attempts} attempts exhausted; last: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
