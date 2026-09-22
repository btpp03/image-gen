#!/usr/bin/env python3
"""Qwen-Image-2.1 free image generator / editor via HuggingFace ZeroGPU Space.

Standard library only (Python 3.8+). No API key required.

Text-to-image:
    python qwen_image.py "a red fox sleeping in autumn leaves"

Image editing (1-10 input images, refer to them as <image1>..<image10>):
    python qwen_image.py "make <image1> wear a tiny red scarf" --image fox.png

Proxy pool (defeats the per-IP ZeroGPU quota — a new IP is a new free quota):
    python qwen_image.py "prompt" --proxy-pool "http://a:8080,socks5://b:1080"
    PROXY_POOL_FILE=proxies.txt python qwen_image.py "prompt"
    python qwen_image.py "prompt" --proxy-list-url https://example.com/proxies.txt

Options:
    --image FILE (repeatable, up to 10)  condition images -> editing mode
    --resolution {1024,1536,2048}        target side length (default 1024 = fastest)
    --ratio Auto|1:1|4:3|3:4|3:2|2:3|16:9|9:16
    --steps N                            inference steps 8-60 (default 20)
    --neg "..."                          negative prompt (only used when --cfg > 1)
    --cfg F                              true CFG scale 1.0-10.0 (default 1.0 = off)
    --seed N                             fixed seed (reproducible); omit for random
    --no-randomize                       use --seed literally instead of randomizing
    --enhance                            rewrite the prompt first (extra Space call)
    --proxy-pool SPEC                    proxies (comma/newline separated, or a file path)
    --proxy-list-url URL                 fetch a proxy list (one per line / comma separated)
    --proxy-policy random|round-robin    default random
    --attempts N                         max tries across the pool (default = pool size + 1)
    --out-dir DIR                        default ./qwen_output
    --timeout SEC                        SSE poll timeout (default 600)

Prints the absolute path of the saved PNG. Also accepts HF token in ~/.hf_token,
HF_TOKEN env var, or --token (raises the shared ZeroGPU quota).

SOCKS proxies need PySocks:  pip install pysocks
"""

import argparse
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SPACE = "hugging-apps/qwen-image-2-1"
BASE = "https://hugging-apps-qwen-image-2-1.hf.space"
UA = "Mozilla/5.0 (compatible; qwen-image-cli/1.0)"
MAX_SEED = 2**31 - 1
MAX_INPUT_IMAGES = 10

# Set by use_proxy(); every request goes through this opener.
OPENER = urllib.request.build_opener()

QUOTA_MARKERS = (
    "zerogpu quota",
    "zerogpu runs limit",
    "exceeded your zerogpu",
    "too many requests",
)


class QuotaError(RuntimeError):
    """This caller IP is out of free ZeroGPU quota -> rotate the proxy."""


def use_proxy(proxy=None):
    """Point every subsequent request at `proxy` (None = direct)."""
    global OPENER
    if not proxy:
        OPENER = urllib.request.build_opener()
        return
    if proxy.startswith(("socks5://", "socks5h://", "socks4://", "socks4a://")):
        try:
            import socks  # noqa: F401  (PySocks registers the SOCKS handler)
        except ImportError:
            raise RuntimeError(
                f"proxy {proxy} is SOCKS; install PySocks first (pip install pysocks)"
            )
    OPENER = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )


def normalize_proxy(entry):
    """'1.2.3.4:8080' -> 'http://1.2.3.4:8080'; keep explicit schemes as-is."""
    entry = entry.strip()
    if not entry:
        return None
    if "://" not in entry:
        return "http://" + entry
    return entry


def parse_proxy_blob(text):
    out = []
    for line in re.split(r"[\r\n,;]+", text or ""):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        url = normalize_proxy(line)
        if url and url not in out:
            out.append(url)
    return out


def load_proxy_pool(spec=None, list_url=None):
    """Collect proxies from --proxy-pool (inline or file), --proxy-list-url and env vars."""
    proxies = []
    sources = []

    spec = spec or os.environ.get("PROXY_POOL") or ""
    if spec:
        if os.path.exists(spec):
            with open(spec) as fh:
                sources.append(parse_proxy_blob(fh.read()))
        else:
            sources.append(parse_proxy_blob(spec))

    pf = os.environ.get("PROXY_POOL_FILE")
    if pf:
        if not os.path.exists(pf):
            raise RuntimeError(f"PROXY_POOL_FILE not found: {pf}")
        with open(pf) as fh:
            sources.append(parse_proxy_blob(fh.read()))

    urls = [u for u in [list_url, os.environ.get("PROXY_POOL_URL")] if u]
    for u in urls:
        req = urllib.request.Request(u, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            sources.append(parse_proxy_blob(r.read().decode(errors="replace")))

    for group in sources:
        for p in group:
            if p not in proxies:
                proxies.append(p)
    return proxies


class Pool:
    """Rotating proxy pool; entries that hit the quota are retired."""

    def __init__(self, proxies, policy="random"):
        self.all = list(proxies)
        self.live = list(proxies) or [None]  # None = direct connection
        self.policy = policy
        self._i = 0

    def next(self):
        if not self.live:
            return None
        if self.policy == "round-robin":
            p = self.live[self._i % len(self.live)]
            self._i += 1
            return p
        return random.choice(self.live)

    def retire(self, proxy):
        if proxy in self.live:
            self.live.remove(proxy)
        return len(self.live)

    def __len__(self):
        return len(self.all)


def _token(args):
    if getattr(args, "token", None):
        return args.token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    path = os.path.expanduser("~/.hf_token")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            pass
    return None


def _headers(token=None, extra=None):
    h = {"User-Agent": UA}
    if token:
        h["Authorization"] = "Bearer " + token
    if extra:
        h.update(extra)
    return h


def _open(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or _headers())
    return OPENER.open(req, timeout=timeout)


def _json(url, data=None, headers=None, timeout=60):
    with _open(url, data, headers, timeout) as r:
        return json.load(r)


def upload_image(path, token=None):
    """Upload a local file to the Space; return its server-side path."""
    with open(path, "rb") as fh:
        blob = fh.read()
    name = os.path.basename(path)
    bnd = "----qw" + uuid.uuid4().hex
    ext = os.path.splitext(name)[1].lower()
    ctype = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")
    body = (
        f"--{bnd}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n".encode()
        + blob
        + f"\r\n--{bnd}--\r\n".encode()
    )
    hdrs = _headers(token, {"Content-Type": f"multipart/form-data; boundary={bnd}"})
    out = _json(BASE + "/gradio_api/upload", body, hdrs, timeout=180)
    if not out:
        raise RuntimeError("upload returned no path")
    return out[0]


def submit(api, data, token=None, timeout=60):
    r = _json(f"{BASE}/gradio_api/call/{api}",
              json.dumps({"data": data}).encode(),
              _headers(token, {"Content-Type": "application/json"}), timeout)
    if "event_id" not in r:
        raise RuntimeError(f"no event_id from /call/{api}: {r!r}")
    return r["event_id"]


def watch(api, event_id, timeout=600, token=None):
    """Poll the SSE result stream; return the parsed 'complete' payload."""
    url = f"{BASE}/gradio_api/call/{api}/{event_id}"
    req = urllib.request.Request(url, headers=_headers(token, {"Accept": "text/event-stream"}))
    t0 = time.time()
    result = None
    event = None
    try:
        with OPENER.open(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").rstrip("\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    payload = line[5:].strip()
                    if event == "complete":
                        result = json.loads(payload)
                        break
                    if event == "error":
                        if any(m in payload.lower() for m in QUOTA_MARKERS):
                            raise QuotaError(payload)
                        raise RuntimeError(f"space error: {payload}")
                    if event == "generating" and os.environ.get("QIMG_VERBOSE"):
                        print(f"  ... {payload}", file=sys.stderr, flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 402):
            raise QuotaError(f"HTTP {exc.code} from the Space")
        raise RuntimeError(f"HTTP {exc.code} on the result stream")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"stream failed after {time.time()-t0:.0f}s: {exc}")
    if result is None:
        raise RuntimeError("stream ended without a complete event (Space waking up? re-run)")
    return result


def download(url, dest, token=None, timeout=300):
    with _open(url, headers=_headers(token), timeout=timeout) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    return dest


def generate_once(args, token, gallery, prompt, seed, ratio, proxy):
    """One full prepare+generate round through `proxy`. Returns the saved path."""
    use_proxy(proxy)
    tag = proxy or "direct"

    prep = watch("prepare",
                 submit("prepare", [prompt, gallery, bool(args.enhance),
                                    0 if seed is None else seed, args.no_randomize is False,
                                    ratio], token),
                 args.timeout, token)
    final_prompt, ratio_used, seed_used = prompt, ratio, seed
    if isinstance(prep, list) and len(prep) >= 4:
        final_prompt, rewritten, seed_used, ratio_used = prep[0], prep[1], prep[2], prep[3]
        if rewritten:
            print(f"  enhanced prompt: {rewritten}", file=sys.stderr, flush=True)

    print(f"  [{tag}] generating ({args.steps} steps, {args.resolution}px, ratio={ratio_used}) ...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    out = watch("generate",
                submit("generate", [final_prompt, gallery, args.neg, args.cfg, args.steps,
                                    int(seed_used or 0), args.resolution, ratio_used], token),
                args.timeout, token)
    print(f"  [{tag}] done in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    item = out[0] if isinstance(out, list) else out
    if isinstance(item, dict):
        url = item.get("url") or (BASE + "/gradio_api/file=" + item.get("path", ""))
    else:
        url = item
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(args.out_dir, f"qwen-{stamp}-{int(seed_used or 0)}.png")
    download(url, dest, token)
    with open(dest, "rb") as fh:
        if fh.read(4) != b"\x89PNG":
            raise RuntimeError(f"downloaded file is not a PNG: {dest}")
    return dest, seed_used, ratio_used


def main(argv=None):
    ap = argparse.ArgumentParser(description="Qwen-Image-2.1 free image gen/edit")
    ap.add_argument("prompt")
    ap.add_argument("--image", action="append", default=[], help="condition image (repeat up to 10)")
    ap.add_argument("--resolution", type=int, default=1024, choices=[1024, 1536, 2048])
    ap.add_argument("--ratio", default="Auto",
                    choices=["Auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--neg", default="")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-randomize", action="store_true",
                    help="use --seed as-is (default: seed is randomized like the UI)")
    ap.add_argument("--enhance", action="store_true", help="rewrite prompt via companion Space first")
    ap.add_argument("--proxy-pool", default=None,
                    help="proxies: comma/newline separated, or a path to a file of one per line")
    ap.add_argument("--proxy-list-url", default=None, help="fetch a proxy list from this URL")
    ap.add_argument("--proxy-policy", default="random", choices=["random", "round-robin"])
    ap.add_argument("--attempts", type=int, default=0, help="max tries across the pool")
    ap.add_argument("--out-dir", default="./qwen_output")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--token")
    args = ap.parse_args(argv)

    if len(args.image) > MAX_INPUT_IMAGES:
        ap.error(f"at most {MAX_INPUT_IMAGES} --image inputs are supported")
    if not 8 <= args.steps <= 60:
        ap.error("--steps must be between 8 and 60")

    token = _token(args)
    os.makedirs(args.out_dir, exist_ok=True)

    pool = Pool(load_proxy_pool(args.proxy_pool, args.proxy_list_url), args.proxy_policy)
    if pool.all:
        print(f"  proxy pool: {len(pool.all)} entries (policy={args.proxy_policy})",
              file=sys.stderr, flush=True)
    attempts = args.attempts or (len(pool.live) + 1)

    # Uploads are proxy-independent (server-side paths), so do them once.
    gallery = []
    for p in args.image:
        if not os.path.exists(p):
            ap.error(f"input image not found: {p}")
        print(f"  uploading {p} ...", file=sys.stderr, flush=True)
        gallery.append(upload_image(p, token))

    seed = args.seed if args.seed is not None else random.randint(0, MAX_SEED)
    last = None
    for attempt in range(1, attempts + 1):
        proxy = pool.next()
        try:
            dest, seed_used, ratio_used = generate_once(
                args, token, gallery, args.prompt, seed, args.ratio, proxy)
            print(f"seed={seed_used} ratio={ratio_used} proxy={proxy or 'direct'} "
                  f"attempt={attempt}")
            print(os.path.abspath(dest))
            return 0
        except QuotaError as exc:
            last = exc
            left = pool.retire(proxy)
            print(f"  quota hit on {proxy or 'direct'} ({len(pool.live)} left) — rotating",
                  file=sys.stderr, flush=True)
            if not left:
                break
        except RuntimeError as exc:
            last = exc
            print(f"  attempt {attempt} failed: {exc}", file=sys.stderr, flush=True)
            if attempt >= attempts:
                break
            time.sleep(2)
        except (urllib.error.URLError, ssl.SSLError, OSError, ValueError) as exc:
            # Dead/flaky proxy, TLS reset mid-stream, truncated response... treat as a
            # bad exit IP and move on to the next one instead of dying.
            last = RuntimeError(f"network error on {proxy or 'direct'}: {exc!r}")
            pool.retire(proxy)
            print(f"  attempt {attempt} network error on {proxy or 'direct'}: {exc!r} — "
                  f"rotating ({len(pool.live)} left)", file=sys.stderr, flush=True)
            if attempt >= attempts or not pool.live:
                break
            time.sleep(2)

    msg = str(last) if last else "all attempts failed"
    if isinstance(last, QuotaError):
        msg += ("\nAll pool entries are out of ZeroGPU quota (or the pool is empty/direct). "
                "Add fresh proxies, or wait for the quota window to reset.")
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
