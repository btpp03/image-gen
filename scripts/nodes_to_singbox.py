#!/usr/bin/env python3
"""Turn a proxy-node subscription (vless / trojan / hysteria2 / ss URIs) into a
single sing-box config that exposes **one local HTTP proxy port per exit node**,
plus a plain proxy-pool file for qwen_image.py.

Why: HuggingFace ZeroGPU free quota is counted per caller IP. Every distinct
exit node = one more free quota window, so we fan out over the airport nodes.

    python nodes_to_singbox.py nodes.txt --out-config sbx.json --out-pool pool.txt
    python nodes_to_singbox.py --sub-url "$SUB" --sub-ua v2rayNG   # fetch + convert

Then:
    sing-box check -c sbx.json && sing-box run -c sbx.json &
    python qwen_image.py "a red fox" --proxy-pool pool.txt --proxy-policy round-robin

Stdlib only. IPv6-only nodes are skipped by default (GitHub runners have no IPv6).
"""

import argparse
import base64
import json
import os
import socket
import sys
import urllib.parse
import urllib.request

DEFAULT_PORTS = range(10800, 10900)
DEFAULT_UA = "v2rayNG/1.9.16"
TRANSPORTS = ("ws", "grpc", "http", "h2", "tcp")


def _flag(q, *names):
    for n in names:
        v = q.get(n)
        if v is not None and v.lower() in ("1", "true", "yes", ""):
            return True
    return False


def _transport(q):
    """Build the sing-box `transport` block from standard share-link query keys."""
    net = (q.get("type") or q.get("net") or "tcp").lower()
    if net in ("ws", "websocket"):
        t = {"type": "ws", "path": q.get("path") or "/"}
        host = q.get("host")
        if host:
            t["headers"] = {"Host": host}
        return t
    if net == "grpc":
        return {"type": "grpc", "service_name": q.get("serviceName") or q.get("servicename") or ""}
    if net in ("http", "h2"):
        t = {"type": "http", "host": [h for h in (q.get("host") or "").split(",") if h]}
        if q.get("path"):
            t["path"] = q["path"]
        return t
    return None


def _tls(q, server, enabled_fn):
    server_name = q.get("sni") or q.get("peer") or q.get("host") or server
    tls = {"enabled": True, "server_name": server_name}
    if enabled_fn(q):
        tls["insecure"] = True
    if q.get("fp"):
        tls["utls"] = {"enabled": True, "fingerprint": q["fp"]}
    if q.get("security") == "reality" or q.get("pbk"):
        tls["reality"] = {"enabled": True,
                          "public_key": q.get("pbk", ""),
                          "short_id": q.get("sid", "")}
    return tls


def parse_node(uri):
    """share-link URI -> (name, sing-box outbound dict) or None if unsupported."""
    uri = uri.strip()
    if not uri or uri.startswith("#"):
        return None
    if "://" not in uri:
        return None
    u = urllib.parse.urlparse(uri)
    scheme = u.scheme.lower()
    server = u.hostname
    if not server:
        return None
    q = {k.lower(): v for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True)}
    name = urllib.parse.unquote(u.fragment).strip() or f"{scheme}:{server}:{u.port}"
    insecure = lambda q: _flag(q, "insecure", "allowinsecure", "allow_insecure")  # noqa: E731

    if scheme in ("hysteria2", "hy2"):
        port = u.port or 443
        ob = {"type": "hysteria2", "server": server, "server_port": port,
              "password": urllib.parse.unquote(u.username or u.password or "")}
        if q.get("sni") or insecure(q):
            ob["tls"] = _tls(q, server, insecure)
        ob["tls"] = ob.get("tls", {"enabled": True, "server_name": server})
        if q.get("obfs"):
            ob["obfs"] = {"type": q["obfs"], "password": q.get("obfs-password", "")}
        return name, ob

    if scheme == "vless":
        port = u.port or 443
        ob = {"type": "vless", "server": server, "server_port": port,
              "uuid": urllib.parse.unquote(u.username or "")}
        if q.get("flow"):
            ob["flow"] = q["flow"]
        if q.get("security") in ("tls", "reality") or _flag(q, "tls"):
            ob["tls"] = _tls(q, server, insecure)
        tr = _transport(q)
        if tr:
            ob["transport"] = tr
        return name, ob

    if scheme == "trojan":
        port = u.port or 443
        ob = {"type": "trojan", "server": server, "server_port": port,
              "password": urllib.parse.unquote(u.username or u.password or ""),
              "tls": _tls(q, server, insecure)}
        tr = _transport(q)
        if tr:
            ob["transport"] = tr
        return name, ob

    if scheme in ("ss", "shadowsocks"):
        port = u.port or 8388
        userinfo = urllib.parse.unquote(u.username or "")
        password = urllib.parse.unquote(u.password or "")
        if not password:  # whole userinfo is base64(method:password)
            try:
                pad = "=" * (-len(userinfo) % 4)
                userinfo = base64.urlsafe_b64decode(userinfo + pad).decode()
            except Exception:
                return None
            userinfo = userinfo.replace(":", ":", 1)
            if ":" not in userinfo:
                return None
            userinfo, password = userinfo.split(":", 1)
        ob = {"type": "shadowsocks", "server": server, "server_port": port,
              "method": userinfo, "password": password}
        return name, ob

    return None


def has_ipv4(host):
    if ":" in host and not host.replace(".", "").replace(":", "").isdigit():
        return False  # literal IPv6 address
    try:
        socket.getaddrinfo(host, None, socket.AF_INET)
        return True
    except socket.gaierror:
        return False


def load_nodes(files, sub_url, sub_ua, token):
    blobs = []
    for path in files:
        if path == "-":
            blobs.append(sys.stdin.read())
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                blobs.append(fh.read())
    if sub_url:
        req = urllib.request.Request(sub_url, headers={"User-Agent": sub_ua})
        if token:
            req.add_header("Authorization", "Bearer " + token)
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="replace")
        blobs.append(raw)
    out = []
    for blob in blobs:
        blob = blob.strip()
        if not blob:
            continue
        # a whole blob may itself be one base64-encoded subscription body
        if "\n" not in blob and "://" not in blob:
            try:
                pad = "=" * (-len(blob) % 4)
                blob = base64.b64decode(blob + pad).decode("utf-8", errors="replace")
            except Exception:
                pass
        for line in blob.splitlines():
            line = line.strip()
            if "://" not in line:
                continue
            if not out or line not in out:
                out.append(line)
    return out


def build(nodes, skip_v6=True, limit=0, only=()):
    outs, table, seen, skipped = [], [], set(), []
    for uri in nodes:
        parsed = parse_node(uri)
        if not parsed:
            skipped.append(("unsupported", uri[:60]))
            continue
        name, ob = parsed
        if only and ob["type"] not in only:
            skipped.append(("proto-filter", f"{ob['type']} {name[:40]}"))
            continue
        if skip_v6 and not has_ipv4(ob["server"]):
            skipped.append(("no-ipv4", f"{ob['server']} {name[:30]}"))
            continue
        key = json.dumps(ob, sort_keys=True)
        if key in seen:
            skipped.append(("duplicate", name[:40]))
            continue
        seen.add(key)
        outs.append((name, ob))
    if limit:
        outs = outs[:limit]
    return outs, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="proxy node URIs -> sing-box config + pool file")
    ap.add_argument("nodes", nargs="*", help="files with one share-link per line ('-' = stdin)")
    ap.add_argument("--sub-url", help="subscription URL (fetched with --sub-ua)")
    ap.add_argument("--sub-ua", default=DEFAULT_UA, help=f"UA for --sub-url (default {DEFAULT_UA})")
    ap.add_argument("--token", help="optional bearer token for --sub-url")
    ap.add_argument("--out-config", default="sbx.json")
    ap.add_argument("--out-pool", default="pool.txt")
    ap.add_argument("--base-port", type=int, default=10800)
    ap.add_argument("--limit", type=int, default=0, help="keep at most N nodes")
    ap.add_argument("--only", default="", help="comma list of protocols to keep, e.g. hysteria2,vless")
    ap.add_argument("--allow-v6", action="store_true", help="keep IPv6-only nodes (needs IPv6 egress)")
    args = ap.parse_args(argv)

    only = tuple(p.strip().lower() for p in args.only.split(",") if p.strip())
    nodes = load_nodes(args.nodes, args.sub_url, args.sub_ua, args.token)
    if not nodes:
        print("ERROR: no nodes found (check the file / subscription token / UA)", file=sys.stderr)
        return 1

    picked, skipped = build(nodes, skip_v6=not args.allow_v6, limit=args.limit, only=only)
    if not picked:
        print("ERROR: every node was skipped (see --allow-v6 / --only)", file=sys.stderr)
        return 1

    cfg = {"log": {"level": "warn"}, "inbounds": [], "outbounds": [{"type": "direct", "tag": "direct"}],
           "route": {"rules": [], "final": "direct"}}
    pool = []
    for i, (name, ob) in enumerate(picked):
        port = args.base_port + i
        ob = dict(ob, tag=f"out-{i}")
        cfg["outbounds"].append(ob)
        cfg["inbounds"].append({"type": "mixed", "tag": f"in-{i}",
                                "listen": "127.0.0.1", "listen_port": port})
        cfg["route"]["rules"].append({"inbound": [f"in-{i}"], "outbound": f"out-{i}"})
        pool.append(f"http://127.0.0.1:{port}")
        print(f"  port {port}  {ob['type']:<10} {ob['server']}:{ob['server_port']}  {name[:44]}")

    with open(args.out_config, "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    with open(args.out_pool, "w") as fh:
        fh.write("\n".join(pool) + "\n")

    for reason, what in skipped:
        print(f"  skip ({reason}): {what}", file=sys.stderr)
    print(f"\n{len(picked)} proxies -> {args.out_config} / {args.out_pool} "
          f"({len(skipped)} skipped of {len(nodes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
