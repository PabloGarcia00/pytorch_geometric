#!/usr/bin/env python3
"""Who's been using the app, from Caddy's access log (see Caddyfile).

Usage: deploy/visitors.py [days]   (default 7)

A Streamlit session is one long-lived websocket to /_stcore/stream; Caddy
logs it only when it closes (with its duration), so "sessions" counts
closed tabs/reloads -- still-open ones are listed under "open now".
"""
import collections
import gzip
import json
import socket
import subprocess
import sys
import time

CADDY = "pv-disaggregation-caddy-1"
HTTP_PORT_HEX = ":1F90"  # caddy listens on 8080 inside its container


def sh(cmd: str) -> bytes:
    return subprocess.run(["docker", "exec", CADDY, "sh", "-c", cmd], capture_output=True, check=True).stdout


def log_lines():
    for name in sh("ls /data/ | grep '^access.*\\.log'").decode().split():
        data = sh(f"cat /data/{name}")
        if data[:2] == b"\x1f\x8b":  # rolled logs are gzipped
            data = gzip.decompress(data)
        yield from data.splitlines()


def decode_ip(h: str) -> str:
    b = bytes.fromhex(h)
    if len(b) == 4:
        return socket.inet_ntoa(b[::-1])
    b = b"".join(b[i:i + 4][::-1] for i in range(0, 16, 4))
    return socket.inet_ntop(socket.AF_INET6, b).removeprefix("::ffff:")


def main() -> None:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 7
    cutoff = time.time() - days * 86400
    v = collections.defaultdict(lambda: dict(first=1e18, last=0, reqs=0, sessions=0, secs=0.0, denied=0, days=set()))
    for line in log_lines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("ts", 0) < cutoff or "request" not in e:
            continue
        req = e["request"]
        r = v[req.get("client_ip") or req["remote_ip"]]
        ts = e["ts"]
        r["first"], r["last"] = min(r["first"], ts), max(r["last"], ts)
        r["reqs"] += 1
        r["days"].add(time.strftime("%Y-%m-%d", time.gmtime(ts)))
        if e.get("status") == 401:
            r["denied"] += 1
        elif req.get("uri", "").startswith("/_stcore/stream"):
            r["sessions"] += 1
            r["secs"] += e.get("duration", 0)

    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))
    print(f"Last {days:g} days (UTC)\n")
    print(f"{'client IP':<40} {'first seen':<17} {'last seen':<17} {'days':>4} {'sessions':>8} {'time in app':>11} {'401s':>5}")
    for ip, r in sorted(v.items(), key=lambda kv: -kv[1]["last"]):
        print(f"{ip:<40} {fmt(r['first']):<17} {fmt(r['last']):<17} {len(r['days']):>4} "
              f"{r['sessions']:>8} {r['secs'] / 60:>10.0f}m {r['denied']:>5}")
    if not v:
        print("(no requests logged yet)")

    print("\nopen now:")
    conns = collections.Counter()
    for line in sh("cat /proc/net/tcp /proc/net/tcp6").decode().splitlines()[1:]:
        f = line.split()
        if len(f) > 3 and f[3] == "01" and f[1].endswith(HTTP_PORT_HEX):
            conns[decode_ip(f[2].split(":")[0])] += 1
    for ip, n in conns.most_common():
        print(f"  {ip}  ({n} connection{'s' if n > 1 else ''})")
    if not conns:
        print("  nobody")


if __name__ == "__main__":
    main()
