"""
Live view of the GPU and whatever pipeline job is running.

    python tools/watch.py              # refresh every 5s
    python tools/watch.py -n 15        # every 15s
    python tools/watch.py --once       # single snapshot

nvidia-smi alone tells you the card is busy; it does not tell you whether the
JOB is progressing. This shows both, plus a rate and ETA computed from what
actually lands on disk. Stdlib only. Ctrl-C to stop.

Costs nothing on the GPU -- nvidia-smi queries are free.
"""
import argparse, json, os, subprocess, time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "build", "pairs_raw.jsonl")
TOTAL_CHUNKS = 1904

GPU_FIELDS = ("utilization.gpu,memory.used,memory.total,temperature.gpu,"
              "power.draw,clocks.sm,utilization.memory,pstate,fan.speed,"
              "power.limit,clocks.max.sm,clocks_event_reasons.sw_power_cap,"
              "clocks_event_reasons.hw_thermal_slowdown")


def sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def gpu():
    out = sh(f"nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits")
    if not out:
        return None
    p = [x.strip() for x in out.split(",")]

    def num(i, default=0.0):
        try:
            return float(p[i])
        except (ValueError, IndexError):
            return default

    try:
        return {"util": num(0), "used": num(1), "total": num(2),
                "temp": num(3), "power": num(4), "clock": num(5),
                "memutil": num(6), "pstate": p[7] if len(p) > 7 else "?",
                "fan": num(8), "plimit": num(9, 1.0), "clkmax": num(10, 1.0),
                "cap_pwr": p[11] if len(p) > 11 else "",
                "cap_thm": p[12] if len(p) > 12 else ""}
    except (ValueError, IndexError):
        return None


def ollama():
    """Query /api/ps directly. The `ollama ps` CLI returns an empty table in
    0.33.0 even while a model is loaded and serving, so do not parse it."""
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://localhost:11434/api/ps", timeout=5).read())
    except Exception:
        return "server unreachable"
    models = d.get("models") or []
    if not models:
        return "no model loaded"
    m = models[0]
    size, vram = m.get("size", 0), m.get("size_vram", 0)
    # size_vram < size means part of the model spilled to system RAM, which
    # collapses throughput. This is the number worth watching.
    where = "100% GPU" if vram >= size > 0 else \
            f"{100*vram/max(size,1):.0f}% GPU  <- SPILLED TO CPU"
    q = m.get("details", {}).get("quantization_level", "?")
    return f"{m['name']} {size/1e9:.1f}GB {q} {where}"


def job():
    if not os.path.exists(RAW):
        return None
    ids, pairs = set(), 0
    try:
        with open(RAW, encoding="utf-8") as fh:
            for line in fh:
                pairs += 1
                try:
                    ids.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    except OSError:
        return None
    return {"chunks": len(ids), "pairs": pairs,
            "mtime": os.path.getmtime(RAW)}


def bar(frac, w=34):
    fill = int(frac * w)
    return "#" * fill + "." * (w - fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--interval", type=int, default=5)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    hist = deque(maxlen=60)          # (timestamp, chunks) for rate over a window
    try:
        while True:
            g, j, om = gpu(), job(), ollama()
            now = time.time()
            lines = []
            lines.append(f"  {time.strftime('%H:%M:%S')}   "
                         f"GPU {'' if g else '(nvidia-smi unavailable)'}")

            if g:
                vram_frac = g["used"] / max(g["total"], 1)
                lines.append(f"  sm     {bar(g['util']/100)} {g['util']:5.1f}%"
                             f"   (kernel resident)")
                # LLM decode is memory-BANDWIDTH bound, not compute bound. This
                # line is why request parallelism did not help: the bus is the
                # bottleneck, and adding streams only adds KV-cache pressure.
                lines.append(f"  mem bw {bar(g['memutil']/100)} "
                             f"{g['memutil']:5.1f}%   <- the real bottleneck")
                lines.append(f"  vram   {bar(vram_frac)} "
                             f"{g['used']/1024:5.2f}/{g['total']/1024:.1f} GB")
                throttle = ""
                if g["cap_pwr"].lower().startswith("active"):
                    throttle = "  THROTTLED: power cap"
                elif g["cap_thm"].lower().startswith("active"):
                    throttle = "  THROTTLED: thermal"
                lines.append(
                    f"  {g['pstate']}  {g['temp']:.0f}C  fan {g['fan']:.0f}%  "
                    f"{g['power']:.0f}/{g['plimit']:.0f}W  "
                    f"{g['clock']:.0f}/{g['clkmax']:.0f} MHz{throttle}")
            lines.append(f"  ollama  {om}")

            if j:
                hist.append((now, j["chunks"]))
                frac = j["chunks"] / TOTAL_CHUNKS
                lines.append("")
                lines.append(f"  job    {bar(frac)} "
                             f"{j['chunks']:>5}/{TOTAL_CHUNKS} "
                             f"({100*frac:4.1f}%)")
                lines.append(f"  pairs  {j['pairs']:,}")

                stale = now - j["mtime"]
                if len(hist) >= 2 and hist[-1][0] - hist[0][0] > 30:
                    dt = (hist[-1][0] - hist[0][0]) / 60
                    dc = hist[-1][1] - hist[0][1]
                    rate = dc / dt if dt else 0
                    if rate > 0:
                        eta_min = (TOTAL_CHUNKS - j["chunks"]) / rate
                        done_at = time.strftime(
                            "%H:%M", time.localtime(now + eta_min * 60))
                        lines.append(f"  rate   {rate:.2f} chunks/min   "
                                     f"ETA {eta_min/60:.1f}h  (~{done_at})")
                    else:
                        lines.append("  rate   0.00 chunks/min  <- STALLED?")
                warn = "  <- nothing written recently" if stale > 180 else ""
                lines.append(f"  last write {stale:.0f}s ago{warn}")

            body = "\n".join(lines)
            if a.once:
                print(body)
                return
            # redraw in place
            print("\033[2J\033[H" + body + "\n\n  Ctrl-C to stop", flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
