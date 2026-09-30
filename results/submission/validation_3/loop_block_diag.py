"""Event-loop blocking during jobs (LiveKit watchdog warnings), with the blocking stack location."""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

for d in sys.argv[1:]:
    OUT = Path(d)
    t0 = json.loads((OUT / "result_fdagent.json").read_text())["stream_start_time"]
    rows = []
    for line in (OUT / "agent.log").read_text().splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = j.get("message", "")
        if "event loop blocked" not in m:
            continue
        ts = datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp() - t0
        if ts < -1:
            continue  # startup imports, before the stream began
        dur = re.search(r"blocked for (\d+)ms", m)
        where = re.search(r' at "([^"]+)", line (\d+)', m)
        stack = j.get("stack", "") or ""
        frames = re.findall(r'File "([^"]+)", line \d+, in (\w+)', stack)
        top = [f"{Path(f).name}:{fn}" for f, fn in frames[-4:]]
        rows.append((round(ts, 1), int(dur.group(1)) if dur else None,
                     f"{Path(where.group(1)).name}:{where.group(2)}" if where else "", top))
    total = sum(r[1] or 0 for r in rows)
    print(f"{OUT.name:<13} blocks={len(rows)} total_blocked_ms={total} longest_ms={max([r[1] or 0 for r in rows], default=0)}")
    for r in sorted(rows, key=lambda r: -(r[1] or 0))[:4]:
        print(f"    t={r[0]:>6}s {r[1]}ms {r[2]} {r[3]}")
