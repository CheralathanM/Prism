"""Cross-process registry of active agent jobs, used to keep model loading out of live sessions.

LiveKit keeps a pool of pre-started worker processes and spawns a replacement as soon as one
takes a job. That replacement's setup (PyTorch/Whisper/Silero loading) used to run while the
benchmark user was speaking and starved voice detection of CPU (VAD fell 8-12 s behind real
time). Jobs register a marker file here while they run; a newly spawned worker waits until no
job is active before loading its models. Markers of processes that no longer exist are
ignored and removed, so a crash cannot block prewarming forever.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable


def default_registry_dir() -> Path:
    return Path(os.getenv("FDAGENT_RUNTIME_DIR", tempfile.gettempdir())) / "fdagent-active-jobs"


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows. Windows runs jobs as threads
        # inside one process (LiveKit's thread executor), so other PIDs never hold jobs there.
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ActiveJobRegistry:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else default_registry_dir()

    def _marker(self, key: str) -> Path:
        return self.directory / f"{key}.job"

    def register(self, key: str | None = None) -> str:
        key = key or str(os.getpid())
        self.directory.mkdir(parents=True, exist_ok=True)
        self._marker(key).write_text(f"{os.getpid()}\n", encoding="utf-8")
        return key

    def unregister(self, key: str | None = None) -> None:
        try:
            self._marker(key or str(os.getpid())).unlink()
        except FileNotFoundError:
            pass

    def active(self) -> list[str]:
        """Keys of jobs whose owning process is still alive (stale markers are removed)."""
        if not self.directory.exists():
            return []
        live = []
        for m in sorted(self.directory.glob("*.job")):
            try:
                pid = int(m.read_text(encoding="utf-8").split()[0])
            except (OSError, ValueError, IndexError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                live.append(m.stem)
            else:
                try:
                    m.unlink()
                except OSError:
                    pass
        return live

    def wait_until_idle(self, poll_s: float = 0.5, max_wait_s: float = 600.0,
                        sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.monotonic) -> float:
        """Block until no job is active (or ``max_wait_s`` elapses). Returns seconds waited."""
        t0 = clock()
        while self.active() and clock() - t0 < max_wait_s:
            sleep(poll_s)
        return round(clock() - t0, 2)
