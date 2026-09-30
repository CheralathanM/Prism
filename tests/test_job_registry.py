"""Active-job registry: replacement workers defer model loading while a job is live."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from fdagent.voice.job_registry import ActiveJobRegistry


def test_register_and_unregister(tmp_path):
    r = ActiveJobRegistry(tmp_path)
    assert r.active() == []
    key = r.register("job-a")
    assert r.active() == ["job-a"]
    r.unregister(key)
    assert r.active() == []
    r.unregister("never-registered")  # idempotent


@pytest.mark.skipif(os.name == "nt", reason="POSIX process liveness check")
def test_markers_of_dead_processes_are_ignored_and_removed(tmp_path):
    (tmp_path / "ghost.job").write_text("999999\n")  # no such process
    (tmp_path / "garbage.job").write_text("not-a-pid\n")
    r = ActiveJobRegistry(tmp_path)
    assert r.active() == []
    assert list(tmp_path.glob("*.job")) == []


def test_wait_until_idle_blocks_while_a_job_is_active_then_returns(tmp_path):
    r = ActiveJobRegistry(tmp_path)
    r.register("live-job")
    clock = SimpleNamespace(t=0.0)

    def sleep(s):
        clock.t += s
        if clock.t >= 3.0:
            r.unregister("live-job")  # the job ends 3 s later

    waited = r.wait_until_idle(poll_s=0.5, sleep=sleep, clock=lambda: clock.t)
    assert waited == 3.0 and r.active() == []


def test_wait_until_idle_gives_up_after_max_wait(tmp_path):
    r = ActiveJobRegistry(tmp_path)
    r.register("stuck-job")
    clock = SimpleNamespace(t=0.0)

    def sleep(s):
        clock.t += s

    assert r.wait_until_idle(poll_s=1.0, max_wait_s=5.0, sleep=sleep, clock=lambda: clock.t) == 5.0


def test_prewarm_defers_loading_until_no_job_is_active(monkeypatch, tmp_path):
    pytest.importorskip("livekit.agents")
    import fdagent.voice.livekit_agent as la

    order = []

    class Reg:
        def wait_until_idle(self, max_wait_s):
            order.append(("wait", max_wait_s))
            return 1.5

    monkeypatch.setattr(la, "load_prewarmed", lambda s: order.append("load") or SimpleNamespace(
        stack=s.stack, whisper_model=s.whisper_model, load_s=0.0))
    proc = SimpleNamespace(userdata={})
    la.prewarm_process(proc, registry=Reg())
    assert order == [("wait", 600.0), "load"]
    assert la.PREWARM_KEY in proc.userdata


def test_torch_thread_cap_setting():
    pytest.importorskip("livekit.agents")
    import fdagent.voice.livekit_agent as la

    assert la.AgentSettings.from_env({}).torch_threads == 4
    assert la.AgentSettings.from_env({"FDAGENT_TORCH_THREADS": "2"}).torch_threads == 2
    la.limit_torch_threads(0)  # 0 = leave torch default; must be a no-op even without torch
