"""Lightweight runtime monitor for diagnosing external interruption and deep task stalls."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


class RunMonitor:
    """Persist a small heartbeat/status file so abrupt interruption leaves evidence on disk."""

    def __init__(self, trace_root: Path) -> None:
        self.trace_root = trace_root
        self.status_path = self.trace_root / "runtime_status.json"
        self.signal_path = self.trace_root / "runtime_signal.json"
        self.abnormal_termination_path = self.trace_root / "abnormal_termination.json"
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "pid": os.getpid(),
            "started_at": time.time(),
            "exit_status": "running",
            "current_task_name": None,
            "current_task_id": None,
            "current_variant": None,
            "current_phase": "startup",
            "phase_details": {},
            "last_update_at": time.time(),
            "last_heartbeat_at": time.time(),
        }
        self._installed_handlers: dict[int, Any] = {}
        self._closed = False
        self.trace_root.mkdir(parents=True, exist_ok=True)
        if self.abnormal_termination_path.exists():
            self.abnormal_termination_path.unlink()
        self._flush()

    def _flush(self) -> None:
        payload = dict(self._state)
        self.status_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update(
        self,
        *,
        task_name: str | None = None,
        task_id: str | None = None,
        variant: str | None = None,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
        heartbeat: bool = True,
    ) -> None:
        with self._lock:
            if task_name is not None:
                self._state["current_task_name"] = task_name
            if task_id is not None:
                self._state["current_task_id"] = task_id
            if variant is not None:
                self._state["current_variant"] = variant
            if phase is not None:
                self._state["current_phase"] = phase
            if details is not None:
                self._state["phase_details"] = details
            now = time.time()
            self._state["last_update_at"] = now
            if heartbeat:
                self._state["last_heartbeat_at"] = now
            self._flush()

    def heartbeat(self, *, phase: str | None = None, details: dict[str, Any] | None = None) -> None:
        self.update(phase=phase, details=details, heartbeat=True)

    def mark_task_complete(self) -> None:
        self.update(
            task_name=None,
            task_id=None,
            variant=None,
            phase="idle",
            details={},
            heartbeat=True,
        )

    def mark_signal(self, signum: int) -> None:
        signal_name = signal.Signals(signum).name
        with self._lock:
            self._state["exit_status"] = f"signal:{signal_name}"
            now = time.time()
            self._state["last_update_at"] = now
            self._state["last_heartbeat_at"] = now
            self._flush()
            self.signal_path.write_text(
                json.dumps(
                    {
                        "signal": signal_name,
                        "signum": signum,
                        "captured_at": now,
                        "state_snapshot": self._state,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def close(self, *, exit_status: str = "normal_exit") -> None:
        with self._lock:
            if self._closed:
                return
            self._state["exit_status"] = exit_status
            now = time.time()
            self._state["last_update_at"] = now
            self._state["last_heartbeat_at"] = now
            self._flush()
            self._closed = True

    def install_signal_handlers(self) -> None:
        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
            signum = getattr(signal, sig_name, None)
            if signum is None:
                continue
            previous = signal.getsignal(signum)
            self._installed_handlers[signum] = previous

            def _handler(
                received_signum: int,
                frame: Any,
                *,
                _monitor: RunMonitor = self,
                _previous: Any = previous,
            ) -> None:
                del frame
                _monitor.mark_signal(received_signum)
                if callable(_previous):
                    _previous(received_signum, None)
                    return
                if _previous == signal.SIG_DFL:
                    raise SystemExit(128 + received_signum)
                if _previous == signal.SIG_IGN:
                    raise SystemExit(128 + received_signum)
                raise SystemExit(128 + received_signum)

            signal.signal(signum, _handler)


_ACTIVE_RUN_MONITOR: RunMonitor | None = None


def set_active_run_monitor(monitor: RunMonitor | None) -> None:
    global _ACTIVE_RUN_MONITOR
    _ACTIVE_RUN_MONITOR = monitor


def get_active_run_monitor() -> RunMonitor | None:
    return _ACTIVE_RUN_MONITOR


def create_and_install_run_monitor(trace_root: Path) -> RunMonitor:
    monitor = RunMonitor(trace_root=trace_root)
    monitor.install_signal_handlers()
    _launch_runtime_watchdog(monitor)
    atexit.register(_atexit_close_monitor, monitor)
    set_active_run_monitor(monitor)
    return monitor


def _atexit_close_monitor(monitor: RunMonitor) -> None:
    if monitor is not get_active_run_monitor():
        return
    monitor.close(exit_status="process_exit")


def _launch_runtime_watchdog(monitor: RunMonitor) -> None:
    """Start a detached sidecar that can diagnose stale `running` status after crashes."""

    script_path = Path(__file__).with_name("runtime_watchdog.py")
    command = [
        sys.executable,
        str(script_path),
        "--trace-root",
        str(monitor.trace_root),
        "--run-pid",
        str(monitor._state["pid"]),
        "--run-started-at",
        str(monitor._state["started_at"]),
    ]
    try:
        with open(os.devnull, "rb") as devnull_in, open(os.devnull, "wb") as devnull_out:
            subprocess.Popen(
                command,
                stdin=devnull_in,
                stdout=devnull_out,
                stderr=devnull_out,
                close_fds=True,
                start_new_session=True,
            )
    except Exception:
        # The runtime monitor should stay best-effort diagnostic infrastructure.
        return
