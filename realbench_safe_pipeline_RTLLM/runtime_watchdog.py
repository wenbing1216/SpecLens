"""Detached sidecar that detects stale `running` runtime status after abnormal termination."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_POLL_INTERVAL_SEC = 5.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime watchdog for stale pipeline status files")
    parser.add_argument("--trace-root", required=True, help="Trace root containing runtime_status.json")
    parser.add_argument("--run-pid", type=int, required=True, help="PID of the monitored pipeline process")
    parser.add_argument("--run-started-at", type=float, required=True, help="Epoch timestamp recorded at run start")
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help="Seconds between lightweight status checks",
    )
    parser.add_argument(
        "--single-check",
        action="store_true",
        help="Run one watchdog check and exit; used for local verification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_root = Path(args.trace_root).resolve()
    poll_interval_sec = max(1.0, float(args.poll_interval_sec))

    while True:
        outcome = run_one_check(
            trace_root=trace_root,
            run_pid=int(args.run_pid),
            run_started_at=float(args.run_started_at),
        )
        if outcome != "continue" or args.single_check:
            return
        time.sleep(poll_interval_sec)


def run_one_check(*, trace_root: Path, run_pid: int, run_started_at: float) -> str:
    """Return `continue`, `stop`, or `abnormal_written` after one lightweight probe."""

    status_path = trace_root / "runtime_status.json"
    if not status_path.exists():
        return "continue"

    status = _read_json_file(status_path)
    if not isinstance(status, dict):
        return "continue"

    recorded_started_at = _as_float(status.get("started_at"))
    if recorded_started_at is None or abs(recorded_started_at - run_started_at) > 1e-6:
        return "stop"

    exit_status = str(status.get("exit_status", "")).strip()
    if exit_status and exit_status != "running":
        return "stop"

    recorded_pid = _as_int(status.get("pid"))
    if recorded_pid is None or recorded_pid != run_pid:
        return "stop"

    if _pid_is_alive(run_pid):
        return "continue"

    abnormal_path = trace_root / "abnormal_termination.json"
    if abnormal_path.exists():
        return "stop"

    summary_exists = _is_current_run_artifact(trace_root / "summary.json", run_started_at)
    run_errors_exists = _is_current_run_artifact(trace_root / "run_errors.json", run_started_at)
    runtime_signal_exists = _is_current_run_artifact(trace_root / "runtime_signal.json", run_started_at)

    payload = {
        "detected_at_epoch": time.time(),
        "trace_root": str(trace_root),
        "recorded_pid": run_pid,
        "recorded_exit_status": exit_status or "running",
        "run_started_at": run_started_at,
        "last_update_at": _as_float(status.get("last_update_at")),
        "last_heartbeat_at": _as_float(status.get("last_heartbeat_at")),
        "current_task_name": status.get("current_task_name"),
        "current_task_id": status.get("current_task_id"),
        "current_variant": status.get("current_variant"),
        "current_phase": status.get("current_phase"),
        "phase_details": status.get("phase_details") or {},
        "summary_exists": summary_exists,
        "run_errors_exists": run_errors_exists,
        "runtime_signal_exists": runtime_signal_exists,
        "pid_alive": False,
        "diagnosis": "process_disappeared_while_runtime_status_still_running",
        "note": (
            "The watchdog observed that the monitored PID no longer exists while runtime_status.json "
            "still reports exit_status='running'."
        ),
    }
    abnormal_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return "abnormal_written"


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _is_current_run_artifact(path: Path, run_started_at: float) -> bool:
    try:
        return path.exists() and path.stat().st_mtime >= run_started_at
    except Exception:
        return False


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
