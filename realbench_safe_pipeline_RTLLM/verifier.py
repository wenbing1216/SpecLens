"""Verification runner for RTLLM tasks.

Default policy:

- use `iverilog + vvp` for most tasks
- use `verilator` for a small set of known Icarus-incompatible tasks
- if `iverilog` compile fails with a known simulator-unsupported pattern, retry with `verilator`
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import CandidateVerilog, TaskSpec, VerificationResult

_KNOWN_VERILATOR_TASK_OVERRIDES = {
    # Icarus rejects `break` in the provided testbench.
    "asyn_fifo",
    # Icarus rejects declaration-time whole-array initialization in the provided testbench.
    "ring_counter",
    # Original verified RTL passes under Verilator but fails under Icarus on this benchmark.
    "clkgenerator",
}

_IVERILOG_UNSUPPORTED_PATTERNS = (
    "sorry: break statements not supported",
    "assignment to an entire array or to an array slice is not yet supported",
)


@dataclass(slots=True)
class _SimulatorRun:
    simulator: str
    compile_result: subprocess.CompletedProcess[str]
    run_result: subprocess.CompletedProcess[str] | None
    passed: bool


class VerificationWorkspace:
    """Context manager that materializes one isolated verification workspace."""

    def __init__(self, task: TaskSpec, candidate: CandidateVerilog) -> None:
        self.task = task
        self.candidate = candidate
        self._tempdir = TemporaryDirectory()
        self.path = Path(self._tempdir.name)

    def __enter__(self) -> "VerificationWorkspace":
        shutil.copytree(
            self.task.task_dir,
            self.path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".DS_Store",
                "Thumbs.db",
                "*~",
                "*.swp",
                "*.tmp",
            ),
        )
        top_file = self.path / "top_module.sv"
        top_file.write_text(self.candidate.clean_verilog + "\n", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tempdir.cleanup()


def run_compile_only(
    task: TaskSpec,
    candidate: CandidateVerilog,
    timeout_sec: int = 600,
) -> VerificationResult:
    """Run only compile inside an isolated temp workspace."""

    with VerificationWorkspace(task=task, candidate=candidate) as workspace:
        primary = _preferred_simulator(task)
        primary_run = _run_with_simulator(primary, workspace, task, timeout_sec, compile_only=True)
        final_run = primary_run
        fallback_from = None
        fallback_details = ""

        if (
            primary == "iverilog"
            and _should_retry_with_verilator(task, primary_run.compile_result)
        ):
            fallback_from = "iverilog"
            fallback_run = _run_with_simulator("verilator", workspace, task, timeout_sec, compile_only=True)
            final_run = fallback_run
            fallback_details = _build_fallback_details(primary_run.compile_result)

        compile_summary = _summarize_compile_result(final_run.compile_result, final_run.simulator)
        if fallback_details:
            compile_summary = f"{fallback_details}\n{compile_summary}".strip()

        return VerificationResult(
            candidate_id=candidate.candidate_id,
            workspace_dir=str(workspace.path),
            simulator=final_run.simulator,
            fallback_from_simulator=fallback_from,
            compile_returncode=final_run.compile_result.returncode,
            run_returncode=None,
            compile_stdout=final_run.compile_result.stdout,
            compile_stderr=final_run.compile_result.stderr,
            run_stdout="",
            run_stderr="",
            passed=False,
            compile_summary=compile_summary,
            evaluation_summary=compile_summary,
        )


def run_verification(
    task: TaskSpec,
    candidate: CandidateVerilog,
    timeout_sec: int = 600,
) -> VerificationResult:
    """Run compile then simulation inside an isolated temp workspace."""

    with VerificationWorkspace(task=task, candidate=candidate) as workspace:
        primary = _preferred_simulator(task)
        primary_run = _run_with_simulator(primary, workspace, task, timeout_sec, compile_only=False)
        final_run = primary_run
        fallback_from = None
        fallback_details = ""

        if (
            primary == "iverilog"
            and _should_retry_with_verilator(task, primary_run.compile_result)
        ):
            fallback_from = "iverilog"
            fallback_run = _run_with_simulator("verilator", workspace, task, timeout_sec, compile_only=False)
            final_run = fallback_run
            fallback_details = _build_fallback_details(primary_run.compile_result)

        compile_summary = _summarize_compile_result(final_run.compile_result, final_run.simulator)
        evaluation_summary = compile_summary
        if final_run.run_result is not None:
            evaluation_summary = _summarize_run_result(final_run.run_result, final_run.passed, final_run.simulator)
        if fallback_details:
            compile_summary = f"{fallback_details}\n{compile_summary}".strip()
            evaluation_summary = f"{fallback_details}\n{evaluation_summary}".strip()

        return VerificationResult(
            candidate_id=candidate.candidate_id,
            workspace_dir=str(workspace.path),
            simulator=final_run.simulator,
            fallback_from_simulator=fallback_from,
            compile_returncode=final_run.compile_result.returncode,
            run_returncode=final_run.run_result.returncode if final_run.run_result else None,
            compile_stdout=final_run.compile_result.stdout,
            compile_stderr=final_run.compile_result.stderr,
            run_stdout=final_run.run_result.stdout if final_run.run_result else "",
            run_stderr=final_run.run_result.stderr if final_run.run_result else "",
            passed=final_run.passed,
            compile_summary=compile_summary,
            evaluation_summary=evaluation_summary,
        )


def _preferred_simulator(task: TaskSpec) -> str:
    if task.task_id in _KNOWN_VERILATOR_TASK_OVERRIDES and shutil.which("verilator"):
        return "verilator"
    return "iverilog"


def _should_retry_with_verilator(task: TaskSpec, compile_result: subprocess.CompletedProcess[str]) -> bool:
    if compile_result.returncode == 0:
        return False
    if not shutil.which("verilator"):
        return False
    joined = f"{compile_result.stdout}\n{compile_result.stderr}".lower()
    return any(pattern in joined for pattern in _IVERILOG_UNSUPPORTED_PATTERNS)


def _run_with_simulator(
    simulator: str,
    workspace: VerificationWorkspace,
    task: TaskSpec,
    timeout_sec: int,
    *,
    compile_only: bool,
) -> _SimulatorRun:
    if simulator == "iverilog":
        return _run_with_iverilog(workspace, task, timeout_sec, compile_only=compile_only)
    if simulator == "verilator":
        return _run_with_verilator(workspace, task, timeout_sec, compile_only=compile_only)
    raise ValueError(f"unsupported simulator: {simulator}")


def _run_with_iverilog(
    workspace: VerificationWorkspace,
    task: TaskSpec,
    timeout_sec: int,
    *,
    compile_only: bool,
) -> _SimulatorRun:
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            "simv",
            "top_module.sv",
            task.testbench_path.name,
        ],
        cwd=workspace.path,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    run_result = None
    passed = False
    if compile_result.returncode == 0 and not compile_only:
        run_result = subprocess.run(
            ["vvp", "simv"],
            cwd=workspace.path,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        passed = run_result.returncode == 0 and _run_output_passed(run_result.stdout, run_result.stderr)

    return _SimulatorRun(
        simulator="iverilog",
        compile_result=compile_result,
        run_result=run_result,
        passed=passed,
    )


def _run_with_verilator(
    workspace: VerificationWorkspace,
    task: TaskSpec,
    timeout_sec: int,
    *,
    compile_only: bool,
) -> _SimulatorRun:
    tb_top = _parse_testbench_top_module(workspace.path / task.testbench_path.name) or "testbench"
    compile_result = subprocess.run(
        [
            "verilator",
            "--binary",
            "--timing",
            "-Wno-fatal",
            "--top-module",
            tb_top,
            "top_module.sv",
            task.testbench_path.name,
        ],
        cwd=workspace.path,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    run_result = None
    passed = False
    if compile_result.returncode == 0 and not compile_only:
        executable = workspace.path / "obj_dir" / f"V{tb_top}"
        if executable.is_file():
            run_result = subprocess.run(
                [str(executable)],
                cwd=workspace.path,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            passed = run_result.returncode == 0 and _run_output_passed(run_result.stdout, run_result.stderr)
        else:
            run_result = subprocess.CompletedProcess(
                args=[str(executable)],
                returncode=1,
                stdout="",
                stderr=f"Verilator produced no executable at {executable}",
            )

    return _SimulatorRun(
        simulator="verilator",
        compile_result=compile_result,
        run_result=run_result,
        passed=passed,
    )


def _parse_testbench_top_module(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text)
    if match is None:
        return None
    return match.group(1)


def _build_fallback_details(compile_result: subprocess.CompletedProcess[str]) -> str:
    joined = "\n".join(
        part for part in [compile_result.stdout.strip(), compile_result.stderr.strip()] if part
    ).strip()
    body = joined if joined else "Icarus compile failed without captured stderr/stdout."
    return (
        "Icarus compile hit a known unsupported construct and verification retried with Verilator.\n"
        f"Original Icarus compile output:\n{body}"
    ).strip()


def _run_output_passed(stdout: str, stderr: str) -> bool:
    joined = f"{stdout}\n{stderr}"
    if re.search(r"your design passed", joined, flags=re.IGNORECASE):
        return True

    mismatch_counts = [
        int(match.group(1))
        for match in re.finditer(
            r"Mismatches:\s*(\d+)\s+in\s+\d+\s+samples",
            joined,
            flags=re.IGNORECASE,
        )
    ]
    if mismatch_counts:
        return mismatch_counts[-1] == 0

    failure_match = re.search(
        r"test completed with\s+(\d+)\s*/?\s*\d*\s*failures",
        joined,
        flags=re.IGNORECASE,
    )
    if failure_match is not None:
        return int(failure_match.group(1)) == 0

    error_match = re.search(
        r"test completed with\s+(\d+)\s+errors?",
        joined,
        flags=re.IGNORECASE,
    )
    if error_match is not None:
        return int(error_match.group(1)) == 0

    lowered = joined.lower()
    if "===========error===========" in lowered or "===========failed===========" in lowered:
        return False
    if "test failed:" in lowered:
        return False

    return False


def _summarize_compile_result(result: subprocess.CompletedProcess[str], simulator: str) -> str:
    simulator_label = "Icarus" if simulator == "iverilog" else "Verilator"
    if result.returncode == 0:
        return f"{simulator_label} compile succeeded."
    joined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return f"{simulator_label} compile failed with return code {result.returncode}.\n{joined}".strip()


def _summarize_run_result(
    result: subprocess.CompletedProcess[str],
    passed: bool,
    simulator: str,
) -> str:
    simulator_label = "Icarus" if simulator == "iverilog" else "Verilator"
    state = "passed" if passed else "failed"
    joined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return f"{simulator_label} run {state} with return code {result.returncode}.\n{joined}".strip()
