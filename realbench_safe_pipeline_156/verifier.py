"""Icarus Verilog runner for Evalhuman tasks."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import CandidateVerilog, TaskSpec, VerificationResult


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
    timeout_sec: int = 1000,
) -> VerificationResult:
    """Run only `iverilog` inside an isolated temp workspace."""

    with VerificationWorkspace(task=task, candidate=candidate) as workspace:
        compile_result = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                "simv",
                "top_module.sv",
                "testbench.sv",
            ],
            cwd=workspace.path,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        compile_summary = _summarize_compile_result(compile_result)
        return VerificationResult(
            candidate_id=candidate.candidate_id,
            workspace_dir=str(workspace.path),
            compile_returncode=compile_result.returncode,
            run_returncode=None,
            compile_stdout=compile_result.stdout,
            compile_stderr=compile_result.stderr,
            run_stdout="",
            run_stderr="",
            passed=False,
            compile_summary=compile_summary,
            evaluation_summary=compile_summary,
        )


def run_verification(
    task: TaskSpec,
    candidate: CandidateVerilog,
    timeout_sec: int = 1000,
) -> VerificationResult:
    """Run ``iverilog`` then ``vvp`` inside an isolated temp workspace."""

    with VerificationWorkspace(task=task, candidate=candidate) as workspace:
        compile_result = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                "simv",
                "top_module.sv",
                "testbench.sv",
            ],
            cwd=workspace.path,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        run_result = None
        passed = False
        compile_summary = _summarize_compile_result(compile_result)
        evaluation_summary = compile_summary

        if compile_result.returncode == 0:
            run_result = subprocess.run(
                ["vvp", "simv"],
                cwd=workspace.path,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            passed = run_result.returncode == 0 and _run_output_passed(run_result.stdout, run_result.stderr)
            evaluation_summary = _summarize_run_result(run_result, passed)

        return VerificationResult(
            candidate_id=candidate.candidate_id,
            workspace_dir=str(workspace.path),
            compile_returncode=compile_result.returncode,
            run_returncode=run_result.returncode if run_result else None,
            compile_stdout=compile_result.stdout,
            compile_stderr=compile_result.stderr,
            run_stdout=run_result.stdout if run_result else "",
            run_stderr=run_result.stderr if run_result else "",
            passed=passed,
            compile_summary=compile_summary,
            evaluation_summary=evaluation_summary,
        )


def _run_output_passed(stdout: str, stderr: str) -> bool:
    joined = f"{stdout}\n{stderr}"
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
    return False


def _summarize_compile_result(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode == 0:
        return "Icarus compile succeeded."
    joined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return f"Compile failed with return code {result.returncode}.\n{joined}".strip()


def _summarize_run_result(result: subprocess.CompletedProcess[str], passed: bool) -> str:
    state = "passed" if passed else "failed"
    joined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return f"Icarus run {state} with return code {result.returncode}.\n{joined}".strip()
