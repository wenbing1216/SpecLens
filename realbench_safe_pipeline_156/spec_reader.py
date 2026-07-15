"""Spec reader for Evalhuman text descriptions."""

from __future__ import annotations

import re
from pathlib import Path

from .models import TaskSpec


FORBIDDEN_ROOT_SUFFIXES = {".v", ".sv"}


class UnsafeTaskAccessError(RuntimeError):
    """Raised when code tries to read a forbidden task-root file."""


def is_forbidden_task_root_file(task_dir: Path, candidate_path: Path) -> bool:
    """Return True only for forbidden files directly under the task root."""

    try:
        relative = candidate_path.resolve().relative_to(task_dir.resolve())
    except ValueError:
        return False

    return len(relative.parts) == 1 and candidate_path.suffix.lower() in FORBIDDEN_ROOT_SUFFIXES


def assert_safe_task_read(task_dir: Path, candidate_path: Path) -> None:
    """Reject reads to task-root `.v` / `.sv` files."""

    if is_forbidden_task_root_file(task_dir=task_dir, candidate_path=candidate_path):
        raise UnsafeTaskAccessError(
            f"Refusing to read forbidden task-root answer file: {candidate_path}"
        )


def safe_read_text(task_dir: Path, candidate_path: Path, encoding: str = "utf-8") -> str:
    """Read text only when the path is outside the forbidden answer boundary."""

    assert_safe_task_read(task_dir=task_dir, candidate_path=candidate_path)
    return candidate_path.read_text(encoding=encoding)


def read_spec_text(task: TaskSpec) -> str:
    """Read and lightly normalize the plain-text task description."""

    raw = safe_read_text(task_dir=task.task_dir, candidate_path=task.spec_path)
    return clean_spec_text(raw)


def clean_spec_text(text: str) -> str:
    """Normalize whitespace while keeping technical content intact."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
