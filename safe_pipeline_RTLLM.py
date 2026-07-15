"""Thin script wrapper around `realbench_safe_pipeline_RTLLM.cli`."""

from __future__ import annotations

import sys


def _ensure_supported_python() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            "SpecLens requires Python 3.10 or newer. "
            f"Detected: {sys.version.split()[0]}"
        )


_ensure_supported_python()

from realbench_safe_pipeline_RTLLM.cli import main


if __name__ == "__main__":
    main()
