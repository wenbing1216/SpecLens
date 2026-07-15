"""Command-line entrypoint for the dataset-adapted RTLLM pipeline."""

from __future__ import annotations

import argparse

from .llm import build_provider_bundle
from .pipeline import (
    SUPPORTED_PIPELINE_VARIANTS,
    default_config,
    discover_tasks,
    run_dataset_pipeline,
)
from .trace import build_run_summary, write_run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RTLLM-safe pipeline")
    parser.add_argument(
        "--dataset-root",
        default="RTLLMv1.1",
        help=(
            "Dataset root to scan. Common aliases: RTLLM or RTLLMv1.1 (default), "
            "RTLLMv2.0 -> RTLLM2.0"
        ),
    )
    parser.add_argument("--trace-root", default="pipeline_runs_RTLLM", help="Directory for traces")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Optional task id or task name. May be provided multiple times.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only scan and list valid tasks")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of tasks to run in parallel")
    parser.add_argument(
        "--task-max-candidate-workers",
        type=int,
        default=4,
        help="Maximum candidate-generation workers inside one task",
    )
    parser.add_argument(
        "--task-max-prefilter-workers",
        type=int,
        default=4,
        help="Maximum scenario-prefilter workers inside one task",
    )
    parser.add_argument("--verification-timeout-sec", type=int, default=600, help="Timeout per compile/run step")
    parser.add_argument("--strict-scan", action="store_true", help="Fail if any task layout is invalid")
    parser.add_argument("--env-file", default=".env", help="Path to local API key env file")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="openai",
        help="LLM provider to use for all pipeline stages",
    )
    parser.add_argument(
        "--variant",
        default="requirements_constraint_selfplanning",
        choices=list(SUPPORTED_PIPELINE_VARIANTS),
        help="Pipeline branch to run",
    )
    parser.add_argument("--allow-stub-fallback", action="store_true", help="Use stub backend if keys are missing")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = default_config(dataset_root=args.dataset_root, trace_root=args.trace_root)
    config.max_workers = max(1, args.max_workers)
    config.task_max_candidate_workers = max(1, args.task_max_candidate_workers)
    config.task_max_prefilter_workers = max(1, args.task_max_prefilter_workers)
    config.verification_timeout_sec = args.verification_timeout_sec
    config.fail_fast_on_invalid_task = args.strict_scan
    if args.list_only:
        tasks = discover_tasks(config)
        for task in tasks:
            print(
                f"{task.task_id}\tspec={task.spec_path.name}\tscan=ok\t"
                f"testbench={task.testbench_path.name}"
            )
        return

    backends = build_provider_bundle(
        env_path=args.env_file,
        allow_stub_fallback=args.allow_stub_fallback,
        provider=args.provider,
    )

    traces = run_dataset_pipeline(
        config=config,
        backends=backends,
        task_filter=args.task,
        variant=args.variant,
    )
    print("all_tasks_completed")
    _print_pass_summary(traces)
    write_run_summary(config.trace_root, traces)


def _print_pass_summary(traces) -> None:
    summary = build_run_summary(traces)
    variants = summary.get("variants", {})
    for variant in SUPPORTED_PIPELINE_VARIANTS:
        bucket = variants.get(variant)
        if not bucket:
            continue
        print(
            "summary\t"
            f"path={variant}\t"
            f"syntax_passed={bucket['syntax_pass_count']}/{bucket['total_tasks']}\t"
            f"syntax_pass_rate={bucket['syntax_pass_rate']:.4f}\t"
            f"functional_passed={bucket['functional_pass_count']}/{bucket['total_tasks']}\t"
            f"functional_pass_rate={bucket['functional_pass_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
