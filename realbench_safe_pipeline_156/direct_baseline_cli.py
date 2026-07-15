"""Temporary CLI runner for `_156` spec-direct baseline experiments."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import ProviderBundle, StubBackend, build_provider_bundle, load_dotenv_file
from .pipeline import (
    default_config,
    discover_tasks,
    run_task_pipeline_direct_few_shots,
    run_task_pipeline_selfplanning_only,
    run_task_pipeline_spec_direct_baseline,
)
from .runtime_monitor import create_and_install_run_monitor, set_active_run_monitor
from .trace import write_run_summary, write_task_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary direct-generation runner for realbench_safe_pipeline_156"
    )
    parser.add_argument("--dataset-root", default="data_Evalhuman", help="Dataset root to scan")
    parser.add_argument(
        "--trace-root",
        default="pipeline_runs_156_direct_temp_o3mini_medium",
        help="Directory for traces",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Optional task id or task name. May be provided multiple times.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only scan and list valid tasks")
    parser.add_argument("--max-workers", type=int, default=5, help="Number of tasks to run in parallel")
    parser.add_argument("--verification-timeout-sec", type=int, default=1000, help="Timeout per compile/run step")
    parser.add_argument("--strict-scan", action="store_true", help="Fail if any task layout is invalid")
    parser.add_argument("--env-file", default=".env", help="Path to local API key env file")
    parser.add_argument(
        "--model",
        default="o3-mini",
        help="OpenAI model used by the temporary direct baseline runner",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort passed to the OpenAI backend",
    )
    parser.add_argument(
        "--variant",
        default="spec_direct_baseline",
        choices=["spec_direct_baseline", "direct_few_shots", "selfplanning_only"],
        help="Single-call generation branch to run",
    )
    return parser


def _build_unused_stub_bundle() -> ProviderBundle:
    stub = StubBackend(name="direct-baseline-runner-unused-stub")
    return ProviderBundle(
        planner_backend=stub,
        answerer_backend=stub,
        reviewer_backend=stub,
        generator_backend=stub,
        repair_backend=stub,
    )


def _build_backends_for_variant(*, variant: str, env_file: str) -> ProviderBundle:
    if variant == "selfplanning_only":
        return build_provider_bundle(
            env_path=env_file,
            allow_stub_fallback=False,
            provider="openai",
        )
    return _build_unused_stub_bundle()


def _write_run_errors(trace_root: Path, run_errors: list[dict[str, str]]) -> Path | None:
    output_path = trace_root / "run_errors.json"
    if not run_errors:
        if output_path.exists():
            output_path.unlink()
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run_errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def run_dataset_pipeline_direct_variant(
    *,
    dataset_root: str | Path,
    trace_root: str | Path,
    task_filter: str | list[str] | None,
    max_workers: int,
    verification_timeout_sec: int,
    strict_scan: bool,
    variant: str,
    env_file: str,
) -> list:
    run_started_at_epoch = time.time()
    run_started_at_perf = time.perf_counter()
    config = default_config(dataset_root=dataset_root, trace_root=trace_root)
    config.max_workers = max(1, max_workers)
    config.verification_timeout_sec = verification_timeout_sec
    config.fail_fast_on_invalid_task = strict_scan

    monitor = create_and_install_run_monitor(config.trace_root)
    monitor.update(
        variant=variant,
        phase="discover_tasks",
        details={},
    )

    tasks = discover_tasks(config)
    allowed_tasks: set[str] | None = None
    if task_filter:
        if isinstance(task_filter, str):
            allowed_tasks = {task_filter}
        else:
            allowed_tasks = {item for item in task_filter if item}

    selected_tasks = [
        task
        for task in tasks
        if not allowed_tasks or task.task_id in allowed_tasks or task.task_name in allowed_tasks
    ]

    if not selected_tasks:
        write_task_index(config.trace_root, [])
        finished_at_epoch = time.time()
        write_run_summary(
            config.trace_root,
            [],
            run_timing={
                "started_at_epoch": run_started_at_epoch,
                "finished_at_epoch": finished_at_epoch,
                "wall_clock_elapsed_seconds": max(0.0, time.perf_counter() - run_started_at_perf),
            },
        )
        monitor.close(exit_status="normal_exit")
        set_active_run_monitor(None)
        return []

    write_task_index(config.trace_root, selected_tasks)
    monitor.update(
        variant=variant,
        phase="task_index_written",
        details={"selected_task_count": len(selected_tasks)},
    )

    backends = _build_backends_for_variant(variant=variant, env_file=env_file)
    run_errors: list[dict[str, str]] = []
    if variant == "spec_direct_baseline":
        task_runner = run_task_pipeline_spec_direct_baseline
    elif variant == "direct_few_shots":
        task_runner = run_task_pipeline_direct_few_shots
    else:
        task_runner = run_task_pipeline_selfplanning_only

    def _run_one(task):
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant=variant,
            phase="task_start",
            details={},
        )
        return task_runner(
            task=task,
            config=config,
            backends=backends,
        )

    indexed_results: list[tuple[int, object]] = []
    if config.max_workers <= 1 or len(selected_tasks) == 1:
        for index, task in enumerate(selected_tasks):
            try:
                trace = _run_one(task)
                indexed_results.append((index, trace))
                monitor.mark_task_complete()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                run_errors.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.task_name,
                        "branch_variant": variant,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                monitor.update(
                    task_name=task.task_name,
                    task_id=task.task_id,
                    variant=variant,
                    phase="task_exception",
                    details={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_index = {
                executor.submit(_run_one, task): index
                for index, task in enumerate(selected_tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                task = selected_tasks[index]
                try:
                    trace = future.result()
                    indexed_results.append((index, trace))
                    monitor.mark_task_complete()
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    run_errors.append(
                        {
                            "task_id": task.task_id,
                            "task_name": task.task_name,
                            "branch_variant": variant,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    monitor.update(
                        task_name=task.task_name,
                        task_id=task.task_id,
                        variant=variant,
                        phase="task_exception",
                        details={
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )

    indexed_results.sort(key=lambda item: item[0])
    traces = [trace for _, trace in indexed_results]
    _write_run_errors(config.trace_root, run_errors)
    finished_at_epoch = time.time()
    write_run_summary(
        config.trace_root,
        traces,
        run_timing={
            "started_at_epoch": run_started_at_epoch,
            "finished_at_epoch": finished_at_epoch,
            "wall_clock_elapsed_seconds": max(0.0, time.perf_counter() - run_started_at_perf),
        },
    )
    monitor.close(exit_status="normal_exit")
    set_active_run_monitor(None)
    return traces


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv_file(env_path=args.env_file, override=True)
    os.environ["OPENAI_DIRECT_MODEL"] = args.model
    os.environ["OPENAI_DIRECT_REASONING_EFFORT"] = args.reasoning_effort
    os.environ["OPENAI_MODEL"] = args.model
    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    config = default_config(dataset_root=args.dataset_root, trace_root=args.trace_root)
    config.fail_fast_on_invalid_task = args.strict_scan
    if args.list_only:
        tasks = discover_tasks(config)
        for task in tasks:
            print(
                f"{task.task_id}\tspec={task.spec_path.name}\tverification=ok\t"
                f"testbench={task.testbench_path.name}"
            )
        return

    traces = run_dataset_pipeline_direct_variant(
        dataset_root=args.dataset_root,
        trace_root=args.trace_root,
        task_filter=args.task,
        max_workers=args.max_workers,
        verification_timeout_sec=args.verification_timeout_sec,
        strict_scan=args.strict_scan,
        variant=args.variant,
        env_file=args.env_file,
    )
    print("all_tasks_completed")
    print(f"trace_root={Path(args.trace_root).resolve()}")
    print(f"total_traces={len(traces)}")


if __name__ == "__main__":
    main()
