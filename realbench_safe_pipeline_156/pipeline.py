"""High-level Evalhuman pipeline orchestration."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import OpenAIResponsesBackend, ProviderBundle
from .models import CandidateVerilog, PipelineConfig, PipelineTrace, SpecUnderstanding, TaskSpec, VerificationResult
from .requirements_constraint_selfplanning import (
    RequirementsConstraintSelfPlanningTaskBase,
    build_task_base as build_requirements_constraint_selfplanning_task_base,
)
from .reasoning import (
    generate_candidate_from_spec_few_shots,
    generate_candidate_from_spec_only,
    generate_candidate_from_spec_selfplanning_style,
    generate_candidate_from_spec_with_constraints,
)
from .runtime_monitor import create_and_install_run_monitor, get_active_run_monitor, set_active_run_monitor
from .spec_reader import read_spec_text
from .trace import write_run_summary, write_task_index, write_trace
from .verifier import run_compile_only, run_verification


class TaskScanError(RuntimeError):
    """Raised when a task directory does not match the expected Evalhuman layout."""


SUPPORTED_PIPELINE_VARIANTS = (
    "requirements_constraint_selfplanning",
    "spec_direct_baseline",
    "selfplanning_only",
    "direct_few_shots",
    "direct_with_constraint",
)


def scan_tasks(dataset_root: Path, strict: bool = False) -> list[TaskSpec]:
    """Scan Evalhuman tasks from ``data_Evalhuman/<task>/`` folders."""

    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    tasks: list[TaskSpec] = []
    errors: list[str] = []
    for task_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        try:
            tasks.append(_scan_one_task(task_dir))
        except TaskScanError as exc:
            message = f"{task_dir}: {exc}"
            if strict:
                raise
            errors.append(message)

    if strict and errors:
        raise TaskScanError("\n".join(errors))
    return tasks


def _scan_one_task(task_dir: Path) -> TaskSpec:
    spec_path = task_dir / "design_description.txt"
    testbench_path = task_dir / "testbench.sv"
    if not spec_path.is_file():
        raise TaskScanError("missing design_description.txt")
    if not testbench_path.is_file():
        raise TaskScanError("missing testbench.sv")

    task_name = task_dir.name
    return TaskSpec(
        task_id=task_name,
        task_level="task",
        system_name="",
        task_name=task_name,
        task_dir=task_dir.resolve(),
        spec_path=spec_path.resolve(),
        testbench_path=testbench_path.resolve(),
    )


def _write_timed_trace(config: PipelineConfig, trace: PipelineTrace, started_at: float) -> PipelineTrace:
    trace.elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    write_trace(config.trace_root, trace)
    return trace


def _persist_passed_constraints(task: TaskSpec, trace: PipelineTrace) -> Path | None:
    """Persist final disambiguation constraints back into the source task folder.

    This is only used for the active requirements->constraints branch so the
    saved constraints can be reused later as migration/evaluation material.
    """

    if trace.pipeline_variant != "requirements_constraint_selfplanning":
        return None

    raw_constraints = trace.constraint_trace.get("final_disambiguation_constraints", [])
    if not isinstance(raw_constraints, list):
        return None

    cleaned_constraints: list[str] = []
    seen: set[str] = set()
    for item in raw_constraints:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned_constraints.append(text)

    if not cleaned_constraints:
        return None

    output_path = task.task_dir / "constraint.txt"
    lines = [f"{index}. {constraint}" for index, constraint in enumerate(cleaned_constraints, start=1)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_constraint_file(task: TaskSpec) -> list[str]:
    """Read `constraint.txt` beside one task and normalize it into a list of constraint strings."""

    path = task.task_dir / "constraint.txt"
    if not path.exists():
        return []
    constraints: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "." in line:
            prefix, suffix = line.split(".", 1)
            if prefix.strip().isdigit() and suffix.strip():
                line = suffix.strip()
        if line:
            constraints.append(line)
    return constraints


def _build_direct_branches_backend() -> OpenAIResponsesBackend:
    """Create the dedicated backend used by the direct comparison branches."""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the direct comparison branches.")
    model = os.environ.get("OPENAI_DIRECT_MODEL", "").strip() or "o3-mini"
    reasoning_effort = (
        os.environ.get("OPENAI_DIRECT_REASONING_EFFORT", "").strip().lower() or "medium"
    )
    return OpenAIResponsesBackend(
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
    )


def discover_tasks(config: PipelineConfig) -> list[TaskSpec]:
    """Scan tasks and return the discovered task list."""

    tasks = scan_tasks(
        config.dataset_root,
        strict=config.fail_fast_on_invalid_task,
    )
    return tasks


def run_task_pipeline_selfplanning_only(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
) -> PipelineTrace:
    """Execute a self-planning-style direct generation baseline."""

    started_at = time.perf_counter()
    spec_text = read_spec_text(task)
    direct_backend = _build_direct_branches_backend()
    candidate, generation_record = generate_candidate_from_spec_selfplanning_style(
        task=task,
        spec_text=spec_text,
        backend=direct_backend,
    )
    candidates, compile_results, current_candidate, repair_io = _run_compile_only_sequence(
        task=task,
        initial_candidate=candidate,
        config=config,
    )
    verification_results: list[VerificationResult] = []
    if current_candidate is not None:
        verification_results.append(
            run_verification(
                task=task,
                candidate=current_candidate,
                timeout_sec=config.verification_timeout_sec,
            )
        )

    trace = PipelineTrace(
        pipeline_variant="selfplanning_only",
        task=task,
        understanding=SpecUnderstanding(),
        baseline_candidate=None,
        baseline_verification_result=None,
        candidates=candidates,
        compile_results=compile_results,
        verification_results=verification_results,
        constraint_trace={},
        verilog_selfcheck_trace={},
        llm_io={
            "generation": generation_record,
            "selfplanning_only": {
                "mode": "single_call_selfplanning_style_direct_generation",
                "generator_backend": f"OpenAI:{direct_backend.model}",
                "reasoning_effort": direct_backend.reasoning_effort,
            },
            "compile_repairs": repair_io,
        },
    )
    trace.elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    write_trace(config.trace_root, trace)
    return trace


def run_task_pipeline_spec_direct_baseline(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
) -> PipelineTrace:
    """Execute a pure spec-only Verilog generation baseline."""

    started_at = time.perf_counter()
    spec_text = read_spec_text(task)
    direct_backend = _build_direct_branches_backend()
    candidate, generation_record = generate_candidate_from_spec_only(
        task=task,
        spec_text=spec_text,
        backend=direct_backend,
    )
    candidates, compile_results, current_candidate, repair_io = _run_compile_only_sequence(
        task=task,
        initial_candidate=candidate,
        config=config,
    )
    verification_results: list[VerificationResult] = []
    if current_candidate is not None:
        verification_results.append(
            run_verification(
                task=task,
                candidate=current_candidate,
                timeout_sec=config.verification_timeout_sec,
            )
        )

    trace = PipelineTrace(
        pipeline_variant="spec_direct_baseline",
        task=task,
        understanding=SpecUnderstanding(),
        baseline_candidate=None,
        baseline_verification_result=None,
        candidates=candidates,
        compile_results=compile_results,
        verification_results=verification_results,
        constraint_trace={},
        verilog_selfcheck_trace={},
        llm_io={
            "generation": generation_record,
            "spec_direct_baseline": {
                "generator_backend": f"OpenAI:{direct_backend.model}",
            },
            "compile_repairs": repair_io,
        },
    )
    trace.elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    write_trace(config.trace_root, trace)
    return trace


def run_task_pipeline_direct_few_shots(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
) -> PipelineTrace:
    """Execute a direct spec-only Verilog generation baseline with two few-shot examples."""

    started_at = time.perf_counter()
    spec_text = read_spec_text(task)
    direct_backend = _build_direct_branches_backend()
    candidate, generation_record = generate_candidate_from_spec_few_shots(
        task=task,
        spec_text=spec_text,
        backend=direct_backend,
    )
    candidates, compile_results, current_candidate, repair_io = _run_compile_only_sequence(
        task=task,
        initial_candidate=candidate,
        config=config,
    )
    verification_results: list[VerificationResult] = []
    if current_candidate is not None:
        verification_results.append(
            run_verification(
                task=task,
                candidate=current_candidate,
                timeout_sec=config.verification_timeout_sec,
            )
        )

    trace = PipelineTrace(
        pipeline_variant="direct_few_shots",
        task=task,
        understanding=SpecUnderstanding(),
        baseline_candidate=None,
        baseline_verification_result=None,
        candidates=candidates,
        compile_results=compile_results,
        verification_results=verification_results,
        constraint_trace={},
        verilog_selfcheck_trace={},
        llm_io={
            "generation": generation_record,
            "direct_few_shots": {
                "generator_backend": f"OpenAI:{direct_backend.model}",
                "few_shot_example_count": 2,
            },
            "compile_repairs": repair_io,
        },
    )
    trace.elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    write_trace(config.trace_root, trace)
    return trace


def run_task_pipeline_direct_with_constraint(
    task: TaskSpec,
    config: PipelineConfig,
) -> PipelineTrace:
    """Execute a direct spec-driven branch that additionally injects external saved constraints."""

    started_at = time.perf_counter()
    spec_text = read_spec_text(task)
    constraints = _read_constraint_file(task)
    backend = _build_direct_branches_backend()
    candidate, generation_record = generate_candidate_from_spec_with_constraints(
        task=task,
        spec_text=spec_text,
        constraints=constraints,
        backend=backend,
    )
    candidates, compile_results, current_candidate, repair_io = _run_compile_only_sequence(
        task=task,
        initial_candidate=candidate,
        config=config,
    )
    verification_results: list[VerificationResult] = []
    if current_candidate is not None:
        verification_results.append(
            run_verification(
                task=task,
                candidate=current_candidate,
                timeout_sec=config.verification_timeout_sec,
            )
        )

    trace = PipelineTrace(
        pipeline_variant="direct_with_constraint",
        task=task,
        understanding=SpecUnderstanding(),
        baseline_candidate=None,
        baseline_verification_result=None,
        candidates=candidates,
        compile_results=compile_results,
        verification_results=verification_results,
        constraint_trace={
            "external_constraints": constraints,
        },
        verilog_selfcheck_trace={},
        llm_io={
            "generation": generation_record,
            "direct_with_constraint": {
                "constraint_file_path": str(task.task_dir / "constraint.txt"),
                "constraints": constraints,
                "generator_backend": f"OpenAI:{backend.model}",
            },
            "compile_repairs": repair_io,
        },
    )
    trace.elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    write_trace(config.trace_root, trace)
    return trace


def run_task_pipeline_requirements_constraint_selfplanning(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
    layered_base: RequirementsConstraintSelfPlanningTaskBase | None = None,
) -> PipelineTrace:
    """Execute the independent requirements -> constraints branch."""

    base = layered_base or build_requirements_constraint_selfplanning_task_base(
        task=task,
        config=config,
        backends=backends,
    )

    trace = PipelineTrace(
        pipeline_variant="requirements_constraint_selfplanning",
        task=task,
        understanding=base.understanding,
        baseline_candidate=None,
        baseline_verification_result=None,
        candidates=list(base.candidates),
        compile_results=list(base.compile_results),
        verification_results=list(base.verification_results),
        constraint_trace=copy.deepcopy(base.constraint_trace),
        verilog_selfcheck_trace={},
        llm_io=copy.deepcopy(base.llm_io),
    )
    trace.elapsed_seconds = base.elapsed_seconds
    write_trace(config.trace_root, trace)
    _persist_passed_constraints(task, trace)
    return trace


def run_dataset_pipeline(
    config: PipelineConfig,
    backends: ProviderBundle,
    task_filter: str | list[str] | None = None,
    variant: str = "requirements_constraint_selfplanning",
) -> list[PipelineTrace]:
    """Run the pipeline over all scanned tasks or one selected task."""

    if variant not in SUPPORTED_PIPELINE_VARIANTS:
        raise RuntimeError(
            f"Unsupported pipeline variant '{variant}'. Expected one of: {', '.join(SUPPORTED_PIPELINE_VARIANTS)}."
        )

    run_started_at_epoch = time.time()
    run_started_at_perf = time.perf_counter()
    monitor = create_and_install_run_monitor(config.trace_root)
    monitor.update(variant=variant, phase="discover_tasks", details={})

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

    run_errors: list[dict[str, str]] = []

    if config.max_workers <= 1 or len(selected_tasks) == 1:
        traces: list[PipelineTrace] = []
        for task in selected_tasks:
            try:
                monitor.update(
                    task_name=task.task_name,
                    task_id=task.task_id,
                    variant=variant,
                    phase="task_start",
                    details={},
                )
                task_traces, task_errors = _run_variant_for_task_with_retry(
                    task=task,
                    config=config,
                    backends=backends,
                    variant=variant,
                )
                traces.extend(task_traces)
                run_errors.extend(task_errors)
                monitor.mark_task_complete()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                run_errors.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.task_name,
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

    indexed_results: list[tuple[int, list[PipelineTrace]]] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_index = {
            executor.submit(_run_variant_for_task_with_retry, task, config, backends, variant): index
            for index, task in enumerate(selected_tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = selected_tasks[index]
            try:
                task_traces, task_errors = future.result()
                indexed_results.append((index, task_traces))
                run_errors.extend(task_errors)
                monitor.mark_task_complete()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                run_errors.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.task_name,
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
    traces: list[PipelineTrace] = []
    for _, task_traces in indexed_results:
        traces.extend(task_traces)
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


def default_config(
    dataset_root: str | Path = "data_Evalhuman",
    trace_root: str | Path = "pipeline_runs_156",
) -> PipelineConfig:
    """Construct a default runtime config with project-relative paths."""

    return PipelineConfig(
        dataset_root=Path(dataset_root).resolve(),
        trace_root=Path(trace_root).resolve(),
    )


def _run_variants_for_task(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
) -> tuple[list[PipelineTrace], list[dict[str, str]]]:
    traces: list[PipelineTrace] = []
    branch_errors: list[dict[str, str]] = []

    try:
        monitor = get_active_run_monitor()
        if monitor is not None:
            monitor.update(
                task_name=task.task_name,
                task_id=task.task_id,
                variant="requirements_constraint_selfplanning",
                phase="build_task_base_start",
                details={},
            )
        layered_base = build_requirements_constraint_selfplanning_task_base(
            task=task,
            config=config,
            backends=backends,
        )
        if monitor is not None:
            monitor.update(
                task_name=task.task_name,
                task_id=task.task_id,
                variant="requirements_constraint_selfplanning",
                phase="write_trace_start",
                details={},
            )
        traces.append(
            run_task_pipeline_requirements_constraint_selfplanning(
                task=task,
                config=config,
                backends=backends,
                layered_base=layered_base,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        branch_errors.append(_build_branch_error(task, "requirements_constraint_selfplanning", exc))

    return traces, branch_errors


def _run_single_variant_for_task(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
    variant: str,
) -> tuple[list[PipelineTrace], list[dict[str, str]]]:
    try:
        if variant == "spec_direct_baseline":
            return [run_task_pipeline_spec_direct_baseline(task=task, config=config, backends=backends)], []
        if variant == "selfplanning_only":
            return [run_task_pipeline_selfplanning_only(task=task, config=config, backends=backends)], []
        if variant == "direct_few_shots":
            return [run_task_pipeline_direct_few_shots(task=task, config=config, backends=backends)], []
        if variant == "direct_with_constraint":
            return [run_task_pipeline_direct_with_constraint(task=task, config=config)], []
        if variant == "requirements_constraint_selfplanning":
            return _run_variants_for_task(task=task, config=config, backends=backends)
        raise RuntimeError(f"Unsupported pipeline variant '{variant}'.")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return [], [_build_branch_error(task, variant, exc)]


def _variant_task_output_dir(config: PipelineConfig, task: TaskSpec, variant: str) -> Path:
    return config.trace_root / variant / task.task_name


def _run_variant_for_task_with_retry(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
    variant: str,
) -> tuple[list[PipelineTrace], list[dict[str, str]]]:
    if variant == "requirements_constraint_selfplanning":
        return _run_variants_for_task_with_retry(task=task, config=config, backends=backends)

    last_traces: list[PipelineTrace] = []
    attempts = max(0, config.max_task_retries) + 1
    task_output_dir = _variant_task_output_dir(config, task, variant)

    for attempt in range(1, attempts + 1):
        try:
            if task_output_dir.exists():
                shutil.rmtree(task_output_dir)
            traces, branch_errors = _run_single_variant_for_task(
                task=task,
                config=config,
                backends=backends,
                variant=variant,
            )
            last_traces = traces
            if not branch_errors:
                return traces, []

            current_errors = []
            for item in branch_errors:
                copied = dict(item)
                copied["attempt"] = attempt
                current_errors.append(copied)
            if attempt == attempts:
                return traces, current_errors
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            current_error = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "attempt": attempt,
                "branch_variant": variant,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if attempt == attempts:
                return last_traces, [current_error]

    return last_traces, []


def _run_variants_for_task_with_retry(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
) -> tuple[list[PipelineTrace], list[dict[str, str]]]:
    last_traces: list[PipelineTrace] = []
    attempts = max(0, config.max_task_retries) + 1
    task_output_dir = (
        config.trace_root / "requirements_constraint_selfplanning" / task.task_name
    )

    for attempt in range(1, attempts + 1):
        try:
            if task_output_dir.exists():
                shutil.rmtree(task_output_dir)
            traces, branch_errors = _run_variants_for_task(task=task, config=config, backends=backends)
            last_traces = traces
            if not branch_errors:
                return traces, []

            current_errors = []
            for item in branch_errors:
                copied = dict(item)
                copied["attempt"] = attempt
                current_errors.append(copied)
            if attempt == attempts:
                return traces, current_errors
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            current_error = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if attempt == attempts:
                return last_traces, [current_error]

    return last_traces, []


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


def _build_branch_error(task: TaskSpec, branch_variant: str, exc: Exception) -> dict[str, str]:
    return {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "branch_variant": branch_variant,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _run_compile_only_sequence(
    task: TaskSpec,
    initial_candidate,
    config: PipelineConfig,
):
    candidates = [initial_candidate]
    compile_results = []
    current_candidate = initial_candidate
    repair_io: dict[str, object] = {
        "disabled": True,
        "disabled_reason": "compile_repairs_disabled_in_active_pipeline_variants",
        "rounds": [],
    }

    result = run_compile_only(
        task=task,
        candidate=current_candidate,
        timeout_sec=config.verification_timeout_sec,
    )
    compile_results.append(result)

    return candidates, compile_results, current_candidate, repair_io
