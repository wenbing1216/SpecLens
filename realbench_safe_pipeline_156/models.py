"""Core dataclasses used by the safe Verilog-generation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    """Metadata collected from one Evalhuman task folder."""

    task_id: str
    task_level: str
    system_name: str
    task_name: str
    task_dir: Path
    spec_path: Path
    testbench_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_dir"] = str(self.task_dir)
        data["spec_path"] = str(self.spec_path)
        data["testbench_path"] = str(self.testbench_path)
        return data


@dataclass(slots=True)
class SpecUnderstanding:
    """Structured understanding extracted from the spec text."""

    explicit_requirements: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    self_planning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanChallenge:
    """One spec-grounded challenge against a current implementation plan step."""

    target_step: str
    why_it_conflicts_with_spec: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChallengeJudgment:
    """Planner judgment on one challenger item."""

    target_step: str
    verdict: str
    reason: str
    revision_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanRound:
    """One challenge/refine round for the implementation plan."""

    round_index: int
    plan_before: list[str] = field(default_factory=list)
    challenges: list[PlanChallenge] = field(default_factory=list)
    judgments: list[ChallengeJudgment] = field(default_factory=list)
    plan_after: list[str] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "plan_before": self.plan_before,
            "challenges": [item.to_dict() for item in self.challenges],
            "judgments": [item.to_dict() for item in self.judgments],
            "plan_after": self.plan_after,
            "stop_reason": self.stop_reason,
        }


@dataclass(slots=True)
class PlanRefinementTrace:
    """Initial plan plus iterative challenge/refine rounds."""

    initial_plan: list[str] = field(default_factory=list)
    rounds: list[PlanRound] = field(default_factory=list)
    final_plan: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_plan": self.initial_plan,
            "rounds": [item.to_dict() for item in self.rounds],
            "final_plan": self.final_plan,
        }


@dataclass(slots=True)
class CandidateVerilog:
    """One generated candidate and its provenance."""

    candidate_id: str
    raw_response: str
    clean_verilog: str
    module_name: str | None = None
    generation_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "module_name": self.module_name,
            "generation_notes": self.generation_notes,
        }


@dataclass(slots=True)
class VerificationResult:
    """Result of running `make compile` and `make run` in a temp workspace."""

    candidate_id: str
    workspace_dir: str
    compile_returncode: int | None
    run_returncode: int | None
    compile_stdout: str
    compile_stderr: str
    run_stdout: str
    run_stderr: str
    passed: bool
    compile_summary: str
    evaluation_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineTrace:
    """Per-task trace capturing all intermediate artifacts."""

    pipeline_variant: str
    task: TaskSpec
    understanding: SpecUnderstanding
    candidates: list[CandidateVerilog]
    verification_results: list[VerificationResult]
    compile_results: list[VerificationResult] = field(default_factory=list)
    baseline_candidate: CandidateVerilog | None = None
    baseline_verification_result: VerificationResult | None = None
    second_compile_candidates: list[CandidateVerilog] = field(default_factory=list)
    second_compile_results: list[VerificationResult] = field(default_factory=list)
    constraint_trace: dict[str, Any] = field(default_factory=dict)
    verilog_selfcheck_trace: dict[str, Any] = field(default_factory=dict)
    llm_io: dict[str, Any] = field(default_factory=dict)
    llm_io_files: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_variant": self.pipeline_variant,
            "understanding": self.understanding.to_dict(),
            "baseline_candidate": (
                self.baseline_candidate.to_trace_dict() if self.baseline_candidate is not None else None
            ),
            "baseline_verification_result": (
                self.baseline_verification_result.to_trace_dict()
                if self.baseline_verification_result is not None
                else None
            ),
            "compile_results": [item.to_trace_dict() for item in self.compile_results],
            "second_compile_candidates": [item.to_trace_dict() for item in self.second_compile_candidates],
            "second_compile_results": [item.to_trace_dict() for item in self.second_compile_results],
            "constraint_trace": self.constraint_trace,
            "candidates": [item.to_trace_dict() for item in self.candidates],
            "verification_results": [item.to_trace_dict() for item in self.verification_results],
            "verilog_selfcheck_trace": self.verilog_selfcheck_trace,
            "llm_io_files": self.llm_io_files,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class PipelineConfig:
    """Runtime knobs for the first-pass pipeline implementation."""

    dataset_root: Path
    trace_root: Path
    max_workers: int = 5
    task_max_candidate_workers: int = 4
    task_max_prefilter_workers: int = 4
    max_task_retries: int = 1
    max_compile_repair_rounds: int = 5
    verification_timeout_sec: int = 1000
    fail_fast_on_invalid_task: bool = False
    behavior_to_constraint_run_count: int = 3
    behavior_to_constraint_max_rounds: int = 2
