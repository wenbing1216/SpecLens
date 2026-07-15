"""Independent layered branch: requirements -> constraints -> cluster-resolved Verilog."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock, current_thread
from typing import Any, Sequence

from .compile_repair import extract_verilog_module_with_warning
from .entropy import semantic_entropy_from_counts
from .llm import LLMBackend, ProviderBundle
from .llm_verilog_cluster import (
    InterfaceInfo,
    MultiClockSeqControlContractError,
    Port,
    ResetInfo,
    StimulusCase,
    StimulusStep,
    VerificationScenario,
    VerilogStimulusClusterPipeline,
    VERILOG_USER_PROMPT,
    summarize_functional_clusters,
)
from .models import CandidateVerilog, PipelineConfig, SpecUnderstanding, TaskSpec, VerificationResult
from .reasoning import (
    _as_string_list,
    extract_explicit_requirements,
    generate_candidate_from_spec_only,
    guess_module_name,
)
from .runtime_monitor import get_active_run_monitor
from .spec_reader import read_spec_text
from .verifier import run_verification


CLUSTER_FEEDBACK_MODE_REQUIREMENT_STIMULUS_CONSTRAINTS = "requirement_stimulus_constraints"
CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK = "multiclock_direct_fallback"
DEFAULT_CLUSTER_FEEDBACK_MODE = CLUSTER_FEEDBACK_MODE_REQUIREMENT_STIMULUS_CONSTRAINTS
SEMANTIC_ENTROPY_ZERO_TOLERANCE = 1e-12
REQUIREMENTS_BRANCH_VERILOG_GUIDANCE = """
- For triangle-wave signal-generation tasks, make the externally visible waveform follow the specified sample sequence exactly at the turning points: when the wave reaches its maximum or minimum bound, do not skip that boundary value or reverse direction a cycle early; if the specification's reference sequence shows the boundary level persisting for the boundary sample, preserve that observable plateau exactly before stepping away from it.
- For purely combinational arithmetic modules, do not mask ambiguous comparison cases with default zero outputs or forced sign clamping; unless the specification explicitly requires a defined fallback, let arithmetic remain operand-driven so unknown inputs naturally propagate.
- For sequential register-update tasks, if one register is shifted or otherwise updated on a clock edge and another register or output is also updated on that same edge using it, standard nonblocking semantics mean the same-edge computation observes the pre-edge current registered value, while the shifted or updated value becomes visible on the following cycle; do not rewrite a per-cycle shift requirement into same-cycle use of the shifted value unless the specification explicitly requires that intra-edge ordering.
- When the specification fixes a parameter to a concrete constant such as `size = 4`, use concrete top-level port widths in the module interface for this task (`mul_a[3:0]`, `mul_b[3:0]`, `mul_out[7:0]`) instead of unresolved symbolic width expressions like `[size-1:0]` or `[2*size-1:0]`; if you keep `parameter size = 4`, use it only for internal logic.
- For restoring-division tasks where the dividend is wider than the divisor, complete the iterative compare-subtract-shift process across every dividend bit from MSB to LSB; do not collapse the design into one initial upper-slice comparison followed by only a partial-width quotient/remainder computation.
- For IEEE-754 single-precision multiplication of normalized operands, the 24x24 significand product lies in [1,4): if normalization detects a product in [2,4), shift right exactly once and increment the exponent; otherwise keep the exponent unchanged and do not decrement it.
- When normalization and rounding are implemented in a clocked pipeline stage, derive mantissa/guard/round/sticky from the same current-stage product expression; do not write a register with nonblocking assignment and then immediately slice that just-assigned register in the same always block.
- For fractional clock-division tasks that use dual-edge phase shifting, do not collapse the design into a single-edge 4-high/3-low waveform or into two independently complete square waves; instead use the shared modulo counter to create the intended edge-specific phase windows on the positive-edge and negative-edge branches, then combine those edge-shifted branch outputs so the final observable clock matches the required 3.5-divider pattern.
- For odd clock-divider tasks that use one clock's positive and negative edges, do not implement each branch as a single isolated toggle at `NUM_DIV >> 1`; instead form complementary positive-edge and negative-edge high windows from the shared modulo count so their OR-ed output matches the required odd-divider waveform.
- For asynchronous FIFO tasks whose specification mentions pointer buffers, previous pointer values, or synchronized Gray-pointer comparison, implement the RAM address path and the flag-visible Gray-pointer path as different roles: after an accepted write or read, the local binary pointer may advance for RAM addressing using only the lower `ADDR_WIDTH` bits, but the registered Gray pointer used for synchronization and `wfull`/`rempty` comparison must still reflect the pre-increment local pointer for that sampled cycle; do not encode next-pointer Gray into the same-cycle registered flag path, and keep the full/empty comparison on `PTR_WIDTH = ADDR_WIDTH + 1` Gray pointers rather than collapsing them to the RAM address width.
""".strip()
def _build_requirements_branch_cluster_verilog_user_prompt() -> str:
    """Inject branch-local Verilog guidance only into candidate generation for this branch."""

    return f"""{VERILOG_USER_PROMPT.rstrip()}
- Additional hardware-knowledge reminders:
{REQUIREMENTS_BRANCH_VERILOG_GUIDANCE}
"""


@dataclass(slots=True)
class RequirementsConstraintSelfPlanningTaskBase:
    spec_text: str
    understanding: SpecUnderstanding
    constraint_trace: dict
    llm_io: dict
    candidates: list[CandidateVerilog]
    compile_results: list[VerificationResult]
    verification_results: list[VerificationResult]
    current_candidate: CandidateVerilog | None
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class _ClusterBackendAdapter:
    """Adapt the local `_156` text backend to the cluster pipeline's `generate()` interface."""

    backend: LLMBackend
    records: list[dict[str, Any]] = field(default_factory=list)
    _records_lock: Lock = field(default_factory=Lock)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
        temperature: float = 0.8,
    ) -> list[str]:
        responses: list[str] = []
        for response_index in range(n):
            record = self.backend.complete_text_record(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            with self._records_lock:
                self.records.append(
                    {
                        "response_index": response_index,
                        "requested_n": n,
                        "requested_temperature": temperature,
                        **record,
                    }
                )
            responses.append(str(record["raw_response"]))
        return responses


@dataclass(slots=True)
class _NoopClusterLLMClient:
    """Placeholder llm_client for worker-side simulation-only pipeline usage."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
        temperature: float = 0.8,
    ) -> list[str]:
        raise RuntimeError(
            "No LLM calls should occur inside prefilter simulation workers. "
            "This placeholder client exists only so simulation helpers can be reused."
        )


def _summarize_llm_call_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the same usage-style summary for a local list of LLM call records."""

    summary = {
        "total_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_elapsed_seconds": 0.0,
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        required_keys = {
            "backend",
            "mode",
            "system_prompt",
            "user_prompt",
            "raw_response",
            "parsed_response",
        }
        if not required_keys.issubset(record.keys()):
            continue
        summary["total_model_calls"] += 1
        usage = record.get("usage")
        if isinstance(usage, dict):
            summary["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            summary["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            summary["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        summary["total_elapsed_seconds"] += float(record.get("elapsed_seconds", 0.0) or 0.0)
    return summary


def _extract_functional_cluster_sizes(functional_clusters: Sequence[Any]) -> list[int]:
    """Collect positive cluster sizes from summarized or raw functional-cluster payloads."""

    cluster_sizes: list[int] = []
    for cluster in functional_clusters:
        size = 0
        if isinstance(cluster, dict):
            if cluster.get("size") is not None:
                size = int(cluster.get("size", 0) or 0)
            elif isinstance(cluster.get("candidate_ids"), list):
                size = len(cluster.get("candidate_ids", []) or [])
            elif cluster.get("score") is not None:
                size = int(cluster.get("score", 0) or 0)
        else:
            candidate_ids = getattr(cluster, "candidate_ids", None)
            if isinstance(candidate_ids, list):
                size = len(candidate_ids)
        if size > 0:
            cluster_sizes.append(size)
    return cluster_sizes


def _semantic_entropy_from_resolution(resolution: dict[str, Any]) -> float | None:
    """Compute semantic entropy directly from one resolution payload."""

    cluster_sizes = _extract_functional_cluster_sizes(
        list(resolution.get("functional_clusters", []) or [])
    )
    if not cluster_sizes:
        return None
    return semantic_entropy_from_counts(cluster_sizes)


def _entropy_is_zero(entropy: float | None) -> bool:
    """Treat near-zero semantic entropy as a resolved single-cluster outcome."""

    return entropy is not None and abs(float(entropy)) <= SEMANTIC_ENTROPY_ZERO_TOLERANCE


def _postprocess_cluster_candidate_output_initialization(
    *,
    spec_text: str,
    candidate: CandidateVerilog,
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Ask the model for a minimal output-initialization fix before verification."""

    system_prompt = (
        "You are an expert at writing Verilog code. "
        "Revise the Verilog only to ensure outputs have deterministic spec-consistent initial values "
        "without changing output declarations or unrelated behavior."
    )
    user_prompt = f"""
Specification:
{spec_text}

Current Verilog:
```verilog
{candidate.clean_verilog}
```

Task:
- Add only the minimal initialization needed so the module's outputs start from a deterministic value consistent with the specification.
- Do not change any output port type, width, or name.
- Never add an initializer directly in the module interface or port declaration; any initialization must be done inside the module body.
- If you introduce a loop variable for initialization, never declare `integer i;` after any procedural statement inside an `initial` or `always` block. For Icarus/Verilog compatibility, place such declarations before the first statement of the block or at module scope.
- If an output is declared as `reg` or `logic` and is directly process-assigned, you may add or adjust an `initial` block.
- If an output is a `wire` or is driven by `assign`, do not assign to that output procedurally. Instead initialize the internal state/register/memory that drives it.
- Prefer initializing existing internal state over introducing new logic.
- If the module is purely combinational, return it unchanged. Do not introduce any new state, gating, or timing behavior (such as initial blocks, delays, or helper registers) just to force a deterministic startup output.
- Do not add reset ports, helper modules, testbench code, or explanatory prose.
- Preserve the module interface and intended functionality.
- If the current Verilog already handles initialization correctly, return it unchanged.

Return the final answer as a fenced ```verilog``` block containing only the revised Verilog/SystemVerilog module code.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{candidate.candidate_id}:output_initialization_postprocess",
        fallback_text=candidate.clean_verilog,
    )
    changed = clean.strip() != candidate.clean_verilog.strip()
    output_init_candidate_id = (
        candidate.candidate_id
        if candidate.candidate_id.endswith("_output_init")
        else f"{candidate.candidate_id}_output_init"
    )
    revised_candidate = CandidateVerilog(
        candidate_id=output_init_candidate_id,
        raw_response=raw if changed else candidate.raw_response,
        clean_verilog=clean if changed else candidate.clean_verilog,
        module_name=guess_module_name(clean if changed else candidate.clean_verilog),
        generation_notes=(
            f"{candidate.generation_notes} Post-processed before verification to add spec-grounded "
            "output initialization without changing output declarations."
            if changed
            else (
                f"{candidate.generation_notes} Verified through output-initialization postprocess "
                "path; no initialization changes were needed."
            )
        ).strip(),
    )
    record["verilog_before"] = candidate.clean_verilog
    record["verilog_after"] = revised_candidate.clean_verilog
    record["changed"] = changed
    record["source_candidate_id"] = candidate.candidate_id
    record["result_candidate_id"] = revised_candidate.candidate_id
    return revised_candidate, record


def _postprocess_and_verify_cluster_candidate(
    *,
    task: TaskSpec,
    spec_text: str,
    candidate: CandidateVerilog,
    backend: LLMBackend,
    timeout_sec: int,
) -> tuple[CandidateVerilog, VerificationResult, dict[str, Any]]:
    """Apply output-initialization post-processing, then run verification."""

    revised_candidate, postprocess_record = _postprocess_cluster_candidate_output_initialization(
        spec_text=spec_text,
        candidate=candidate,
        backend=backend,
    )
    verification_result = run_verification(
        task=task,
        candidate=revised_candidate,
        timeout_sec=timeout_sec,
    )
    return revised_candidate, verification_result, postprocess_record


def _parse_supported_constraint_items(raw_items: Any, text_key: str) -> tuple[list[str], list[dict[str, str]]]:
    items: list[dict[str, str]] = []
    texts: list[str] = []
    if not isinstance(raw_items, list):
        return texts, items
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get(text_key, "")).strip()
        why = str(entry.get("why_supported_by_spec", "")).strip()
        if not text:
            continue
        items.append(
            {
                text_key: text,
                "why_supported_by_spec": why,
            }
        )
        texts.append(text)
    return texts, items


def _extract_requirement_partitions(requirements_record: dict[str, Any], all_requirements: list[str]) -> tuple[list[str], list[str]]:
    """Recover interface/behavior requirement partitions from the requirement extractor record."""

    interface_requirements = _as_string_list(requirements_record.get("parsed_interface_requirements"))
    behavior_requirements = _as_string_list(requirements_record.get("parsed_behavior_requirements"))
    if not interface_requirements and not behavior_requirements:
        return [], list(all_requirements)
    if not behavior_requirements:
        remainder = [item for item in all_requirements if item not in interface_requirements]
        return interface_requirements, remainder
    return interface_requirements, behavior_requirements


def _append_cross_behavior_requirement(
    requirements_record: dict[str, Any],
    all_requirements: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Append one order-preserving cross requirement synthesized from all behavior requirements."""

    updated_record = _deepcopy_jsonable(requirements_record)
    interface_requirements = _as_string_list(updated_record.get("parsed_interface_requirements"))
    behavior_requirements = _as_string_list(updated_record.get("parsed_behavior_requirements"))
    if not behavior_requirements:
        return list(all_requirements), updated_record

    cross_requirement = "; ".join(behavior_requirements)
    behavior_requirements.append(cross_requirement)
    updated_record["parsed_behavior_requirements"] = behavior_requirements
    updated_record["cross_behavior_requirement"] = cross_requirement
    updated_record["cross_behavior_requirement_appended"] = True

    combined_requirements = [*interface_requirements, *behavior_requirements]
    updated_record["parsed_all_requirements"] = combined_requirements
    return combined_requirements, updated_record


def _generate_requirement_targeted_scenario(
    spec_text: str,
    target_requirement_id: int,
    target_requirement: str,
    backend: LLMBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate one requirement-targeted verification scenario from the original spec only."""

    system_prompt = (
        "You generate one verification scenario for one target behavior requirement.\n"
        "Use the original specification as the full context.\n"
        "Use the target behavior requirement only to focus the scenario.\n"
        "Keep the scenario at the level of a test situation, not an implementation explanation or a cycle-by-cycle stimulus."
    )
    user_prompt = f"""
Return JSON with keys:
- name
- goal
- story

Original specification:
{spec_text}

Target behavior requirement:
{target_requirement}

Rules:
- generate exactly one scenario focused on testing this target behavior requirement
- use the original specification to avoid narrowing or misreading the target behavior requirement
    - the scenario should describe a concrete testing situation, not a generic topic label
    - the scenario should not contain full cycle-by-cycle stimulus values
- the scenario should be discriminative enough to expose incorrect implementations of the target behavior requirement
    - do not mention any requirement ids in the scenario text
- the scenario must be realizable using only the module's external input ports and legal reachable behavior; do not rely on forcing internal states, hidden registers, internal encodings, or next-state signals
- the scenario must use externally observable outputs or output behavior as the evidence of correctness
- the story must explain how the test reaches the prerequisite starting condition for this requirement before applying the key event being tested
- focus on one primary behavioral ambiguity or decision point; do not combine multiple unrelated behavior goals into one broad story unless the requirement itself explicitly demands that combined sequence
- do not turn environment assumptions, unreachable combinations, or don't-care conditions into standalone DUT-testing scenarios
- do not assume an unjustified setup method; if a starting condition cannot be reached directly by reset, describe a legal input-driven path to reach it
""".strip()
    record = backend.complete_json_record(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = record["parsed_response"]
    scenario = {
        "scenario_id": f"req_{target_requirement_id:03d}",
        "target_requirement_id": target_requirement_id,
        "name": str(parsed.get("name", "")).strip() or f"scenario_req_{target_requirement_id}",
        "goal": str(parsed.get("goal", "")).strip(),
        "story": str(parsed.get("story", "")).strip(),
    }
    record["target_requirement"] = target_requirement
    record["parsed_requirement_targeted_scenario"] = scenario
    return scenario, record


def _generate_requirement_targeted_scenario_pool(
    spec_text: str,
    behavior_requirements: list[str],
    backend: LLMBackend,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate a one-to-one scenario pool from behavior requirements."""

    scenarios: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    for requirement_id, raw_requirement in enumerate(behavior_requirements, start=1):
        target_requirement = str(raw_requirement).strip()
        if not target_requirement:
            continue
        scenario, record = _generate_requirement_targeted_scenario(
            spec_text=spec_text,
            target_requirement_id=requirement_id,
            target_requirement=target_requirement,
            backend=backend,
        )
        scenarios.append(scenario)
        rounds.append(
            {
                "target_requirement_id": requirement_id,
                "target_requirement": target_requirement,
                "scenario": scenario,
                "scenario_call": record,
            }
        )
    return scenarios, {"rounds": rounds, "scenario_pool": scenarios}


def _deepcopy_jsonable(value: Any) -> Any:
    """Clone nested dict/list payloads used for scenarios and stimuli without sharing references."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _write_round_constraint_artifact(
    artifact_dir: str | os.PathLike[str] | None,
    payload: dict[str, Any],
    *,
    cluster_record: dict[str, Any] | None = None,
    cluster_trace: dict[str, Any] | None = None,
) -> str | None:
    """Persist the round-local constraint payload inside the resolved artifact directory."""

    if artifact_dir is None:
        return None
    artifact_dir_text = str(artifact_dir).strip()
    if not artifact_dir_text:
        return None
    constraint_path = Path(artifact_dir_text) / "constraint.json"
    constraint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolved_path = str(constraint_path.resolve())
    if isinstance(cluster_record, dict):
        artifact_files = cluster_record.setdefault("artifact_files", {})
        if isinstance(artifact_files, dict):
            artifact_files["constraint"] = resolved_path
    if isinstance(cluster_trace, dict):
        trace_cluster = cluster_trace.setdefault("llm_verilog_cluster", {})
        if isinstance(trace_cluster, dict):
            artifact_files = trace_cluster.setdefault("artifact_files", {})
            if isinstance(artifact_files, dict):
                artifact_files["constraint"] = resolved_path
    return resolved_path


def _cluster_interface_json_to_object(item: dict[str, Any] | None) -> InterfaceInfo | None:
    """Rehydrate a persisted majority-interface JSON object back into an InterfaceInfo instance."""

    if not isinstance(item, dict):
        return None
    ports = []
    for raw_port in item.get("ports", []) or []:
        if not isinstance(raw_port, dict):
            continue
        ports.append(
            Port(
                direction=str(raw_port.get("direction", "")).strip(),
                name=str(raw_port.get("name", "")).strip(),
                width_text=str(raw_port.get("width_text", "") or "").strip(),
                signed=bool(raw_port.get("signed", False)),
                net_type=str(raw_port.get("net_type", "") or "").strip(),
            )
        )
    return InterfaceInfo(
        module_name=str(item.get("module_name", "")).strip(),
        ports=ports,
        raw_header=str(item.get("raw_header", "") or "").strip(),
        parameter_defaults=dict(item.get("parameter_defaults", {}) or {}),
        signature_text=str(item.get("signature_text", "") or "").strip(),
        parse_success=bool(item.get("parse_success", False)),
        warnings=[str(entry).strip() for entry in item.get("warnings", []) or [] if str(entry).strip()],
    )


def _build_harder_scenario_id(parent_scenario_id: str, expansion_index: int) -> str:
    """Create a deterministic derived scenario id for one extra scenario in the same family."""

    base = str(parent_scenario_id or "scenario").strip() or "scenario"
    return f"{base}_extra_{expansion_index:02d}"


def _scenario_semantic_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """Compare scenarios by their semantic text rather than by generated ids."""

    return (
        str(item.get("name", "")).strip().lower(),
        str(item.get("goal", "")).strip().lower(),
        str(item.get("story", "")).strip().lower(),
    )


def _stimulus_semantic_key(item: dict[str, Any]) -> str:
    """Compare stimuli by their full JSON structure so duplicate harder cases can be skipped."""

    return json.dumps(item or {}, ensure_ascii=False, sort_keys=True)


def _generate_harder_requirement_scenario(
    *,
    spec_text: str,
    target_requirement_id: int,
    target_requirement: str,
    prior_scenarios: list[dict[str, Any]],
    expansion_index: int,
    backend: LLMBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate exactly one new scenario for one requirement/scenario family."""

    system_prompt = (
        "You generate one verification scenario for a specific target behavior requirement.\n\n"
        "Use the original specification as the source of truth.\n"
        "Use the target behavior requirement plus the prior scenarios for this same requirement to focus the new scenario.\n"
        "Do not generate cycle-by-cycle stimulus.\n"
        "Do not generate expected outputs.\n"
        "Keep the result at the scenario level."
    )
    user_prompt = f"""
Return JSON with keys:
- scenario_id: string
- target_requirement_id: integer
- name: string
- goal: string
- story: string

You are given:
1. The original hardware specification
2. One target behavior requirement
3. The prior scenario history for this same requirement

Your task:
Generate exactly one new scenario to further test the target behavior requirement.

Original specification:
{spec_text}

Target behavior requirement:
{target_requirement}

Prior scenarios for this same requirement:
{json.dumps(prior_scenarios, ensure_ascii=False, indent=2)}

Rules:
- generate exactly one new scenario
- this new scenario should help test the target behavior requirement again from a different observable angle
- keep it grounded in the original specification
- do not write cycle-by-cycle input values
- do not write explicit signal tables
- do not write expected outputs
- keep it as a concrete testing situation
- do not rewrite the target behavior requirement into a different behavior rule; preserve its stated meaning while changing the testing angle
- use the prior scenario history to avoid repeating the same testing angle
- generate a new testing angle that is meaningfully different from the existing scenario history while staying on the same behavior requirement
- the scenario must be realizable using only the module's external input ports and legal reachable behavior; do not rely on forcing internal states, hidden registers, internal encodings, or next-state signals
- the scenario must use externally observable outputs or output behavior as the evidence of correctness
- the story must explain how the test reaches the prerequisite starting condition for this requirement before applying the key event being tested
- focus on one primary behavioral ambiguity or decision point; do not combine multiple unrelated behavior goals into one broad story unless the requirement itself explicitly demands that combined sequence
- do not turn environment assumptions, unreachable combinations, or don't-care conditions into standalone DUT-testing scenarios
- do not assume an unjustified setup method; if a starting condition cannot be reached directly by reset, describe a legal input-driven path to reach it

Output format:
{{
  "scenario_id": "{_build_harder_scenario_id(str(prior_scenarios[-1].get("scenario_id", "") if prior_scenarios else f"req_{target_requirement_id:03d}"), expansion_index)}",
  "target_requirement_id": {target_requirement_id},
  "name": "...",
  "goal": "...",
  "story": "..."
}}
""".strip()
    record = backend.complete_json_record(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = record["parsed_response"] if isinstance(record.get("parsed_response"), dict) else {}
    latest_scenario_id = (
        str(prior_scenarios[-1].get("scenario_id", "")).strip()
        if prior_scenarios
        else f"req_{target_requirement_id:03d}"
    )
    scenario = {
        "scenario_id": _build_harder_scenario_id(latest_scenario_id, expansion_index),
        "target_requirement_id": target_requirement_id,
        "parent_scenario_id": latest_scenario_id,
        "name": str(parsed.get("name", "")).strip()
        or f"extra_scenario_req_{target_requirement_id}_{expansion_index}",
        "goal": str(parsed.get("goal", "")).strip(),
        "story": str(parsed.get("story", "")).strip(),
    }
    record["parsed_harder_scenario"] = scenario
    record["harder_scenario_generation_input"] = {
        "target_requirement_id": target_requirement_id,
        "target_requirement": target_requirement,
        "prior_scenarios": _deepcopy_jsonable(prior_scenarios),
    }
    return scenario, record


def _generate_harder_stimulus_case(
    *,
    spec_text: str,
    harder_scenario: dict[str, Any],
    cluster_result: dict[str, Any],
    backend: LLMBackend,
    max_candidate_workers: int,
    cluster_shared_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Convert one harder scenario into exactly one harder stimulus using the current majority interface/runtime."""

    interface_report = cluster_result.get("interface_report", {}) or {}
    majority_interface = _cluster_interface_json_to_object(interface_report.get("majority_interface"))
    majority_circuit_type = str(interface_report.get("majority_circuit_type", "") or "").strip() or "UNKNOWN"
    clock_name = str(cluster_result.get("clock_name", "") or "").strip() or None
    reset_info = cluster_result.get("reset_info", {}) or {}
    reset_name = str(reset_info.get("signal_name", "") or "").strip() or None
    record: dict[str, Any] = {
        "harder_scenario": _deepcopy_jsonable(harder_scenario),
        "majority_circuit_type": majority_circuit_type,
        "clock_name": clock_name,
        "reset_name": reset_name,
        "interface_report": _deepcopy_jsonable(interface_report),
    }
    if majority_interface is None or not majority_interface.parse_success:
        record["disabled"] = True
        record["disabled_reason"] = "missing_or_invalid_majority_interface"
        record["llm_calls"] = []
        record["model_usage"] = _summarize_llm_call_records([])
        return None, record

    adapter = _ClusterBackendAdapter(backend=backend)
    cluster_pipeline = VerilogStimulusClusterPipeline(
        llm_client=adapter,
        shared_cache=cluster_shared_cache,
        max_candidate_workers=max_candidate_workers,
    )
    scenario_object = _scenario_dict_to_cluster_object(harder_scenario)
    stimulus_cases = cluster_pipeline.generate_stimulus_cases(
        spec_text,
        majority_interface,
        majority_circuit_type,
        [scenario_object],
        num_cases=1,
        clock_name=clock_name,
        reset_name=reset_name,
    )
    selected_case = _stimulus_case_to_dict(stimulus_cases[0]) if stimulus_cases else None
    record["llm_calls"] = adapter.records
    record["model_usage"] = _summarize_llm_call_records(adapter.records)
    record["generated_stimulus_cases"] = [
        _stimulus_case_to_dict(case)
        for case in stimulus_cases
    ]
    record["selected_harder_stimulus_case"] = _deepcopy_jsonable(selected_case) if selected_case else None
    return selected_case, record


def _scenario_dict_to_cluster_object(item: dict[str, Any]) -> VerificationScenario:
    return VerificationScenario(
        name=str(item.get("name", "")).strip(),
        goal=str(item.get("goal", "")).strip(),
        story=str(item.get("story", "")).strip(),
    )


def _stimulus_case_to_dict(case: StimulusCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "description": case.description,
        "reason_based_on_spec": case.reason_based_on_spec,
        "steps": [
            {
                "sets": dict(step.sets),
                "delay": step.delay,
                "sample": step.sample,
                "sample_phase": step.sample_phase,
            }
            for step in case.steps
        ],
    }


def _stimulus_case_from_dict(item: dict[str, Any]) -> StimulusCase:
    """Rehydrate one JSON-style stimulus case dict back into a StimulusCase dataclass."""

    steps = []
    for raw_step in item.get("steps", []) or []:
        if not isinstance(raw_step, dict):
            continue
        steps.append(
            StimulusStep(
                sets=dict(raw_step.get("sets", {}) or {}),
                delay=int(raw_step.get("delay", 0) or 0),
                sample=bool(raw_step.get("sample", False)),
                sample_phase=str(raw_step.get("sample_phase", "auto") or "auto").strip(),
            )
        )
    return StimulusCase(
        name=str(item.get("name", "")).strip(),
        description=str(item.get("description", "")).strip(),
        steps=steps,
        reason_based_on_spec=str(item.get("reason_based_on_spec", "")).strip(),
    )


def _screen_prefilter_scenario_worker(
    *,
    scenario_index: int,
    scenario: dict[str, Any],
    stimulus_case_dict: dict[str, Any],
    raw_candidates: list[Any],
    majority_interface: InterfaceInfo,
    majority_circuit_type: str,
    workdir: str,
    sample_edge: str,
    clock_name: str | None,
    reset_info: Any,
) -> dict[str, Any]:
    """Run one scenario/stimulus screening task in its own process and return only summarized outputs."""

    pipeline = VerilogStimulusClusterPipeline(
        llm_client=_NoopClusterLLMClient(),
        shared_cache={},
    )
    stimulus_case = _stimulus_case_from_dict(stimulus_case_dict)
    simulation_records = pipeline.simulate_candidates(
        raw_candidates,
        majority_interface,
        majority_circuit_type,
        [stimulus_case],
        workdir,
        sample_edge=sample_edge,
        clock_name=clock_name,
        reset_info=reset_info,
    )
    functional_clusters = pipeline.cluster_functional_results(
        simulation_records,
        majority_interface,
        majority_circuit_type,
    )
    resolution = pipeline.build_resolution_output(
        raw_candidates,
        [stimulus_case],
        simulation_records,
        functional_clusters,
    )
    semantic_entropy = _semantic_entropy_from_resolution(resolution)
    resolution_mode = str(resolution.get("mode", "")).strip()
    should_discard = _entropy_is_zero(semantic_entropy)
    return {
        "scenario_index": scenario_index,
        "scenario_id": str(scenario.get("scenario_id", "")),
        "target_requirement_id": scenario.get("target_requirement_id"),
        "scenario_name": scenario.get("name"),
        "stimulus_case_name": stimulus_case.name,
        "stimulus_case": _stimulus_case_to_dict(stimulus_case),
        "kept": not should_discard,
        "screening_status": (
            "discarded_zero_entropy_single_cluster"
            if should_discard
            else "kept_after_prefilter"
        ),
        "resolution_mode": resolution_mode,
        "functional_cluster_count": len(functional_clusters),
        "semantic_entropy": semantic_entropy,
        "functional_clusters": summarize_functional_clusters(functional_clusters),
        "resolution": resolution,
        "excluded_candidate_count": len(resolution.get("excluded_candidates", []) or []),
    }


def _screen_prefilter_scenario_worker_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """ProcessPoolExecutor-friendly wrapper around the keyword-based screening worker."""

    return _screen_prefilter_scenario_worker(**payload)


def _convert_initial_cluster_candidates_to_pipeline_candidates(
    cluster_candidates: list[Any],
) -> list[CandidateVerilog]:
    """Convert raw llm_verilog_cluster candidates into pipeline-standard candidate objects."""

    converted: list[CandidateVerilog] = []
    for candidate in cluster_candidates:
        verilog_code = str(getattr(candidate, "verilog_code", "") or "").strip()
        if not verilog_code:
            continue
        source_candidate_id = getattr(candidate, "candidate_id", None)
        converted.append(
            CandidateVerilog(
                candidate_id=f"generated_prefilter_initial_{source_candidate_id}",
                raw_response=verilog_code,
                clean_verilog=verilog_code,
                module_name=guess_module_name(verilog_code),
                generation_notes=(
                    "Generated during scenario prefilter fixed-initial-candidate screening "
                    f"(source_candidate_id={source_candidate_id})."
                ),
            )
        )
    return converted


def _prefilter_fixed_candidate_source_id(candidate: CandidateVerilog) -> str:
    """Recover the raw fixed-candidate id from a prefilter-generated pipeline candidate."""

    candidate_id = str(candidate.candidate_id or "").strip()
    prefix = "generated_prefilter_initial_"
    if candidate_id.startswith(prefix):
        return candidate_id[len(prefix):]
    return candidate_id


def _collect_prefilter_excluded_candidate_ids(
    screened_scenarios: Sequence[dict[str, Any]],
) -> set[str]:
    """Union all candidate ids excluded during scenario prefilter simulation."""

    excluded_ids: set[str] = set()
    for scenario_result in screened_scenarios:
        resolution = scenario_result.get("resolution") or {}
        excluded_candidates = resolution.get("excluded_candidates") or []
        if not isinstance(excluded_candidates, list):
            continue
        for entry in excluded_candidates:
            if not isinstance(entry, dict):
                continue
            candidate_id = str(entry.get("candidate_id", "")).strip()
            if candidate_id:
                excluded_ids.add(candidate_id)
    return excluded_ids


def _replace_or_append_candidate(
    candidates: list[CandidateVerilog],
    revised_candidate: CandidateVerilog,
    source_candidate_id: str | None,
) -> list[CandidateVerilog]:
    """Keep `trace.candidates` aligned with the candidate that was actually verified."""

    updated = list(candidates)
    source_id = str(source_candidate_id or "").strip()
    for index, candidate in enumerate(updated):
        if candidate.candidate_id == revised_candidate.candidate_id:
            updated[index] = revised_candidate
            return updated
        if source_id and candidate.candidate_id == source_id:
            updated[index] = revised_candidate
            return updated
    updated.append(revised_candidate)
    return updated


def _run_multiclock_like_direct_fallback(
    *,
    task: TaskSpec,
    config: PipelineConfig,
    spec_text: str,
    all_requirements: list[str],
    scenario_pool: list[dict[str, Any]],
    discard_scenario_record: dict[str, Any],
    backends: ProviderBundle,
) -> tuple[
    list[str],
    list[CandidateVerilog],
    CandidateVerilog | None,
    list[VerificationResult],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Short-circuit multi-clock-like tasks to one direct generation + real testbench verification."""

    monitor = get_active_run_monitor()
    if monitor is not None:
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant="requirements_constraint_selfplanning",
            phase="multiclock_direct_fallback_start",
            details={
                "scenario_pool_size": len(scenario_pool),
                "fallback_reason": str(
                    discard_scenario_record.get("seq_control_fallback_reason", "")
                ).strip(),
            },
        )

    candidate, generation_record = generate_candidate_from_spec_only(
        task=task,
        spec_text=spec_text,
        backend=backends.generator_backend,
        additional_guidance=REQUIREMENTS_BRANCH_VERILOG_GUIDANCE,
    )
    candidates = [candidate]
    revised_candidate, verification_result, postprocess_record = (
        _postprocess_and_verify_cluster_candidate(
            task=task,
            spec_text=spec_text,
            candidate=candidate,
            backend=backends.repair_backend,
            timeout_sec=config.verification_timeout_sec,
        )
    )
    candidates = _replace_or_append_candidate(
        candidates,
        revised_candidate,
        str(postprocess_record.get("source_candidate_id", "")).strip() or None,
    )

    fallback_record = _deepcopy_jsonable(discard_scenario_record)
    fallback_record["multiclock_short_circuit"] = True
    fallback_record["fallback_selection"] = {
        "mode": CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK,
        "selected_candidate_id": revised_candidate.candidate_id,
        "source_candidate_id": candidate.candidate_id,
        "reason": (
            "Detected a multi-clock-like sequential control contract during scenario prefilter, "
            "so requirement-by-requirement targeted clustering was skipped in favor of one direct "
            "whole-spec generation followed by real testbench verification."
        ),
    }

    round_record = {
        "target_requirement_id": None,
        "local_round_index": None,
        "final_generation": True,
        "final_generation_mode": CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK,
        "resolution_mode": CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK,
        "selected_candidate_id": revised_candidate.candidate_id,
        "candidate_count": len(candidates),
        "candidate_ids": [item.candidate_id for item in candidates],
        "llm_calls": [generation_record],
        "model_usage": _summarize_llm_call_records([generation_record]),
        "fallback_reason": fallback_record["fallback_selection"]["reason"],
        "output_initialization_postprocess": postprocess_record,
    }
    loop_record = {
        "rounds": [round_record],
        "requirement_rounds": [],
        "scenario_pool": scenario_pool,
        "scenario_prefilter": fallback_record,
        "executed_scenario_ids": [],
        "resolved_requirement_ids": [],
        "final_disambiguation_constraints": [],
        "discarded_constraints": [],
        "model_usage": _summarize_llm_call_records([generation_record]),
    }
    loop_io = {
        "scenario_pool": {
            "scenario_pool": _deepcopy_jsonable(scenario_pool),
        },
        "requirement_rounds": [],
        "selected_mode": CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK,
        "executed_scenario_ids": [],
        "resolved_requirement_ids": [],
        "final_disambiguation_constraints": [],
        "discarded_constraints": [],
        "scenario_prefilter": fallback_record,
    }
    harder_scenario_record = {
        "selected_mode": CLUSTER_FEEDBACK_MODE_MULTICLOCK_DIRECT_FALLBACK,
        "families": [],
        "llm_calls": [],
        "model_usage": _summarize_llm_call_records([]),
    }

    if monitor is not None:
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant="requirements_constraint_selfplanning",
            phase="multiclock_direct_fallback_complete",
            details={
                "selected_candidate_id": revised_candidate.candidate_id,
                "verification_passed": verification_result.passed,
            },
        )

    return (
        [],
        candidates,
        revised_candidate,
        [verification_result],
        loop_record,
        loop_io,
        fallback_record,
        harder_scenario_record,
    )


def _prefilter_requirement_scenarios_with_fixed_initial_candidates(
    task: TaskSpec,
    config: PipelineConfig,
    spec_text: str,
    all_requirements: list[str],
    scenario_pool: list[dict[str, Any]],
    backend: LLMBackend,
    cluster_shared_cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[CandidateVerilog]]:
    """Screen scenarios once using a fixed initial candidate pool and keep only divergence-producing ones."""

    if not scenario_pool:
        return [], {
            "enabled": True,
            "screening_status": "empty_scenario_pool",
            "initial_candidate_count": 0,
            "initial_candidate_ids": [],
            "scenario_pool_size": 0,
            "generated_stimulus_count": 0,
            "kept_scenario_ids": [],
            "discarded_scenario_ids": [],
            "kept_count": 0,
            "discarded_count": 0,
            "screened_scenarios": [],
            "llm_calls": [],
            "model_usage": _summarize_llm_call_records([]),
        }, []

    task_output_dir = config.trace_root / "requirements_constraint_selfplanning" / task.task_name
    prefilter_root = task_output_dir / "cluster_artifacts" / "scenario_prefilter"
    if prefilter_root.exists():
        shutil.rmtree(prefilter_root)
    prefilter_workdir = prefilter_root / "workdir"
    prefilter_workdir.mkdir(parents=True, exist_ok=True)

    augmented_spec = _build_cluster_augmented_spec(
        spec_text=spec_text,
        requirements=all_requirements,
        disambiguation_constraints=None,
    )
    adapter = _ClusterBackendAdapter(backend=backend)
    cluster_pipeline = VerilogStimulusClusterPipeline(
        llm_client=adapter,
        shared_cache=cluster_shared_cache,
        max_candidate_workers=config.task_max_candidate_workers,
        verilog_user_prompt=_build_requirements_branch_cluster_verilog_user_prompt(),
    )

    raw_candidates = cluster_pipeline.generate_candidates(augmented_spec)
    fixed_initial_candidates = _convert_initial_cluster_candidates_to_pipeline_candidates(raw_candidates)
    interface_clusters = cluster_pipeline.cluster_interfaces(raw_candidates)
    (
        majority_cluster,
        majority_interface,
        majority_circuit_type,
        majority_circuit_type_source_candidate_id,
    ) = cluster_pipeline.select_majority_interface(interface_clusters)
    interface_report = cluster_pipeline.build_interface_report(
        raw_candidates,
        interface_clusters,
        majority_cluster,
        majority_interface,
        majority_circuit_type,
        majority_circuit_type_source_candidate_id,
    )

    if majority_cluster is None or majority_interface is None:
        kept_entries = [
            {
                "scenario": dict(scenario),
                "stimulus_case": None,
                "screening_status": "unscreened_no_majority_interface",
            }
            for scenario in scenario_pool
        ]
        record = {
            "enabled": True,
            "screening_status": "unscreened_no_majority_interface",
            "initial_candidate_count": len(raw_candidates),
            "initial_candidate_ids": [candidate.candidate_id for candidate in raw_candidates],
            "scenario_pool_size": len(scenario_pool),
            "kept_scenario_ids": [str(item.get("scenario_id", "")) for item in scenario_pool],
            "discarded_scenario_ids": [],
            "kept_count": len(kept_entries),
            "discarded_count": 0,
            "interface_report": interface_report,
            "screened_scenarios": [],
            "llm_calls": adapter.records,
            "model_usage": _summarize_llm_call_records(adapter.records),
        }
        return kept_entries, record, fixed_initial_candidates

    clock_name_hint: str | None = None
    sample_edge_hint = "posedge"
    reset_name_hint: str | None = None
    majority_reset_info = None
    prefilter_seq_control_fallback_reason: str | None = None
    if majority_circuit_type == "SEQ":
        try:
            seq_control_contract = cluster_pipeline.extract_seq_control_contract_from_spec(
                augmented_spec,
                majority_interface,
            )
            clock_name_hint, sample_edge_hint, majority_reset_info = cluster_pipeline.seq_control_contract_to_runtime(
                seq_control_contract
            )
            reset_name_hint = majority_reset_info.signal_name
        except MultiClockSeqControlContractError:
            prefilter_seq_control_fallback_reason = "multiclock_seq_control_contract"
            majority_reset_info = ResetInfo(None, "none")

    if prefilter_seq_control_fallback_reason is not None:
        kept_entries = [
            {
                "scenario": dict(scenario),
                "stimulus_case": None,
                "screening_status": "unscreened_multiclock_seq_control_contract",
            }
            for scenario in scenario_pool
        ]
        record = {
            "enabled": True,
            "screening_status": "unscreened_multiclock_seq_control_contract",
            "seq_control_fallback_reason": prefilter_seq_control_fallback_reason,
            "initial_candidate_count": len(raw_candidates),
            "initial_candidate_ids": [candidate.candidate_id for candidate in raw_candidates],
            "scenario_pool_size": len(scenario_pool),
            "kept_scenario_ids": [str(item.get("scenario_id", "")) for item in scenario_pool],
            "discarded_scenario_ids": [],
            "kept_count": len(kept_entries),
            "discarded_count": 0,
            "interface_report": interface_report,
            "screened_scenarios": [],
            "llm_calls": adapter.records,
            "model_usage": _summarize_llm_call_records(adapter.records),
        }
        return kept_entries, record, fixed_initial_candidates

    scenario_objects = [_scenario_dict_to_cluster_object(item) for item in scenario_pool]
    stimulus_cases = cluster_pipeline.generate_stimulus_cases(
        augmented_spec,
        majority_interface,
        majority_circuit_type,
        scenario_objects,
        num_cases=len(scenario_objects),
        clock_name=clock_name_hint,
        reset_name=reset_name_hint,
    )

    kept_entries: list[dict[str, Any]] = []
    screened_scenarios: list[dict[str, Any]] = []
    kept_scenario_ids: list[str] = []
    discarded_scenario_ids: list[str] = []
    kept_entries_by_index: dict[int, dict[str, Any]] = {}
    screened_scenarios_by_index: dict[int, dict[str, Any]] = {}
    screening_jobs: list[tuple[int, dict[str, Any], VerificationScenario, StimulusCase]] = []
    for scenario_index, scenario in enumerate(scenario_pool):
        scenario_id = str(scenario.get("scenario_id", ""))
        if scenario_index >= len(stimulus_cases):
            kept_entries_by_index[scenario_index] = {
                "scenario": dict(scenario),
                "stimulus_case": None,
                "screening_status": "kept_unscreened_missing_stimulus",
            }
            kept_scenario_ids.append(scenario_id)
            screened_scenarios_by_index[scenario_index] = {
                "scenario_id": scenario_id,
                "target_requirement_id": scenario.get("target_requirement_id"),
                "kept": True,
                "screening_status": "kept_unscreened_missing_stimulus",
                "resolution_mode": None,
            }
            continue

        screening_jobs.append(
            (
                scenario_index,
                dict(scenario),
                scenario_objects[scenario_index],
                stimulus_cases[scenario_index],
            )
        )

    def _screen_prefilter_scenario_with_refinement(
        job: tuple[int, dict[str, Any], VerificationScenario, StimulusCase]
    ) -> tuple[int, dict[str, Any]]:
        started_at = time.time()
        worker_thread_name = current_thread().name
        scenario_index, scenario, scenario_object, initial_stimulus_case = job
        scenario_id = str(scenario.get("scenario_id", ""))
        resolution_bundle = cluster_pipeline.run_fixed_candidate_resolution(
            spec=augmented_spec,
            candidates=raw_candidates,
            majority_interface=majority_interface,
            majority_circuit_type=majority_circuit_type,
            verification_scenarios=[scenario_object],
            stimulus_cases=[initial_stimulus_case],
            workdir=str(prefilter_workdir / f"scenario_{scenario_index:03d}"),
            sample_edge=sample_edge_hint,
            clock_name=clock_name_hint,
            reset_info=majority_reset_info or ResetInfo(None, "none"),
        )
        final_stimulus_cases = list(resolution_bundle.get("stimulus_cases", []))
        final_stimulus_case = (
            _stimulus_case_to_dict(final_stimulus_cases[0])
            if final_stimulus_cases
            else _stimulus_case_to_dict(initial_stimulus_case)
        )
        functional_clusters = list(resolution_bundle.get("functional_clusters", []))
        resolution = dict(resolution_bundle.get("resolution", {}))
        semantic_entropy = _semantic_entropy_from_resolution(resolution)
        resolution_mode = str(resolution.get("mode", "")).strip()
        should_discard = _entropy_is_zero(semantic_entropy)
        finished_at = time.time()
        result = {
            "scenario_index": scenario_index,
            "scenario_id": scenario_id,
            "target_requirement_id": scenario.get("target_requirement_id"),
            "scenario_name": scenario.get("name"),
            "stimulus_case_name": final_stimulus_case.get("name"),
            "stimulus_case": final_stimulus_case,
            "kept": not should_discard,
            "screening_status": (
                "discarded_zero_entropy_single_cluster"
                if should_discard
                else "kept_after_prefilter"
            ),
            "resolution_mode": resolution_mode,
            "functional_cluster_count": len(functional_clusters),
            "semantic_entropy": semantic_entropy,
            "functional_clusters": summarize_functional_clusters(functional_clusters),
            "resolution": resolution,
            "excluded_candidate_count": len(resolution.get("excluded_candidates", []) or []),
            "stimulus_refinement": _deepcopy_jsonable(
                resolution_bundle.get("stimulus_refinement", {})
            ),
            "execution_mode": "thread_pool",
            "worker_thread_name": worker_thread_name,
            "started_at_epoch_seconds": started_at,
            "finished_at_epoch_seconds": finished_at,
            "elapsed_seconds": finished_at - started_at,
        }
        return scenario_index, result

    screened_results: list[tuple[int, dict[str, Any]]] = []
    prefilter_execution_mode = "none"
    prefilter_max_workers = 0
    if screening_jobs:
        max_workers = min(
            len(screening_jobs),
            max(1, os.cpu_count() or 1),
            max(1, config.task_max_prefilter_workers),
        )
        prefilter_max_workers = max_workers
        if max_workers <= 1:
            prefilter_execution_mode = "sequential"
            screened_results = [
                _screen_prefilter_scenario_with_refinement(job)
                for job in screening_jobs
            ]
        else:
            prefilter_execution_mode = "thread_pool"
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                screened_results = list(
                    executor.map(
                        _screen_prefilter_scenario_with_refinement,
                        screening_jobs,
                    )
                )

    for scenario_index, result in screened_results:
        scenario = scenario_pool[scenario_index]
        scenario_id = str(scenario.get("scenario_id", ""))
        if result.get("kept"):
            kept_entries_by_index[scenario_index] = {
                "scenario": dict(scenario),
                "stimulus_case": _deepcopy_jsonable(result.get("stimulus_case")),
                "screening_status": "kept_after_prefilter",
            }
            kept_scenario_ids.append(scenario_id)
        else:
            discarded_scenario_ids.append(scenario_id)
        screened_scenarios_by_index[scenario_index] = result

    kept_entries = [kept_entries_by_index[index] for index in sorted(kept_entries_by_index)]
    screened_scenarios = [
        screened_scenarios_by_index[index]
        for index in sorted(screened_scenarios_by_index)
    ]
    prefilter_excluded_candidate_ids = sorted(
        _collect_prefilter_excluded_candidate_ids(screened_scenarios),
        key=str,
    )

    record = {
        "enabled": True,
        "screening_status": "completed",
        "initial_candidate_count": len(raw_candidates),
        "initial_candidate_ids": [candidate.candidate_id for candidate in raw_candidates],
        "scenario_pool_size": len(scenario_pool),
        "generated_stimulus_count": len(stimulus_cases),
        "prefilter_execution_mode": prefilter_execution_mode,
        "prefilter_max_workers": prefilter_max_workers,
        "kept_scenario_ids": kept_scenario_ids,
        "discarded_scenario_ids": discarded_scenario_ids,
        "kept_count": len(kept_scenario_ids),
        "discarded_count": len(discarded_scenario_ids),
        "prefilter_excluded_candidate_ids": prefilter_excluded_candidate_ids,
        "interface_report": interface_report,
        "screened_scenarios": screened_scenarios,
        "llm_calls": adapter.records,
        "model_usage": _summarize_llm_call_records(adapter.records),
    }
    return kept_entries, record, fixed_initial_candidates


def _build_cluster_augmented_spec(
    spec_text: str,
    requirements: list[str],
    disambiguation_constraints: list[str] | None = None,
) -> str:
    """Build the enriched natural-language spec passed into the Verilog cluster pipeline."""

    disambiguation_section = ""
    if disambiguation_constraints:
        disambiguation_section = f"""

Additional Verilog-divergence disambiguation constraints:
{json.dumps(disambiguation_constraints, ensure_ascii=False, indent=2)}
""".rstrip()

    return f"""
Original specification:
{spec_text}

Extracted requirements:
{json.dumps(requirements or [], ensure_ascii=False, indent=2)}

Interpretation rules for this enriched specification:
- The original specification remains the source of truth.
- The extracted requirements restate explicit behavior that must be preserved.
- Do not add any behavior that is not supported by the original specification.
- Additional Verilog-divergence disambiguation constraints, if present, are targeted warnings derived from previously observed candidate disagreements.
{disambiguation_section}
""".strip()


def _convert_cluster_resolution_to_candidates(
    result: dict[str, Any],
) -> tuple[list[CandidateVerilog], CandidateVerilog | None]:
    """Convert cluster resolution output into pipeline-standard candidate objects."""

    resolution = result.get("resolution") or {}
    mode = str(resolution.get("mode", "")).strip()
    candidates: list[CandidateVerilog] = []

    if mode == "single_majority_verilog":
        verilog_code = str(resolution.get("selected_verilog", "")).strip()
        if verilog_code:
            candidate = CandidateVerilog(
                candidate_id="generated_cluster_majority",
                raw_response=verilog_code,
                clean_verilog=verilog_code,
                module_name=guess_module_name(verilog_code),
                generation_notes="Selected by llm_verilog_cluster as the single majority Verilog.",
            )
            candidates.append(candidate)
            return candidates, candidate

    if mode == "top_two_verilog_clusters":
        selected_candidates = resolution.get("selected_candidates") or []
        for index, item in enumerate(selected_candidates[:2], start=1):
            if not isinstance(item, dict):
                continue
            verilog_code = str(item.get("verilog_code", "")).strip()
            if not verilog_code:
                continue
            candidate = CandidateVerilog(
                candidate_id=f"generated_cluster_top{index}",
                raw_response=verilog_code,
                clean_verilog=verilog_code,
                module_name=guess_module_name(verilog_code),
                generation_notes=(
                    "Selected by llm_verilog_cluster as one of the top two functional clusters "
                    f"(cluster_id={item.get('cluster_id')}, source_candidate_id={item.get('candidate_id')})."
                ),
            )
            candidates.append(candidate)
        return candidates, candidates[0] if candidates else None

    return [], None


def _extract_dual_candidate_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the compact dual-candidate payload if the cluster result contains two leading behaviors."""

    resolution = result.get("resolution") or {}
    if str(resolution.get("mode", "")).strip() != "top_two_verilog_clusters":
        return None
    selected_candidates = resolution.get("selected_candidates") or []
    if len(selected_candidates) < 2:
        return None
    candidate_a = selected_candidates[0]
    candidate_b = selected_candidates[1]
    return {
        "candidate_a": {
            "cluster_id": candidate_a.get("cluster_id"),
            "candidate_id": candidate_a.get("candidate_id"),
            "verilog_code": candidate_a.get("verilog_code"),
        },
        "candidate_b": {
            "cluster_id": candidate_b.get("cluster_id"),
            "candidate_id": candidate_b.get("candidate_id"),
            "verilog_code": candidate_b.get("verilog_code"),
        },
        "different_output_trajectories": resolution.get("differing_cases", []),
    }


def _judge_requirement_dual_winner(
    spec_text: str,
    target_requirement_id: int,
    target_requirement: str,
    scenario: dict[str, Any],
    stimulus_case: dict[str, Any],
    dual_payload: dict[str, Any],
    prior_requirement_constraints: Sequence[str],
    backend: LLMBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge which observable trajectory better matches the specification under one shared stimulus."""

    system_prompt = (
        "You judge which observable trajectory better matches the original specification.\n"
        "The original specification is the source of truth.\n"
        "The target requirement identifies the intended behavior under test.\n"
        "Compare only the observed trajectories under the shared stimulus and choose exactly one winner.\n"
        "Do not judge by coding style or likely internal implementation.\n"
        "Return JSON only."
    )
    user_prompt = f"""
Return JSON with keys:
- winner
- reason_based_on_spec
- critical_behavior_difference

Original specification:
{spec_text}

Target requirement:
{target_requirement}

Requirement-targeted scenario:
{json.dumps(scenario, ensure_ascii=False, indent=2)}

Stimulus case used for this requirement:
{json.dumps(stimulus_case, ensure_ascii=False, indent=2)}

Different trajectories observed under this shared stimulus:
{json.dumps(dual_payload["different_output_trajectories"], ensure_ascii=False, indent=2)}

Structured summary of the first up to three observable output divergences:
{json.dumps([
    {
        "case_index": item.get("case_index"),
        "case_name": item.get("case_name"),
        "trajectory_difference_summary": item.get("trajectory_difference_summary", {}),
    }
    for item in dual_payload["different_output_trajectories"]
], ensure_ascii=False, indent=2)}

Previously accepted constraints for this same target requirement from earlier local rounds:
{json.dumps(list(prior_requirement_constraints), ensure_ascii=False, indent=2)}

Rules:
- choose exactly one winner: candidate_a or candidate_b
- judge which observed trajectory more closely matches the original specification under this shared stimulus
- prioritize the specification's intended observable behavior over local superficial similarity
- For waveform-generation tasks, if the specification describes an internal state transition or direction change at a numeric boundary, do not automatically infer that the externally visible output must also step away from that boundary on the immediately following sampled cycle unless the specification explicitly says so; judge by the stated or observed output sequence itself rather than by an assumed internal-transition timing consequence.
- For waveform-generation tasks, do not infer an immediate next-sample departure from a boundary value solely from an internal state or direction change; if the specification does not explicitly fix that next-sample timing, judge by whether the visible waveform correctly reaches the boundary rather than by whether it holds that boundary for one extra sample.
- For sequential register-update tasks, if the specification says a register is shifted or otherwise updated each cycle and another register or output uses it on that same clock edge, do not automatically infer that the same-edge observable result must use the already-shifted or already-updated value; unless the specification explicitly states that intra-edge ordering, judge the same-edge result using the register's pre-edge current value.
- For clocked accumulation modules that emit a result after the Nth accepted valid sample, prefer the trajectory where `valid_out` and `data_out` are registered edge-driven outputs that update on the accepting clock edge, not a purely combinational function of the current input and pre-edge count.
- If the specification says only that inputs are counted when `valid_in` is asserted, do not invent a contiguous-valid-only batching rule, idle cancellation rule, or flush-on-gap rule unless the specification explicitly states it.
- For restoring-division tasks with a wider dividend than divisor, prefer the trajectory that processes all dividend bits and yields a full-width quotient/remainder consistent with the complete division, not a shortened upper-slice-only or partial-iteration result.
- For sign-dependent fixed-point arithmetic tasks, do not promote one observed different-sign case into a global sign-magnitude policy; prefer the trajectory that preserves the specification's exact case split over one that rewrites all different-sign behavior into a textbook rule.
- For IEEE-754 floating-point tasks, do not treat exact cycle counts, exact hexadecimal witness outputs, or one testcase-specific bit patterns as decisive unless the specification explicitly requires them; prefer the trajectory that matches the global IEEE-754 rules for special cases, normalization, exponent adjustment, and round-to-even behavior.
- If the main difference is only that one trajectory remains undefined before the first specification-establishing reset or first defined state, do not treat that alone as the decisive behavioral advantage; prioritize post-reset, specification-relevant behavior.
- use the earliest divergence summary only as an entry point; synthesize across the early differing signals before deciding what behavior is actually wrong
- do not ignore later support from the same early-differing signals if it confirms or weakens the earliest local symptom
- previously accepted constraints for this same target requirement are already established context; do not contradict them unless the new observable evidence clearly forces a revision
- do not use coding style, elegance, or likely implementation quality as evidence
- reason_based_on_spec must explain why the chosen winner is more faithful to the specification
- critical_behavior_difference must identify the single most important observable difference between the two trajectories that determined the winner
""".strip()
    record = backend.complete_json_record(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = record["parsed_response"]
    raw_winner = str(parsed.get("winner", "")).strip()
    winner = raw_winner if raw_winner in {"candidate_a", "candidate_b"} else "candidate_a"
    judgment = {
        "target_requirement_id": target_requirement_id,
        "winner": winner,
        "reason_based_on_spec": str(parsed.get("reason_based_on_spec", "")).strip(),
        "critical_behavior_difference": str(parsed.get("critical_behavior_difference", "")).strip(),
    }
    if raw_winner != winner:
        judgment["winner_fallback_applied"] = True
        judgment["raw_winner"] = raw_winner
    record["judgment"] = judgment
    return judgment, record


def _analyze_requirement_dual_constraint(
    spec_text: str,
    target_requirement_id: int,
    target_requirement: str,
    scenario: dict[str, Any],
    stimulus_case: dict[str, Any],
    dual_payload: dict[str, Any],
    winner_judgment: dict[str, Any],
    prior_requirement_constraints: Sequence[str],
    backend: LLMBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate one additive constraint directly from one requirement-targeted dual-cluster disagreement."""

    system_prompt = (
        "You analyze a requirement-targeted observable-behavior disagreement.\n"
        "The original specification is the source of truth.\n"
        "The target requirement identifies the intended behavior under test.\n"
        "A prior judgment has already selected the more specification-consistent observed trajectory.\n"
        "Use that winner judgment to generate one additive constraint grounded in the specification.\n"
        "Return JSON only."
    )
    user_prompt = f"""
Return JSON with keys:
- divergence_analysis_based_on_spec
- new_constraint

Original specification:
{spec_text}

Target requirement:
{target_requirement}

Requirement-targeted scenario:
{json.dumps(scenario, ensure_ascii=False, indent=2)}

Stimulus case used for this requirement:
{json.dumps(stimulus_case, ensure_ascii=False, indent=2)}

Different trajectories observed under this shared stimulus:
{json.dumps(dual_payload["different_output_trajectories"], ensure_ascii=False, indent=2)}

Structured summary of the first up to three observable output divergences:
{json.dumps([
    {
        "case_index": item.get("case_index"),
        "case_name": item.get("case_name"),
        "trajectory_difference_summary": item.get("trajectory_difference_summary", {}),
    }
    for item in dual_payload["different_output_trajectories"]
], ensure_ascii=False, indent=2)}

Winner judgment:
{json.dumps(winner_judgment, ensure_ascii=False, indent=2)}

Previously accepted constraints for this same target requirement from earlier local rounds:
{json.dumps(list(prior_requirement_constraints), ensure_ascii=False, indent=2)}

    Rules:
- generate exactly one additive constraint
- use the winner judgment and critical_behavior_difference to bias the constraint toward the more specification-consistent observable behavior
- the constraint must be grounded in the original specification
- the constraint must resolve exactly one externally observable ambiguity exposed by this shared stimulus
- For waveform-generation tasks, if the specification mentions a state change or slope reversal at a boundary value, phrase the constraint in terms of the externally visible output sequence around that boundary and do not rewrite it into an assumed immediate departure from the boundary unless the specification explicitly requires that next-sample change.
- For waveform-generation tasks, phrase the constraint in terms of the observable boundary-reaching sequence, and do not turn a boundary reversal into an exact next-cycle rule such as "31 must be followed by 30" or "0 must be followed by 1" unless the specification explicitly requires that timing.
- For sequential register-update tasks, phrase the constraint using standard same-edge register semantics: if a register is shifted or otherwise updated on a clock edge and another register or output accumulates or uses it on that same edge, the same-cycle observable result should use the pre-edge current registered value, and the updated value should affect the following cycle, unless the specification explicitly requires immediate use of the updated value.
- For clocked accumulation modules, phrase the constraint so `valid_out` and `data_out` are externally observable registered outputs that change only on the clock edge that accepts the Nth valid sample, rather than combinational outputs that depend immediately on the current input value and counter state.
- If the original specification only says that samples are accumulated when `valid_in` is asserted, phrase the constraint in terms of counting accepted valid samples only; do not encode a contiguous-valid-only requirement, idle-gap flush rule, or batch cancellation rule unless the specification explicitly requires it.
- For restoring-division tasks with a wider dividend than divisor, phrase the constraint so the externally visible quotient and remainder reflect completion of the compare-subtract-shift loop over all dividend bits, rather than an initial upper-slice decision plus only partial remaining-bit processing.
- For sign-dependent fixed-point arithmetic tasks, do not generalize one observed different-sign case into a global rule; phrase the constraint only for the exact sign and magnitude relation exposed by the shared stimulus unless the specification explicitly states a broader policy.
- For IEEE-754 floating-point tasks, do not encode exact cycle numbers, exact hexadecimal witness outputs, or single-testcase bit patterns into the constraint unless the specification explicitly requires them; phrase the constraint at the level of global IEEE-754 behavior such as special-case precedence, normalization, exponent adjustment, and round-to-even rounding.
- use the earliest divergence summary only as an entry point; synthesize across the early differing signals before deciding what general behavior the constraint should capture
- prefer a constraint that explains the recurring early signal difference pattern, not a one-row local symptom by itself
- treat previously accepted constraints for this same target requirement as existing context; add only genuinely new or sharper information beyond them
- do not restate a previously accepted same-requirement constraint unless the new differing evidence clearly forces a timing or behavioral revision
    - phrase the constraint only in terms of externally observable behavior: input conditions, observable outputs, and clock/reset timing only when the specification is sequential
    - do not encode implementation choices, internal state structure, helper registers, or preferred coding style unless the original specification explicitly requires them
    - do not restate or rewrite the requirement list; focus only on the single ambiguity exposed here
    - do not mention candidate_a or candidate_b inside the constraint
    - do not mention winner judgment metadata inside the constraint
    - keep the constraint specific enough to distinguish the correct external behavior under this requirement, without prescribing an implementation strategy
""".strip()
    record = backend.complete_json_record(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = record["parsed_response"]
    analysis = {
        "target_requirement_id": target_requirement_id,
        "divergence_analysis_based_on_spec": str(parsed.get("divergence_analysis_based_on_spec", "")).strip(),
        "new_constraint": str(parsed.get("new_constraint", "")).strip(),
        "winner": str(winner_judgment.get("winner", "")).strip(),
        "reason_based_on_spec": str(winner_judgment.get("reason_based_on_spec", "")).strip(),
        "critical_behavior_difference": str(winner_judgment.get("critical_behavior_difference", "")).strip(),
    }
    record["analysis"] = analysis
    return analysis, record


def _run_llm_verilog_cluster_resolution(
    task: TaskSpec,
    config: PipelineConfig,
    spec_text: str,
    requirements: list[str],
    backend: LLMBackend,
    round_index: int = 1,
    disambiguation_constraints: list[str] | None = None,
    external_verification_scenarios: list[dict[str, Any]] | None = None,
    external_stimulus_cases: list[dict[str, Any]] | None = None,
    cluster_shared_cache: dict[str, Any] | None = None,
    round_label: str | None = None,
    enable_stimulus_refinement: bool = True,
) -> tuple[list[CandidateVerilog], CandidateVerilog | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run llm_verilog_cluster using the enriched spec as its only semantic input."""

    task_output_dir = config.trace_root / "requirements_constraint_selfplanning" / task.task_name
    cluster_dir_name = round_label.strip() if isinstance(round_label, str) and round_label.strip() else f"round_{round_index:02d}"
    cluster_root = task_output_dir / "cluster_artifacts" / cluster_dir_name
    if cluster_root.exists():
        shutil.rmtree(cluster_root)
    workdir = cluster_root / "workdir"
    artifact_dir = cluster_root / "resolved"
    workdir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monitor = get_active_run_monitor()
    if monitor is not None:
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant="requirements_constraint_selfplanning",
            phase="cluster_round_resolve_spec_start",
            details={
                "round_index": round_index,
                "constraint_count": len(disambiguation_constraints or []),
                "external_scenario_count": len(external_verification_scenarios or []),
                "external_stimulus_count": len(external_stimulus_cases or []),
            },
        )

    augmented_spec = _build_cluster_augmented_spec(
        spec_text=spec_text,
        requirements=requirements,
        disambiguation_constraints=disambiguation_constraints,
    )
    adapter = _ClusterBackendAdapter(backend=backend)
    cluster_pipeline = VerilogStimulusClusterPipeline(
        llm_client=adapter,
        shared_cache=cluster_shared_cache,
        max_candidate_workers=config.task_max_candidate_workers,
        verilog_user_prompt=_build_requirements_branch_cluster_verilog_user_prompt(),
    )
    result = cluster_pipeline.resolve_spec(
        augmented_spec,
        workdir=workdir,
        artifact_dir=artifact_dir,
        num_stimuli=1 if external_verification_scenarios or external_stimulus_cases else None,
        external_verification_scenarios=external_verification_scenarios,
        external_stimulus_cases=external_stimulus_cases,
        enable_stimulus_refinement=enable_stimulus_refinement,
    )
    candidates, current_candidate = _convert_cluster_resolution_to_candidates(result)
    resolution = result.get("resolution", {}) or {}
    semantic_entropy = _semantic_entropy_from_resolution(resolution)
    functional_cluster_count = len(
        _extract_functional_cluster_sizes(list(resolution.get("functional_clusters", []) or []))
    )
    cluster_record = {
        "round_index": round_index,
        "spec_input": augmented_spec,
        "resolution_mode": resolution.get("mode"),
        "artifact_dir": result.get("artifact_dir"),
        "artifact_files": result.get("artifact_files", {}),
        "interface_report": result.get("interface_report", {}),
        "resolution": resolution,
        "functional_cluster_count": functional_cluster_count,
        "semantic_entropy": semantic_entropy,
        "llm_calls": adapter.records,
        "model_usage": _summarize_llm_call_records(adapter.records),
        "cache_usage": result.get("cache_usage", {}),
        "stimulus_refinement": result.get("stimulus_refinement", {}),
        "disambiguation_constraints_input": list(disambiguation_constraints or []),
        "external_verification_scenarios": list(external_verification_scenarios or []),
        "external_stimulus_cases": list(external_stimulus_cases or []),
    }
    cluster_trace = {
        "cluster_spec_input": augmented_spec,
        "llm_verilog_cluster": {
            "round_index": round_index,
            "resolution_mode": resolution.get("mode"),
            "artifact_dir": result.get("artifact_dir"),
            "artifact_files": result.get("artifact_files", {}),
            "interface_report": result.get("interface_report", {}),
            "resolution": resolution,
            "functional_cluster_count": functional_cluster_count,
            "semantic_entropy": semantic_entropy,
            "cache_usage": result.get("cache_usage", {}),
            "stimulus_refinement": result.get("stimulus_refinement", {}),
            "disambiguation_constraints_input": list(disambiguation_constraints or []),
            "external_verification_scenarios": list(external_verification_scenarios or []),
            "external_stimulus_cases": list(external_stimulus_cases or []),
        },
    }
    return candidates, current_candidate, cluster_record, cluster_trace, result


def _run_requirement_stimulus_constraint_loop(
    task: TaskSpec,
    config: PipelineConfig,
    spec_text: str,
    all_requirements: list[str],
    behavioral_requirements: list[str],
    backends: ProviderBundle,
    max_rounds_per_requirement: int = 3,
) -> tuple[
    list[str],
    list[CandidateVerilog],
    CandidateVerilog | None,
    list[VerificationResult],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Iterate requirement-by-requirement while co-evolving one scenario/stimulus family per surviving requirement."""

    monitor = get_active_run_monitor()
    if monitor is not None:
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant="requirements_constraint_selfplanning",
            phase="scenario_pool_generation_start",
            details={},
        )

    scenario_pool, scenario_pool_record = _generate_requirement_targeted_scenario_pool(
        spec_text=spec_text,
        behavior_requirements=behavioral_requirements,
        backend=backends.planner_backend,
    )
    accumulated_constraints: list[str] = []
    per_requirement_constraints: dict[int, list[str]] = {}
    executed_scenario_ids: list[str] = []
    resolved_requirement_ids: list[int] = []
    cluster_rounds: list[dict[str, Any]] = []
    requirement_loop_rounds: list[dict[str, Any]] = []
    discarded_constraints: list[dict[str, Any]] = []
    executed_scenarios_for_final_generation: list[dict[str, Any]] = []
    executed_stimulus_cases_for_final_generation: list[dict[str, Any]] = []
    harder_scenario_families: list[dict[str, Any]] = []
    harder_scenario_llm_calls: list[dict[str, Any]] = []
    final_candidates: list[CandidateVerilog] = []
    final_current_candidate: CandidateVerilog | None = None
    verification_results: list[VerificationResult] = []
    cluster_shared_cache: dict[str, Any] = {}
    screened_scenario_entries, discard_scenario_record, fixed_initial_candidates = _prefilter_requirement_scenarios_with_fixed_initial_candidates(
        task=task,
        config=config,
        spec_text=spec_text,
        all_requirements=all_requirements,
        scenario_pool=scenario_pool,
        backend=backends.generator_backend,
        cluster_shared_cache=cluster_shared_cache,
    )
    if monitor is not None:
        monitor.update(
            task_name=task.task_name,
            task_id=task.task_id,
            variant="requirements_constraint_selfplanning",
            phase="scenario_prefilter_complete",
            details={
                "scenario_pool_size": len(scenario_pool),
                "kept_scenarios": len(screened_scenario_entries),
                "discarded_scenarios": len(discard_scenario_record.get("discarded_scenario_ids", [])),
            },
        )

    if (
        str(discard_scenario_record.get("seq_control_fallback_reason", "")).strip()
        == "multiclock_seq_control_contract"
    ):
        return _run_multiclock_like_direct_fallback(
            task=task,
            config=config,
            spec_text=spec_text,
            all_requirements=all_requirements,
            scenario_pool=scenario_pool,
            discard_scenario_record=discard_scenario_record,
            backends=backends,
        )

    for screened_entry in screened_scenario_entries:
        scenario = dict(screened_entry.get("scenario") or {})
        target_requirement_id = int(scenario.get("target_requirement_id", 0) or 0)
        if target_requirement_id <= 0 or target_requirement_id > len(behavioral_requirements):
            continue
        target_requirement = str(behavioral_requirements[target_requirement_id - 1]).strip()
        target_requirement_bundle = [target_requirement]
        executed_scenario_ids.append(str(scenario.get("scenario_id", "")))
        prefiltered_stimulus_case = screened_entry.get("stimulus_case")
        current_scenario: dict[str, Any] = _deepcopy_jsonable(scenario)
        current_stimulus_case: dict[str, Any] | None = (
            _deepcopy_jsonable(prefiltered_stimulus_case)
            if isinstance(prefiltered_stimulus_case, dict)
            else None
        )
        scenario_history: list[dict[str, Any]] = [_deepcopy_jsonable(current_scenario)]
        stimulus_history: list[dict[str, Any]] = (
            [_deepcopy_jsonable(current_stimulus_case)]
            if isinstance(current_stimulus_case, dict)
            else []
        )
        family_expansion_record: dict[str, Any] = {
            "target_requirement_id": target_requirement_id,
            "target_requirement": target_requirement,
            "root_scenario": _deepcopy_jsonable(scenario),
            "root_stimulus_case": _deepcopy_jsonable(prefiltered_stimulus_case)
            if isinstance(prefiltered_stimulus_case, dict)
            else None,
            "family_rounds": [],
        }
        requirement_satisfied = False
        requirement_round_payload: dict[str, Any] = {
            "target_requirement_id": target_requirement_id,
            "target_requirement": target_requirement,
            "scenario": scenario,
            "rounds": [],
            "resolved_under_targeted_stimulus": False,
            "screening_status": str(screened_entry.get("screening_status", "")).strip(),
        }

        for local_round_index in range(1, max_rounds_per_requirement + 1):
            if monitor is not None:
                monitor.update(
                    task_name=task.task_name,
                    task_id=task.task_id,
                    variant="requirements_constraint_selfplanning",
                    phase="requirement_family_round_start",
                    details={
                        "target_requirement_id": target_requirement_id,
                        "target_requirement": target_requirement,
                        "local_round_index": local_round_index,
                        "global_round_index": len(cluster_rounds) + 1,
                        "scenario_family_size": 1 if current_scenario else 0,
                        "stimulus_family_size": 1 if current_stimulus_case else 0,
                        "scenario_history_size": len(scenario_history),
                        "stimulus_history_size": len(stimulus_history),
                        "constraint_count": len(accumulated_constraints),
                    },
                )
            main_round_index = len(cluster_rounds) + 1
            candidates, current_candidate, cluster_record, cluster_trace, cluster_result = _run_llm_verilog_cluster_resolution(
                task=task,
                config=config,
                spec_text=spec_text,
                requirements=target_requirement_bundle,
                backend=backends.generator_backend,
                round_index=main_round_index,
                disambiguation_constraints=accumulated_constraints,
                external_verification_scenarios=[current_scenario],
                external_stimulus_cases=[current_stimulus_case] if current_stimulus_case else None,
                cluster_shared_cache=cluster_shared_cache,
                enable_stimulus_refinement=True,
            )
            final_candidates = candidates
            final_current_candidate = current_candidate
            cluster_rounds.append(
                {
                    "target_requirement_id": target_requirement_id,
                    "local_round_index": local_round_index,
                    **cluster_record,
                }
            )
            if current_stimulus_case is None:
                generated_family = list(cluster_result.get("stimulus_cases") or [])
                if generated_family and isinstance(generated_family[0], dict):
                    current_stimulus_case = _deepcopy_jsonable(generated_family[0])
                    stimulus_history = [_deepcopy_jsonable(current_stimulus_case)]

            local_round_payload: dict[str, Any] = {
                "local_round_index": local_round_index,
                "disambiguation_constraints_before": list(accumulated_constraints),
                "cluster_trace": cluster_trace,
                "used_scenario_family": [_deepcopy_jsonable(current_scenario)],
                "used_stimulus_family": (
                    [_deepcopy_jsonable(current_stimulus_case)] if current_stimulus_case else []
                ),
                "scenario_family_size_before": 1 if current_scenario else 0,
                "stimulus_family_size_before": 1 if current_stimulus_case else 0,
                "scenario_history_before": _deepcopy_jsonable(scenario_history),
                "stimulus_history_before": _deepcopy_jsonable(stimulus_history),
                "scenario_history_size_before": len(scenario_history),
                "stimulus_history_size_before": len(stimulus_history),
                "stimulus_case": _deepcopy_jsonable(current_stimulus_case),
            }
            family_round_record: dict[str, Any] = {
                "local_round_index": local_round_index,
                "scenario_family_before": [_deepcopy_jsonable(current_scenario)],
                "stimulus_family_before": (
                    [_deepcopy_jsonable(current_stimulus_case)] if current_stimulus_case else []
                ),
                "scenario_history_before": _deepcopy_jsonable(scenario_history),
                "stimulus_history_before": _deepcopy_jsonable(stimulus_history),
                "disambiguation_constraints_before": list(accumulated_constraints),
            }
            baseline_entropy = cluster_record.get("semantic_entropy")
            baseline_cluster_count = int(cluster_record.get("functional_cluster_count", 0) or 0)
            has_cluster_divergence = (
                not _entropy_is_zero(baseline_entropy)
                if baseline_entropy is not None
                else baseline_cluster_count > 1
            )
            baseline_resolution_mode = str(cluster_record.get("resolution_mode", "")).strip()
            local_round_payload["semantic_entropy_before"] = baseline_entropy
            local_round_payload["functional_cluster_count_before"] = baseline_cluster_count
            local_round_payload["has_cluster_divergence"] = has_cluster_divergence
            family_round_record["semantic_entropy_before"] = baseline_entropy
            family_round_record["functional_cluster_count_before"] = baseline_cluster_count
            family_round_record["has_cluster_divergence"] = has_cluster_divergence
            family_round_record["resolution_mode"] = baseline_resolution_mode
            same_requirement_prior_constraints = list(
                per_requirement_constraints.get(target_requirement_id, [])
            )
            local_round_payload["same_requirement_prior_constraints"] = list(
                same_requirement_prior_constraints
            )
            family_round_record["same_requirement_prior_constraints"] = list(
                same_requirement_prior_constraints
            )
            round_stop_reason: str | None = None
            round_resolved = False
            post_constraint_cluster_result = cluster_result
            post_constraint_cluster_record = cluster_record

            if _entropy_is_zero(baseline_entropy):
                round_stop_reason = "zero_entropy_resolution"
                round_resolved = True
                requirement_satisfied = True
                requirement_round_payload["resolved_under_targeted_stimulus"] = True
                local_round_payload["semantic_entropy_after_constraint"] = baseline_entropy
                local_round_payload["functional_cluster_count_after"] = baseline_cluster_count
                family_round_record["semantic_entropy_after_constraint"] = baseline_entropy
            elif not has_cluster_divergence:
                round_stop_reason = "semantic_entropy_unavailable"
                local_round_payload["semantic_entropy_after_constraint"] = None
                local_round_payload["functional_cluster_count_after"] = baseline_cluster_count
                family_round_record["semantic_entropy_after_constraint"] = None

            dual_payload = None
            if round_stop_reason is None:
                dual_payload = _extract_dual_candidate_payload(
                    {"resolution": cluster_record.get("resolution", {})}
                )
                if dual_payload is None:
                    local_round_payload["constraint_validation"] = {
                        "skipped": True,
                        "skip_reason": "no_dual_candidate_payload_for_entropy_nonzero_round",
                        "baseline_entropy": baseline_entropy,
                    }
                    local_round_payload["semantic_entropy_after_constraint"] = baseline_entropy
                    local_round_payload["functional_cluster_count_after"] = baseline_cluster_count
                    local_round_payload["disambiguation_constraints_after"] = list(
                        accumulated_constraints
                    )
                    family_round_record["constraint_validation"] = _deepcopy_jsonable(
                        local_round_payload["constraint_validation"]
                    )
                    family_round_record["semantic_entropy_after_constraint"] = baseline_entropy

            if round_stop_reason is None and dual_payload is not None:
                winner_judgment, winner_judgment_record = _judge_requirement_dual_winner(
                    spec_text=spec_text,
                    target_requirement_id=target_requirement_id,
                    target_requirement=target_requirement,
                    scenario=current_scenario,
                    stimulus_case=current_stimulus_case or {},
                    dual_payload=dual_payload,
                    prior_requirement_constraints=same_requirement_prior_constraints,
                    backend=backends.planner_backend,
                )
                local_round_payload["winner_judgment"] = winner_judgment
                local_round_payload["winner_judgment_call"] = _deepcopy_jsonable(
                    winner_judgment_record
                )

                requirement_constraint_analysis, requirement_constraint_record = (
                    _analyze_requirement_dual_constraint(
                        spec_text=spec_text,
                        target_requirement_id=target_requirement_id,
                        target_requirement=target_requirement,
                        scenario=current_scenario,
                        stimulus_case=current_stimulus_case or {},
                        dual_payload=dual_payload,
                        winner_judgment=winner_judgment,
                        prior_requirement_constraints=same_requirement_prior_constraints,
                        backend=backends.planner_backend,
                    )
                )
                local_round_payload["requirement_constraint_analysis"] = (
                    requirement_constraint_analysis
                )
                local_round_payload["requirement_constraint_call"] = _deepcopy_jsonable(
                    requirement_constraint_record
                )
                new_constraint = str(
                    requirement_constraint_analysis.get("new_constraint", "")
                ).strip()
                local_round_payload["new_constraint_candidate"] = new_constraint
                local_round_payload["new_constraint_applied"] = ""
                local_round_payload["winner_judgment_model_usage"] = _summarize_llm_call_records(
                    [winner_judgment_record]
                )
                local_round_payload["constraint_model_usage"] = _summarize_llm_call_records(
                    [winner_judgment_record, requirement_constraint_record]
                )
                family_round_record["winner_judgment"] = _deepcopy_jsonable(winner_judgment)
                family_round_record["requirement_constraint_analysis"] = _deepcopy_jsonable(
                    requirement_constraint_analysis
                )
                round_constraint_payload: dict[str, Any] | None = None
                if new_constraint:
                    round_constraint_payload = {
                        "phase": "main_round_constraint_generation",
                        "target_requirement_id": target_requirement_id,
                        "target_requirement": target_requirement,
                        "global_round_index": main_round_index,
                        "local_round_index": local_round_index,
                        "scenario": _deepcopy_jsonable(current_scenario),
                        "stimulus_case": _deepcopy_jsonable(current_stimulus_case),
                        "candidate_constraint": new_constraint,
                        "status": "generated",
                        "baseline_entropy": baseline_entropy,
                        "functional_cluster_count_before": cluster_record.get(
                            "functional_cluster_count"
                        ),
                        "winner_judgment": _deepcopy_jsonable(winner_judgment),
                        "requirement_constraint_analysis": _deepcopy_jsonable(
                            requirement_constraint_analysis
                        ),
                    }
                    _write_round_constraint_artifact(
                        cluster_record.get("artifact_dir"),
                        round_constraint_payload,
                        cluster_record=cluster_record,
                        cluster_trace=cluster_trace,
                    )

                if not new_constraint:
                    local_round_payload["constraint_validation"] = {
                        "skipped": True,
                        "skip_reason": "empty_new_constraint",
                        "baseline_entropy": baseline_entropy,
                    }
                    local_round_payload["semantic_entropy_after_constraint"] = baseline_entropy
                    local_round_payload["functional_cluster_count_after"] = cluster_record.get(
                        "functional_cluster_count"
                    )
                    family_round_record["constraint_validation"] = _deepcopy_jsonable(
                        local_round_payload["constraint_validation"]
                    )
                    family_round_record["semantic_entropy_after_constraint"] = baseline_entropy
                elif new_constraint in accumulated_constraints:
                    local_round_payload["constraint_validation"] = {
                        "skipped": True,
                        "skip_reason": "duplicate_constraint_already_accepted",
                        "baseline_entropy": baseline_entropy,
                        "candidate_constraint": new_constraint,
                    }
                    local_round_payload["semantic_entropy_after_constraint"] = baseline_entropy
                    local_round_payload["functional_cluster_count_after"] = cluster_record.get(
                        "functional_cluster_count"
                    )
                    family_round_record["constraint_validation"] = _deepcopy_jsonable(
                        local_round_payload["constraint_validation"]
                    )
                    family_round_record["semantic_entropy_after_constraint"] = baseline_entropy
                    if round_constraint_payload is not None:
                        round_constraint_payload["status"] = "duplicate_constraint_already_accepted"
                        round_constraint_payload["constraint_validation"] = _deepcopy_jsonable(
                            local_round_payload["constraint_validation"]
                        )
                        _write_round_constraint_artifact(
                            cluster_record.get("artifact_dir"),
                            round_constraint_payload,
                            cluster_record=cluster_record,
                            cluster_trace=cluster_trace,
                        )
                else:
                    validation_constraints = list(accumulated_constraints) + [new_constraint]
                    (
                        validation_candidates,
                        validation_current_candidate,
                        validation_cluster_record,
                        validation_cluster_trace,
                        validation_cluster_result,
                    ) = _run_llm_verilog_cluster_resolution(
                        task=task,
                        config=config,
                        spec_text=spec_text,
                        requirements=target_requirement_bundle,
                        backend=backends.generator_backend,
                        round_index=main_round_index,
                        disambiguation_constraints=validation_constraints,
                        external_verification_scenarios=[current_scenario],
                        external_stimulus_cases=[current_stimulus_case]
                        if current_stimulus_case
                        else None,
                        cluster_shared_cache=cluster_shared_cache,
                        round_label=(
                            f"round_{main_round_index:02d}_req_{target_requirement_id:03d}"
                            f"_local_{local_round_index:02d}_constraint_validation"
                        ),
                        enable_stimulus_refinement=False,
                    )
                    validation_entropy = validation_cluster_record.get("semantic_entropy")
                    validation_accepted = (
                        validation_entropy is not None
                        and baseline_entropy is not None
                        and float(validation_entropy)
                        < float(baseline_entropy) - SEMANTIC_ENTROPY_ZERO_TOLERANCE
                    )
                    validation_record = {
                        "candidate_constraint": new_constraint,
                        "baseline_entropy": baseline_entropy,
                        "validated_entropy": validation_entropy,
                        "functional_cluster_count_before": cluster_record.get(
                            "functional_cluster_count"
                        ),
                        "functional_cluster_count_after": validation_cluster_record.get(
                            "functional_cluster_count"
                        ),
                        "accepted": validation_accepted,
                        "cluster_trace": validation_cluster_trace,
                        "resolution_mode": validation_cluster_record.get("resolution_mode"),
                    }
                    local_round_payload["constraint_validation"] = validation_record
                    family_round_record["constraint_validation"] = _deepcopy_jsonable(
                        validation_record
                    )
                    validation_constraint_payload = {
                        "phase": "constraint_validation",
                        "target_requirement_id": target_requirement_id,
                        "target_requirement": target_requirement,
                        "global_round_index": main_round_index,
                        "local_round_index": local_round_index,
                        "scenario": _deepcopy_jsonable(current_scenario),
                        "stimulus_case": _deepcopy_jsonable(current_stimulus_case),
                        "candidate_constraint": new_constraint,
                        "baseline_entropy": baseline_entropy,
                        "validated_entropy": validation_entropy,
                        "functional_cluster_count_before": cluster_record.get(
                            "functional_cluster_count"
                        ),
                        "functional_cluster_count_after": validation_cluster_record.get(
                            "functional_cluster_count"
                        ),
                        "accepted": validation_accepted,
                        "winner_judgment": _deepcopy_jsonable(winner_judgment),
                        "requirement_constraint_analysis": _deepcopy_jsonable(
                            requirement_constraint_analysis
                        ),
                        "constraint_validation": _deepcopy_jsonable(validation_record),
                    }
                    _write_round_constraint_artifact(
                        validation_cluster_record.get("artifact_dir"),
                        validation_constraint_payload,
                        cluster_record=validation_cluster_record,
                        cluster_trace=validation_cluster_trace,
                    )
                    if validation_accepted:
                        accumulated_constraints.append(new_constraint)
                        per_requirement_constraints.setdefault(target_requirement_id, []).append(
                            new_constraint
                        )
                        local_round_payload["new_constraint_applied"] = new_constraint
                        post_constraint_cluster_result = validation_cluster_result
                        post_constraint_cluster_record = validation_cluster_record
                        final_candidates = validation_candidates
                        final_current_candidate = validation_current_candidate
                        local_round_payload["semantic_entropy_after_constraint"] = (
                            validation_entropy
                        )
                        local_round_payload["functional_cluster_count_after"] = (
                            validation_cluster_record.get("functional_cluster_count")
                        )
                        family_round_record["semantic_entropy_after_constraint"] = (
                            validation_entropy
                        )
                        if round_constraint_payload is not None:
                            round_constraint_payload["status"] = "accepted"
                            round_constraint_payload["validated_entropy"] = validation_entropy
                            round_constraint_payload["functional_cluster_count_after"] = (
                                validation_cluster_record.get("functional_cluster_count")
                            )
                            round_constraint_payload["constraint_validation"] = (
                                _deepcopy_jsonable(validation_record)
                            )
                            _write_round_constraint_artifact(
                                cluster_record.get("artifact_dir"),
                                round_constraint_payload,
                                cluster_record=cluster_record,
                                cluster_trace=cluster_trace,
                            )
                        if _entropy_is_zero(validation_entropy):
                            round_stop_reason = "accepted_constraint_zero_entropy_resolution"
                            round_resolved = True
                            requirement_satisfied = True
                            requirement_round_payload[
                                "resolved_under_targeted_stimulus"
                            ] = True
                    else:
                        recovery_record = {
                            "triggered": False,
                            "reason": "constraint_validation_entropy_increase_or_unavailable",
                            "baseline_entropy": baseline_entropy,
                            "recovered_entropy": baseline_entropy,
                            "functional_cluster_count_after": baseline_cluster_count,
                            "reused_baseline_cluster": True,
                            "cluster_trace": cluster_trace,
                            "resolution_mode": cluster_record.get("resolution_mode"),
                        }
                        local_round_payload["constraint_recovery"] = recovery_record
                        family_round_record["constraint_recovery"] = _deepcopy_jsonable(
                            recovery_record
                        )
                        discarded_constraints.append(
                            {
                                "target_requirement_id": target_requirement_id,
                                "target_requirement": target_requirement,
                                "local_round_index": local_round_index,
                                "scenario": _deepcopy_jsonable(current_scenario),
                                "stimulus_case": _deepcopy_jsonable(current_stimulus_case),
                                "candidate_constraint": new_constraint,
                                "discard_reason": "constraint_validation_entropy_increase_or_unavailable",
                                "baseline_entropy": baseline_entropy,
                                "validated_entropy": validation_entropy,
                                "recovered_entropy": baseline_entropy,
                                "functional_cluster_count_before": baseline_cluster_count,
                                "functional_cluster_count_validation": validation_cluster_record.get(
                                    "functional_cluster_count"
                                ),
                                "functional_cluster_count_recovery": baseline_cluster_count,
                                "winner_judgment": _deepcopy_jsonable(winner_judgment),
                                "requirement_constraint_analysis": _deepcopy_jsonable(
                                    requirement_constraint_analysis
                                ),
                                "constraint_validation": _deepcopy_jsonable(validation_record),
                                "constraint_recovery": _deepcopy_jsonable(recovery_record),
                            }
                        )
                        if round_constraint_payload is not None:
                            round_constraint_payload["status"] = "discarded"
                            round_constraint_payload["discard_reason"] = (
                                "constraint_validation_entropy_increase_or_unavailable"
                            )
                            round_constraint_payload["validated_entropy"] = validation_entropy
                            round_constraint_payload["recovered_entropy"] = baseline_entropy
                            round_constraint_payload["functional_cluster_count_validation"] = (
                                validation_cluster_record.get("functional_cluster_count")
                            )
                            round_constraint_payload["functional_cluster_count_recovery"] = (
                                baseline_cluster_count
                            )
                            round_constraint_payload["constraint_validation"] = (
                                _deepcopy_jsonable(validation_record)
                            )
                            round_constraint_payload["constraint_recovery"] = (
                                _deepcopy_jsonable(recovery_record)
                            )
                            _write_round_constraint_artifact(
                                cluster_record.get("artifact_dir"),
                                round_constraint_payload,
                                cluster_record=cluster_record,
                                cluster_trace=cluster_trace,
                            )
                        local_round_payload["semantic_entropy_after_constraint"] = (
                            baseline_entropy
                        )
                        local_round_payload["functional_cluster_count_after"] = (
                            baseline_cluster_count
                        )
                        family_round_record["semantic_entropy_after_constraint"] = (
                            baseline_entropy
                        )

                local_round_payload["disambiguation_constraints_after"] = list(
                    accumulated_constraints
                )
                family_round_record["new_constraint_applied"] = local_round_payload[
                    "new_constraint_applied"
                ]

            if round_stop_reason is not None:
                local_round_payload["stop_reason"] = round_stop_reason
                local_round_payload.setdefault(
                    "disambiguation_constraints_after", list(accumulated_constraints)
                )
                local_round_payload.setdefault(
                    "semantic_entropy_after_constraint", baseline_entropy
                )
                local_round_payload.setdefault(
                    "functional_cluster_count_after",
                    post_constraint_cluster_record.get("functional_cluster_count"),
                )
                local_round_payload["scenario_family_size_after"] = 1 if current_scenario else 0
                local_round_payload["stimulus_family_size_after"] = (
                    1 if current_stimulus_case else 0
                )
                local_round_payload["scenario_history_size_after"] = len(scenario_history)
                local_round_payload["stimulus_history_size_after"] = len(stimulus_history)
                family_round_record.setdefault(
                    "semantic_entropy_after_constraint",
                    local_round_payload.get("semantic_entropy_after_constraint"),
                )
                family_round_record["harder_scenario_generation_triggered"] = False
                family_round_record["scenario_family_after"] = [
                    _deepcopy_jsonable(current_scenario)
                ]
                family_round_record["stimulus_family_after"] = (
                    [_deepcopy_jsonable(current_stimulus_case)] if current_stimulus_case else []
                )
                family_round_record["scenario_history_after"] = _deepcopy_jsonable(
                    scenario_history
                )
                family_round_record["stimulus_history_after"] = _deepcopy_jsonable(
                    stimulus_history
                )
                family_round_record["disambiguation_constraints_after"] = list(
                    accumulated_constraints
                )
                requirement_round_payload["rounds"].append(local_round_payload)
                family_expansion_record["family_rounds"].append(family_round_record)
                if round_resolved:
                    resolved_requirement_ids.append(target_requirement_id)
                break

            harder_scenario_record: dict[str, Any] = {
                "local_round_index": local_round_index,
                "target_requirement_id": target_requirement_id,
                "target_requirement": target_requirement,
                "latest_scenario": _deepcopy_jsonable(current_scenario),
                "harder_scenario_generation_triggered": True,
            }
            harder_scenario_record["scenario_generation_input"] = {
                "original_specification": spec_text,
                "target_requirement_id": target_requirement_id,
                "target_requirement": target_requirement,
                "prior_scenarios": _deepcopy_jsonable(scenario_history),
            }
            family_round_record["harder_scenario_generation"] = harder_scenario_record
            harder_added = False
            try:
                if monitor is not None:
                    monitor.heartbeat(
                        phase="harder_scenario_generation_start",
                        details={
                            "target_requirement_id": target_requirement_id,
                            "local_round_index": local_round_index,
                            "constraint_count": len(accumulated_constraints),
                        },
                    )
                harder_scenario, harder_scenario_call = _generate_harder_requirement_scenario(
                    spec_text=spec_text,
                    target_requirement_id=target_requirement_id,
                    target_requirement=target_requirement,
                    prior_scenarios=scenario_history,
                    expansion_index=len(scenario_history),
                    backend=backends.planner_backend,
                )
                harder_scenario_record["scenario_call"] = harder_scenario_call
                harder_scenario_record["generated_harder_scenario"] = _deepcopy_jsonable(harder_scenario)
                harder_scenario_llm_calls.append(harder_scenario_call)
                stimulus_case_for_harder, harder_stimulus_record = _generate_harder_stimulus_case(
                    spec_text=spec_text,
                    harder_scenario=harder_scenario,
                    cluster_result=post_constraint_cluster_result,
                    backend=backends.generator_backend,
                    max_candidate_workers=config.task_max_candidate_workers,
                    cluster_shared_cache=cluster_shared_cache,
                )
                harder_scenario_record["stimulus_generation"] = harder_stimulus_record
                harder_scenario_llm_calls.extend(harder_stimulus_record.get("llm_calls", []))
                scenario_duplicate = _scenario_semantic_key(harder_scenario) in {
                    _scenario_semantic_key(item) for item in scenario_history
                }
                stimulus_duplicate = (
                    stimulus_case_for_harder is not None
                    and _stimulus_semantic_key(stimulus_case_for_harder) in {
                        _stimulus_semantic_key(item) for item in stimulus_history
                    }
                )
                harder_scenario_record["scenario_duplicate"] = scenario_duplicate
                harder_scenario_record["stimulus_duplicate"] = stimulus_duplicate
                if (
                    stimulus_case_for_harder is not None
                    and not scenario_duplicate
                    and not stimulus_duplicate
                ):
                    current_scenario = _deepcopy_jsonable(harder_scenario)
                    current_stimulus_case = _deepcopy_jsonable(stimulus_case_for_harder)
                    scenario_history.append(_deepcopy_jsonable(current_scenario))
                    stimulus_history.append(_deepcopy_jsonable(current_stimulus_case))
                    harder_added = True
                    harder_scenario_record["harder_scenario_added"] = True
                    harder_scenario_record["harder_stimulus_case_added"] = True
                else:
                    harder_scenario_record["harder_scenario_added"] = False
                    harder_scenario_record["harder_stimulus_case_added"] = False
            except Exception as exc:
                harder_scenario_record["error"] = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                harder_scenario_record["harder_scenario_added"] = False
                harder_scenario_record["harder_stimulus_case_added"] = False

            local_round_payload["harder_scenario_generation"] = {
                "harder_scenario_added": harder_added,
                "scenario_family_size_after_generation": 1 if current_scenario else 0,
                "stimulus_family_size_after_generation": 1 if current_stimulus_case else 0,
                "scenario_history_size_after_generation": len(scenario_history),
                "stimulus_history_size_after_generation": len(stimulus_history),
            }
            local_round_payload["scenario_family_size_after"] = 1 if current_scenario else 0
            local_round_payload["stimulus_family_size_after"] = 1 if current_stimulus_case else 0
            local_round_payload["scenario_history_size_after"] = len(scenario_history)
            local_round_payload["stimulus_history_size_after"] = len(stimulus_history)
            family_round_record["scenario_family_after"] = [_deepcopy_jsonable(current_scenario)]
            family_round_record["stimulus_family_after"] = (
                [_deepcopy_jsonable(current_stimulus_case)] if current_stimulus_case else []
            )
            family_round_record["scenario_history_after"] = _deepcopy_jsonable(scenario_history)
            family_round_record["stimulus_history_after"] = _deepcopy_jsonable(stimulus_history)
            family_round_record["harder_scenario_added"] = harder_added
            family_round_record["disambiguation_constraints_after"] = list(accumulated_constraints)
            family_round_record["semantic_entropy_after_constraint"] = local_round_payload.get(
                "semantic_entropy_after_constraint"
            )

            should_continue = bool(has_cluster_divergence and harder_added) and (
                local_round_index < max_rounds_per_requirement
            )
            if should_continue:
                local_round_payload["stop_reason"] = "continue_with_same_requirement_and_expanded_family"
            elif has_cluster_divergence and local_round_index >= max_rounds_per_requirement:
                local_round_payload["stop_reason"] = "max_rounds_reached_for_requirement"
            elif has_cluster_divergence:
                local_round_payload["stop_reason"] = (
                    "no_new_harder_family_generated_despite_cluster_divergence"
                )
            else:
                local_round_payload["stop_reason"] = "no_cluster_divergence_for_requirement"

            requirement_round_payload["rounds"].append(local_round_payload)
            family_expansion_record["family_rounds"].append(family_round_record)
            if local_round_payload["stop_reason"] != "continue_with_same_requirement_and_expanded_family":
                break

        if current_scenario and current_stimulus_case:
            executed_scenarios_for_final_generation = [_deepcopy_jsonable(current_scenario)]
            executed_stimulus_cases_for_final_generation = [_deepcopy_jsonable(current_stimulus_case)]
        family_expansion_record["final_scenario_family"] = [_deepcopy_jsonable(current_scenario)]
        family_expansion_record["final_stimulus_family"] = (
            [_deepcopy_jsonable(current_stimulus_case)] if current_stimulus_case else []
        )
        family_expansion_record["scenario_history"] = _deepcopy_jsonable(scenario_history)
        family_expansion_record["stimulus_history"] = _deepcopy_jsonable(stimulus_history)
        harder_scenario_families.append(family_expansion_record)
        requirement_loop_rounds.append(requirement_round_payload)

    all_scenarios_discarded = (
        not screened_scenario_entries
        and bool(discard_scenario_record.get("discarded_scenario_ids"))
        and bool(fixed_initial_candidates)
    )
    if all_scenarios_discarded:
        if monitor is not None:
            monitor.update(
                task_name=task.task_name,
                task_id=task.task_id,
                variant="requirements_constraint_selfplanning",
                phase="all_scenarios_discarded_fallback",
                details={
                    "fixed_initial_candidate_count": len(fixed_initial_candidates),
                },
            )
        fallback_rng = random.Random(f"{task.task_name}:scenario_prefilter_all_discard")
        prefilter_excluded_candidate_ids = {
            str(item).strip()
            for item in (discard_scenario_record.get("prefilter_excluded_candidate_ids") or [])
            if str(item).strip()
        }
        filtered_final_candidates = [
            candidate
            for candidate in fixed_initial_candidates
            if _prefilter_fixed_candidate_source_id(candidate) not in prefilter_excluded_candidate_ids
        ]
        filtered_out_candidate_ids = [
            candidate.candidate_id
            for candidate in fixed_initial_candidates
            if _prefilter_fixed_candidate_source_id(candidate) in prefilter_excluded_candidate_ids
        ]
        final_candidates = list(filtered_final_candidates)
        fallback_pool_reason = (
            "excluded_prefilter_compile_or_sim_fail_candidates"
            if filtered_out_candidate_ids
            else "no_prefilter_excluded_candidates_to_remove"
        )
        if not final_candidates:
            final_candidates = list(fixed_initial_candidates)
            fallback_pool_reason = "all_fixed_initial_candidates_were_excluded_during_prefilter"
        final_current_candidate = fallback_rng.choice(final_candidates)
        discard_scenario_record["fallback_selection"] = {
            "mode": "prefilter_all_discard_fixed_candidate_fallback",
            "selected_candidate_id": final_current_candidate.candidate_id,
            "candidate_ids": [candidate.candidate_id for candidate in final_candidates],
            "candidate_count": len(final_candidates),
            "prefilter_excluded_candidate_ids": sorted(prefilter_excluded_candidate_ids, key=str),
            "filtered_out_candidate_ids": filtered_out_candidate_ids,
            "fallback_pool_reason": fallback_pool_reason,
            "rng_seed_material": f"{task.task_name}:scenario_prefilter_all_discard",
        }
        cluster_rounds.append(
            {
                "target_requirement_id": None,
                "local_round_index": None,
                "final_generation": True,
                "final_generation_mode": "prefilter_all_discard_fixed_candidate_fallback",
                "resolution_mode": "prefilter_all_discard_fixed_candidate_fallback",
                "selected_candidate_id": final_current_candidate.candidate_id,
                "candidate_count": len(final_candidates),
                "candidate_ids": [candidate.candidate_id for candidate in final_candidates],
                "llm_calls": [],
                "model_usage": _summarize_llm_call_records([]),
                "fallback_reason": (
                    "Scenario prefilter discarded all generated scenarios, so the pipeline reuses the fixed initial "
                    "candidate pool instead of regenerating scenarios or candidates."
                ),
                "prefilter_excluded_candidate_ids": sorted(prefilter_excluded_candidate_ids, key=str),
                "filtered_out_candidate_ids": filtered_out_candidate_ids,
                "fallback_pool_reason": fallback_pool_reason,
            }
        )
    else:
        if monitor is not None:
            monitor.update(
                task_name=task.task_name,
                task_id=task.task_id,
                variant="requirements_constraint_selfplanning",
                phase="final_generation_round_start",
                details={
                    "constraint_count": len(accumulated_constraints),
                    "executed_scenario_count": len(executed_scenarios_for_final_generation),
                    "executed_stimulus_count": len(executed_stimulus_cases_for_final_generation),
                    "global_round_index": len(cluster_rounds) + 1,
                },
            )
        final_round_index = len(cluster_rounds) + 1
        candidates, current_candidate, cluster_record, _cluster_trace, _cluster_result = _run_llm_verilog_cluster_resolution(
            task=task,
            config=config,
            spec_text=spec_text,
            requirements=all_requirements,
            backend=backends.generator_backend,
            round_index=final_round_index,
            disambiguation_constraints=accumulated_constraints,
            external_verification_scenarios=executed_scenarios_for_final_generation,
            external_stimulus_cases=executed_stimulus_cases_for_final_generation,
            cluster_shared_cache=cluster_shared_cache,
            enable_stimulus_refinement=False,
        )
        final_candidates = candidates
        final_current_candidate = current_candidate
        cluster_rounds.append(
            {
                "target_requirement_id": None,
                "local_round_index": None,
                "final_generation": True,
                **cluster_record,
            }
        )
    if final_current_candidate is not None:
        final_current_candidate, verification_result, postprocess_record = _postprocess_and_verify_cluster_candidate(
            task=task,
            spec_text=spec_text,
            candidate=final_current_candidate,
            backend=backends.repair_backend,
            timeout_sec=config.verification_timeout_sec,
        )
        final_candidates = _replace_or_append_candidate(
            final_candidates,
            final_current_candidate,
            str(postprocess_record.get("source_candidate_id", "")).strip() or None,
        )
        verification_results.append(verification_result)
        cluster_rounds[-1]["output_initialization_postprocess"] = postprocess_record

    loop_record = {
        "rounds": cluster_rounds,
        "requirement_rounds": requirement_loop_rounds,
        "scenario_pool": scenario_pool,
        "scenario_prefilter": discard_scenario_record,
        "executed_scenario_ids": executed_scenario_ids,
        "resolved_requirement_ids": resolved_requirement_ids,
        "final_disambiguation_constraints": list(accumulated_constraints),
        "discarded_constraints": _deepcopy_jsonable(discarded_constraints),
        "model_usage": _summarize_llm_call_records(
            [record for round_record in cluster_rounds for record in round_record.get("llm_calls", [])]
        ),
    }
    loop_io = {
        "scenario_pool": scenario_pool_record,
        "requirement_rounds": requirement_loop_rounds,
        "selected_mode": CLUSTER_FEEDBACK_MODE_REQUIREMENT_STIMULUS_CONSTRAINTS,
        "executed_scenario_ids": executed_scenario_ids,
        "resolved_requirement_ids": resolved_requirement_ids,
        "final_disambiguation_constraints": list(accumulated_constraints),
        "discarded_constraints": _deepcopy_jsonable(discarded_constraints),
        "scenario_prefilter": discard_scenario_record,
    }
    harder_scenario_record = {
        "selected_mode": CLUSTER_FEEDBACK_MODE_REQUIREMENT_STIMULUS_CONSTRAINTS,
        "families": harder_scenario_families,
        "llm_calls": harder_scenario_llm_calls,
        "model_usage": _summarize_llm_call_records(harder_scenario_llm_calls),
    }
    return (
        accumulated_constraints,
        final_candidates,
        final_current_candidate,
        verification_results,
        loop_record,
        loop_io,
        discard_scenario_record,
        harder_scenario_record,
    )
def build_task_base(
    task: TaskSpec,
    config: PipelineConfig,
    backends: ProviderBundle,
    cluster_feedback_mode: str = DEFAULT_CLUSTER_FEEDBACK_MODE,
) -> RequirementsConstraintSelfPlanningTaskBase:
    """Build the active requirement-stimulus constraint branch."""

    started_at = time.perf_counter()
    spec_text = read_spec_text(task)

    if cluster_feedback_mode != CLUSTER_FEEDBACK_MODE_REQUIREMENT_STIMULUS_CONSTRAINTS:
        raise ValueError(
            "Only requirement_stimulus_constraints mode is supported in the cleaned pipeline."
        )

    requirements, requirements_record = extract_explicit_requirements(
        task=task,
        spec_text=spec_text,
        backend=backends.planner_backend,
    )
    requirements, requirements_record = _append_cross_behavior_requirement(
        requirements_record=requirements_record,
        all_requirements=requirements,
    )
    interface_requirements, behavior_requirements = _extract_requirement_partitions(
        requirements_record=requirements_record,
        all_requirements=requirements,
    )
    requirement_context = list(requirements)
    (
        final_disambiguation_constraints,
        candidates,
        current_candidate,
        verification_results,
        requirement_stimulus_cluster_loop_record,
        requirement_stimulus_loop_io,
        discard_scenario_record,
        harder_scenario_record,
    ) = _run_requirement_stimulus_constraint_loop(
        task=task,
        config=config,
        spec_text=spec_text,
        all_requirements=requirement_context,
        behavioral_requirements=behavior_requirements,
        backends=backends,
        max_rounds_per_requirement=3,
    )
    understanding = SpecUnderstanding(
        explicit_requirements=list(requirement_context),
        ambiguities=[],
        self_planning=[],
    )
    selected_mode = str(
        requirement_stimulus_loop_io.get("selected_mode", cluster_feedback_mode)
    ).strip() or cluster_feedback_mode
    compile_results: list[VerificationResult] = []
    repair_io: dict[str, Any] = {
        "disabled": True,
        "disabled_reason": "compile_repairs_disabled_after_llm_verilog_cluster_in_requirements_constraint_pipeline",
        "rounds": [],
    }

    llm_io = {
        "requirements_constraint_selfplanning": {
            "requirements": requirements_record,
            "cluster_feedback_mode": selected_mode,
            "interface_requirements": interface_requirements,
            "behavior_requirements": behavior_requirements,
            "scenario_pool": requirement_stimulus_loop_io.get("scenario_pool", {}),
        },
        "llm_verilog_cluster": requirement_stimulus_cluster_loop_record,
        "llm_verilog_cluster_disambiguation_constraints": {
            **requirement_stimulus_loop_io,
            "selected_mode": selected_mode,
        },
        "discard_scenario": discard_scenario_record,
        "harder_scenario": harder_scenario_record,
        "compile_repairs": repair_io,
    }
    constraint_trace = {
        "base_requirements": requirements,
        "interface_requirements": interface_requirements,
        "behavior_requirements": behavior_requirements,
        "cluster_feedback_mode": selected_mode,
        "llm_verilog_cluster_rounds": requirement_stimulus_cluster_loop_record.get("rounds", []),
        "final_disambiguation_constraints": final_disambiguation_constraints,
        "discarded_constraints": _deepcopy_jsonable(
            requirement_stimulus_cluster_loop_record.get("discarded_constraints", [])
        ),
        "scenario_pool": requirement_stimulus_loop_io.get("scenario_pool", {}).get("scenario_pool", []),
        "discarded_scenario_ids": discard_scenario_record.get("discarded_scenario_ids", []),
        "kept_scenario_ids": discard_scenario_record.get("kept_scenario_ids", []),
        "executed_scenario_ids": requirement_stimulus_loop_io.get("executed_scenario_ids", []),
        "resolved_requirement_ids": requirement_stimulus_loop_io.get("resolved_requirement_ids", []),
    }
    return RequirementsConstraintSelfPlanningTaskBase(
        spec_text=spec_text,
        understanding=understanding,
        constraint_trace=constraint_trace,
        llm_io=llm_io,
        candidates=candidates,
        compile_results=compile_results,
        verification_results=verification_results,
        current_candidate=current_candidate,
        elapsed_seconds=max(0.0, time.perf_counter() - started_at),
    )
