"""Trace and summary writers for per-task artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .models import PipelineTrace, TaskSpec


def write_trace(trace_root: Path, trace: PipelineTrace) -> Path:
    """Persist one task trace as readable JSON."""

    output_dir = trace_root / trace.pipeline_variant / trace.task.task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_task_content(output_dir, trace.task)
    write_candidate_artifacts(output_dir, trace)
    trace.llm_io_files = write_llm_io(output_dir, trace.llm_io)
    output_path = output_dir / "trace.json"
    output_path.write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_llm_io(task_output_dir: Path, llm_io: dict) -> dict[str, str]:
    """Persist grouped LLM prompts/responses outside of trace.json."""

    if not llm_io:
        return {}

    llm_dir = task_output_dir / "llm_io"
    if llm_dir.exists():
        shutil.rmtree(llm_dir)
    llm_dir.mkdir(parents=True, exist_ok=True)

    file_order = [
        ("understanding", "001_understanding.json"),
        ("spec_direct_baseline", "001b_spec_direct_baseline.json"),
        ("direct_with_constraint", "001c_direct_with_constraint.json"),
        ("selfplanning_only", "001d_selfplanning_only.json"),
        ("direct_few_shots", "001e_direct_few_shots.json"),
        ("requirements_constraint_selfplanning", "003_requirements_constraint_selfplanning.json"),
        ("generation", "005_generation.json"),
        ("llm_verilog_cluster", "005b_llm_verilog_cluster.json"),
        ("llm_verilog_cluster_disambiguation_constraints", "005e_llm_verilog_cluster_disambiguation_constraints.json"),
        ("discard_scenario", "005ea_discard_scenario.json"),
        ("harder_scenario", "harder_scenario.json"),
        ("llm_verilog_cluster_winner_spec_gap", "005f_llm_verilog_cluster_winner_spec_gap.json"),
    ]
    written: dict[str, str] = {}
    for key, filename in file_order:
        if key not in llm_io:
            continue
        path = llm_dir / filename
        path.write_text(
            json.dumps(llm_io[key], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written[key] = str(path.relative_to(task_output_dir))
        if key == "llm_verilog_cluster_disambiguation_constraints":
            summary_path = llm_dir / "005e_llm_verilog_cluster_disambiguation_constraints_summary.json"
            summary_path.write_text(
                json.dumps(
                    _build_disambiguation_constraints_summary(llm_io[key]),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_summary"] = str(summary_path.relative_to(task_output_dir))
            provenance_path = llm_dir / "005e_llm_verilog_cluster_disambiguation_constraints_provenance.json"
            provenance_path.write_text(
                json.dumps(
                    _build_disambiguation_constraints_provenance(llm_io[key]),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_provenance"] = str(provenance_path.relative_to(task_output_dir))
            round_bundle_path = llm_dir / "005e_llm_verilog_cluster_requirement_round_bundle.json"
            round_bundle_path.write_text(
                json.dumps(
                    _build_requirement_round_bundle(llm_io[key]),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_requirement_round_bundle"] = str(
                round_bundle_path.relative_to(task_output_dir)
            )
            all_constraints_path = llm_dir / "005e_llm_verilog_cluster_all_generated_constraints.json"
            all_constraints_path.write_text(
                json.dumps(
                    _build_all_generated_constraints(llm_io[key]),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_all_generated_constraints"] = str(
                all_constraints_path.relative_to(task_output_dir)
            )
            discarded_constraints_path = (
                llm_dir / "005e_llm_verilog_cluster_discard_constraints_summary.json"
            )
            discarded_constraints_path.write_text(
                json.dumps(
                    _build_discarded_constraints_summary(llm_io[key]),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_discard_constraints_summary"] = str(
                discarded_constraints_path.relative_to(task_output_dir)
            )
            entropy_summary_path = llm_dir / "005e_llm_verilog_cluster_entropy_summary.json"
            entropy_summary_path.write_text(
                json.dumps(
                    _build_entropy_round_summary(
                        llm_io[key],
                        llm_io.get("llm_verilog_cluster"),
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written[f"{key}_entropy_summary"] = str(
                entropy_summary_path.relative_to(task_output_dir)
            )
    if "discard_scenario" in llm_io or "llm_verilog_cluster" in llm_io:
        stimulus_refinement_path = llm_dir / "005eb_stimulus_refinement_summary.json"
        stimulus_refinement_path.write_text(
            json.dumps(
                _build_stimulus_refinement_summary(
                    llm_io.get("discard_scenario"),
                    llm_io.get("llm_verilog_cluster"),
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written["stimulus_refinement_summary"] = str(
            stimulus_refinement_path.relative_to(task_output_dir)
        )
    if _llm_timeout_retry_triggered(llm_io):
        timeout_retry_path = llm_dir / "llm_timeout_retry_tasks.json"
        timeout_retry_path.write_text(
            json.dumps(
                {
                    "task_name": task_output_dir.name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written["llm_timeout_retry_tasks"] = str(timeout_retry_path.relative_to(task_output_dir))
    return written


def _build_disambiguation_constraints_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Compress the requirement-stimulus constraint loop into a human-readable summary."""

    requirement_rounds = payload.get("requirement_rounds")
    if not isinstance(requirement_rounds, list):
        return payload

    summary_rounds: list[dict[str, Any]] = []
    for requirement_round in requirement_rounds:
        if not isinstance(requirement_round, dict):
            continue
        local_rounds = requirement_round.get("rounds")
        summarized_local_rounds: list[dict[str, Any]] = []
        if isinstance(local_rounds, list):
            for local_round in local_rounds:
                if not isinstance(local_round, dict):
                    continue
                dual_choice = local_round.get("dual_choice") or {}
                decision = dual_choice.get("decision") if isinstance(dual_choice, dict) else {}
                analysis = local_round.get("requirement_constraint_analysis") or {}
                summarized_local_rounds.append(
                    {
                        "local_round_index": local_round.get("local_round_index"),
                        "selected_candidate": (
                            decision.get("selected_candidate")
                            if isinstance(decision, dict)
                            else local_round.get("selected_candidate")
                        ),
                        "winner_judgment": local_round.get("winner_judgment"),
                        "reason_based_on_spec": (
                            decision.get("reason_based_on_spec")
                            if isinstance(decision, dict)
                            else None
                        ),
                        "new_constraint_applied": local_round.get("new_constraint_applied") or "",
                        "stop_reason": local_round.get("stop_reason"),
                        "same_requirement_prior_constraints": local_round.get(
                            "same_requirement_prior_constraints", []
                        ),
                        "constraint_model_usage": local_round.get("constraint_model_usage"),
                        "winner_judgment_model_usage": local_round.get(
                            "winner_judgment_model_usage"
                        ),
                        "dual_choice_model_usage": local_round.get("dual_choice_model_usage"),
                        "divergence_analysis_based_on_spec": (
                            analysis.get("divergence_analysis_based_on_spec")
                            if isinstance(analysis, dict)
                            else None
                        ),
                    }
                )

        summary_rounds.append(
            {
                "target_requirement_id": requirement_round.get("target_requirement_id"),
                "target_requirement": requirement_round.get("target_requirement"),
                "scenario_id": (requirement_round.get("scenario") or {}).get("scenario_id")
                if isinstance(requirement_round.get("scenario"), dict)
                else None,
                "scenario_name": (requirement_round.get("scenario") or {}).get("name")
                if isinstance(requirement_round.get("scenario"), dict)
                else None,
                "resolved_under_targeted_stimulus": requirement_round.get(
                    "resolved_under_targeted_stimulus"
                ),
                "num_local_rounds": len(summarized_local_rounds),
                "local_rounds": summarized_local_rounds,
            }
        )

    return {
        "selected_mode": payload.get("selected_mode"),
        "executed_scenario_ids": payload.get("executed_scenario_ids", []),
        "resolved_requirement_ids": payload.get("resolved_requirement_ids", []),
        "final_disambiguation_constraints": payload.get("final_disambiguation_constraints", []),
        "requirement_rounds": summary_rounds,
    }


def _build_stimulus_refinement_summary(
    discard_payload: Any,
    cluster_payload: Any,
) -> dict[str, Any]:
    """Flatten per-task stimulus refinement before/after snapshots for plotting and audits."""

    prefilter_refinements: list[dict[str, Any]] = []
    cluster_round_refinements: list[dict[str, Any]] = []

    if isinstance(discard_payload, dict):
        screened_scenarios = discard_payload.get("screened_scenarios")
        if isinstance(screened_scenarios, list):
            for item in screened_scenarios:
                if not isinstance(item, dict):
                    continue
                refinement = item.get("stimulus_refinement")
                history = refinement.get("history") if isinstance(refinement, dict) else []
                if not isinstance(history, list):
                    history = []
                before = None
                after = None
                if history:
                    first_round = history[0]
                    last_round = history[-1]
                    if isinstance(first_round, dict):
                        before = first_round.get("input_stimulus_cases")
                    if isinstance(last_round, dict):
                        after = last_round.get("refined_stimulus_cases")
                if after is None:
                    after = [item.get("stimulus_case")] if item.get("stimulus_case") else []
                prefilter_refinements.append(
                    {
                        "phase": "prefilter",
                        "scenario_index": item.get("scenario_index"),
                        "scenario_id": item.get("scenario_id"),
                        "target_requirement_id": item.get("target_requirement_id"),
                        "scenario_name": item.get("scenario_name"),
                        "resolution_mode": item.get("resolution_mode"),
                        "semantic_entropy": item.get("semantic_entropy"),
                        "execution_mode": item.get("execution_mode"),
                        "worker_thread_name": item.get("worker_thread_name"),
                        "elapsed_seconds": item.get("elapsed_seconds"),
                        "stimulus_refinement": refinement or {},
                        "before": before or [],
                        "after": after or [],
                    }
                )

    if isinstance(cluster_payload, dict):
        rounds = cluster_payload.get("rounds")
        if isinstance(rounds, list):
            for round_item in rounds:
                if not isinstance(round_item, dict):
                    continue
                refinement = round_item.get("stimulus_refinement")
                if not isinstance(refinement, dict):
                    continue
                history = refinement.get("history")
                if not isinstance(history, list):
                    history = []
                before = None
                after = None
                if history:
                    first_round = history[0]
                    last_round = history[-1]
                    if isinstance(first_round, dict):
                        before = first_round.get("input_stimulus_cases")
                    if isinstance(last_round, dict):
                        after = last_round.get("refined_stimulus_cases")
                if after is None:
                    after = round_item.get("external_stimulus_cases") or []
                cluster_round_refinements.append(
                    {
                        "phase": "cluster_round",
                        "round_index": round_item.get("round_index"),
                        "target_requirement_id": round_item.get("target_requirement_id"),
                        "local_round_index": round_item.get("local_round_index"),
                        "final_generation": round_item.get("final_generation", False),
                        "resolution_mode": round_item.get("resolution_mode"),
                        "stimulus_refinement": refinement,
                        "before": before or [],
                        "after": after or [],
                    }
                )

    return {
        "prefilter_execution_mode": (
            discard_payload.get("prefilter_execution_mode")
            if isinstance(discard_payload, dict)
            else None
        ),
        "prefilter_max_workers": (
            discard_payload.get("prefilter_max_workers")
            if isinstance(discard_payload, dict)
            else None
        ),
        "prefilter_refinements": prefilter_refinements,
        "cluster_round_refinements": cluster_round_refinements,
    }


def _build_entropy_round_summary(
    disambiguation_payload: dict[str, Any],
    cluster_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a plotting-friendly entropy timeline for one task."""

    requirement_rounds = disambiguation_payload.get("requirement_rounds")
    cluster_rounds = []
    if isinstance(cluster_payload, dict):
        raw_rounds = cluster_payload.get("rounds")
        if isinstance(raw_rounds, list):
            cluster_rounds = [round_item for round_item in raw_rounds if isinstance(round_item, dict)]

    main_rounds = [round_item for round_item in cluster_rounds if not round_item.get("final_generation")]
    final_rounds = [round_item for round_item in cluster_rounds if round_item.get("final_generation")]
    main_round_cursor = 0

    entropy_rounds: list[dict[str, Any]] = []
    requirement_entropy_trajectories: list[dict[str, Any]] = []

    if isinstance(requirement_rounds, list):
        for requirement_round in requirement_rounds:
            if not isinstance(requirement_round, dict):
                continue
            local_rounds = requirement_round.get("rounds")
            if not isinstance(local_rounds, list):
                local_rounds = []
            per_requirement_rounds: list[dict[str, Any]] = []
            scenario = requirement_round.get("scenario")
            scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
            for local_round in local_rounds:
                if not isinstance(local_round, dict):
                    continue
                matched_main_round = (
                    main_rounds[main_round_cursor] if main_round_cursor < len(main_rounds) else {}
                )
                main_round_cursor += 1
                local_round_index = local_round.get("local_round_index")
                baseline_entropy = local_round.get("semantic_entropy_before")
                entropy_after_constraint = local_round.get("semantic_entropy_after_constraint")
                validation = local_round.get("constraint_validation")
                recovery = local_round.get("constraint_recovery")
                new_constraint_candidate = local_round.get("new_constraint_candidate")
                new_constraint_applied = local_round.get("new_constraint_applied")
                before_constraints = local_round.get("disambiguation_constraints_before", [])
                after_constraints = local_round.get("disambiguation_constraints_after", [])
                base_entry = {
                    "source": "main_round",
                    "target_requirement_id": requirement_round.get("target_requirement_id"),
                    "target_requirement": requirement_round.get("target_requirement"),
                    "scenario_id": scenario_id,
                    "local_round_index": local_round_index,
                    "global_round_index": matched_main_round.get("round_index"),
                    "semantic_entropy": baseline_entropy,
                    "functional_cluster_count": local_round.get("functional_cluster_count_before"),
                    "constraint_count_before": len(before_constraints)
                    if isinstance(before_constraints, list)
                    else None,
                    "constraint_count_after": len(after_constraints)
                    if isinstance(after_constraints, list)
                    else None,
                    "candidate_constraint": new_constraint_candidate,
                    "applied_constraint": new_constraint_applied or "",
                    "stop_reason": local_round.get("stop_reason"),
                }
                entropy_rounds.append(base_entry)

                validation_entropy = None
                validation_accepted = None
                if isinstance(validation, dict) and not validation.get("skipped"):
                    validation_entropy = validation.get("validated_entropy")
                    validation_accepted = validation.get("accepted")
                    entropy_rounds.append(
                        {
                            "source": "constraint_validation",
                            "target_requirement_id": requirement_round.get("target_requirement_id"),
                            "target_requirement": requirement_round.get("target_requirement"),
                            "scenario_id": scenario_id,
                            "local_round_index": local_round_index,
                            "parent_global_round_index": matched_main_round.get("round_index"),
                            "semantic_entropy": validation_entropy,
                            "functional_cluster_count": validation.get(
                                "functional_cluster_count_after"
                            ),
                            "constraint_count_before": len(before_constraints)
                            if isinstance(before_constraints, list)
                            else None,
                            "constraint_count_after": len(after_constraints)
                            if isinstance(after_constraints, list)
                            else None,
                            "candidate_constraint": validation.get("candidate_constraint"),
                            "accepted": validation_accepted,
                        }
                    )

                recovery_entropy = None
                if isinstance(recovery, dict):
                    recovery_entropy = recovery.get("recovered_entropy")
                    entropy_rounds.append(
                        {
                            "source": "constraint_recovery",
                            "target_requirement_id": requirement_round.get("target_requirement_id"),
                            "target_requirement": requirement_round.get("target_requirement"),
                            "scenario_id": scenario_id,
                            "local_round_index": local_round_index,
                            "parent_global_round_index": matched_main_round.get("round_index"),
                            "semantic_entropy": recovery_entropy,
                            "functional_cluster_count": recovery.get(
                                "functional_cluster_count_after"
                            ),
                            "constraint_count_before": len(before_constraints)
                            if isinstance(before_constraints, list)
                            else None,
                            "constraint_count_after": len(after_constraints)
                            if isinstance(after_constraints, list)
                            else None,
                            "candidate_constraint": new_constraint_candidate,
                            "accepted": False,
                        }
                    )

                per_requirement_rounds.append(
                    {
                        "local_round_index": local_round_index,
                        "global_round_index": matched_main_round.get("round_index"),
                        "constraint_count_before": len(before_constraints)
                        if isinstance(before_constraints, list)
                        else None,
                        "constraint_count_after": len(after_constraints)
                        if isinstance(after_constraints, list)
                        else None,
                        "semantic_entropy_before": baseline_entropy,
                        "semantic_entropy_after_constraint": entropy_after_constraint,
                        "validated_entropy": validation_entropy,
                        "validation_accepted": validation_accepted,
                        "recovered_entropy": recovery_entropy,
                        "new_constraint_candidate": new_constraint_candidate,
                        "new_constraint_applied": new_constraint_applied or "",
                        "stop_reason": local_round.get("stop_reason"),
                    }
                )

            requirement_entropy_trajectories.append(
                {
                    "target_requirement_id": requirement_round.get("target_requirement_id"),
                    "target_requirement": requirement_round.get("target_requirement"),
                    "scenario_id": scenario_id,
                    "resolved_under_targeted_stimulus": requirement_round.get(
                        "resolved_under_targeted_stimulus"
                    ),
                    "rounds": per_requirement_rounds,
                }
            )

    for final_round in final_rounds:
        entropy_rounds.append(
            {
                "source": "final_generation",
                "target_requirement_id": final_round.get("target_requirement_id"),
                "target_requirement": None,
                "scenario_id": None,
                "local_round_index": final_round.get("local_round_index"),
                "global_round_index": final_round.get("round_index"),
                "semantic_entropy": final_round.get("semantic_entropy"),
                "functional_cluster_count": final_round.get("functional_cluster_count"),
                "constraint_count_before": len(
                    final_round.get("disambiguation_constraints_input") or []
                ),
                "constraint_count_after": len(
                    final_round.get("disambiguation_constraints_input") or []
                ),
                "candidate_constraint": "",
                "applied_constraint": "",
                "stop_reason": final_round.get("resolution_mode"),
            }
        )

    return {
        "selected_mode": disambiguation_payload.get("selected_mode"),
        "executed_scenario_ids": disambiguation_payload.get("executed_scenario_ids", []),
        "resolved_requirement_ids": disambiguation_payload.get("resolved_requirement_ids", []),
        "final_disambiguation_constraints": disambiguation_payload.get(
            "final_disambiguation_constraints", []
        ),
        "entropy_rounds": entropy_rounds,
        "requirement_entropy_trajectories": requirement_entropy_trajectories,
    }


def _build_discarded_constraints_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect constraints rejected by entropy validation into one dedicated summary."""

    discarded_constraints = payload.get("discarded_constraints")
    if not isinstance(discarded_constraints, list):
        discarded_constraints = []

    summarized_discards: list[dict[str, Any]] = []
    for item in discarded_constraints:
        if not isinstance(item, dict):
            continue
        summarized_discards.append(
            {
                "target_requirement_id": item.get("target_requirement_id"),
                "target_requirement": item.get("target_requirement"),
                "local_round_index": item.get("local_round_index"),
                "candidate_constraint": item.get("candidate_constraint"),
                "discard_reason": item.get("discard_reason"),
                "baseline_entropy": item.get("baseline_entropy"),
                "validated_entropy": item.get("validated_entropy"),
                "recovered_entropy": item.get("recovered_entropy"),
                "functional_cluster_count_before": item.get(
                    "functional_cluster_count_before"
                ),
                "functional_cluster_count_validation": item.get(
                    "functional_cluster_count_validation"
                ),
                "functional_cluster_count_recovery": item.get(
                    "functional_cluster_count_recovery"
                ),
                "scenario": item.get("scenario"),
                "stimulus_case": item.get("stimulus_case"),
                "winner_judgment": item.get("winner_judgment"),
                "requirement_constraint_analysis": item.get(
                    "requirement_constraint_analysis"
                ),
            }
        )

    return {
        "selected_mode": payload.get("selected_mode"),
        "executed_scenario_ids": payload.get("executed_scenario_ids", []),
        "resolved_requirement_ids": payload.get("resolved_requirement_ids", []),
        "final_disambiguation_constraints": payload.get("final_disambiguation_constraints", []),
        "discarded_constraint_count": len(summarized_discards),
        "discarded_constraints": summarized_discards,
    }


def _build_disambiguation_constraints_provenance(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a flat provenance list showing where each final constraint came from."""

    requirement_rounds = payload.get("requirement_rounds")
    if not isinstance(requirement_rounds, list):
        return []

    provenance: list[dict[str, Any]] = []
    for requirement_round in requirement_rounds:
        if not isinstance(requirement_round, dict):
            continue
        target_requirement_id = requirement_round.get("target_requirement_id")
        target_requirement = requirement_round.get("target_requirement")
        local_rounds = requirement_round.get("rounds")
        if not isinstance(local_rounds, list):
            continue
        for local_round in local_rounds:
            if not isinstance(local_round, dict):
                continue
            new_constraint = str(local_round.get("new_constraint_applied") or "").strip()
            if not new_constraint:
                continue
            provenance.append(
                {
                    "constraint": new_constraint,
                    "target_requirement_id": target_requirement_id,
                    "target_requirement": target_requirement,
                    "scenario": (
                        {
                            "scenario_id": (requirement_round.get("scenario") or {}).get("scenario_id"),
                            "target_requirement_id": (requirement_round.get("scenario") or {}).get("target_requirement_id"),
                            "name": (requirement_round.get("scenario") or {}).get("name"),
                            "goal": (requirement_round.get("scenario") or {}).get("goal"),
                            "story": (requirement_round.get("scenario") or {}).get("story"),
                        }
                        if isinstance(requirement_round.get("scenario"), dict)
                        else None
                    ),
                    "local_round_index": local_round.get("local_round_index"),
                }
            )
    return provenance


def _build_requirement_round_bundle(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Bundle requirement, scenario, stimulus, and per-round constraint details together."""

    requirement_rounds = payload.get("requirement_rounds")
    if not isinstance(requirement_rounds, list):
        return []

    bundled_rounds: list[dict[str, Any]] = []
    for requirement_round in requirement_rounds:
        if not isinstance(requirement_round, dict):
            continue
        scenario = requirement_round.get("scenario")
        local_rounds = requirement_round.get("rounds")
        if not isinstance(local_rounds, list):
            local_rounds = []
        bundled_local_rounds: list[dict[str, Any]] = []
        for local_round in local_rounds:
            if not isinstance(local_round, dict):
                continue
            bundled_local_rounds.append(
                {
                    "local_round_index": local_round.get("local_round_index"),
                    "stimulus_case": local_round.get("stimulus_case"),
                    "selected_candidate": local_round.get("selected_candidate"),
                    "winner_judgment": local_round.get("winner_judgment"),
                    "new_constraint_applied": local_round.get("new_constraint_applied") or "",
                    "stop_reason": local_round.get("stop_reason"),
                    "same_requirement_prior_constraints": local_round.get(
                        "same_requirement_prior_constraints", []
                    ),
                    "disambiguation_constraints_before": local_round.get(
                        "disambiguation_constraints_before", []
                    ),
                    "disambiguation_constraints_after": local_round.get(
                        "disambiguation_constraints_after", []
                    ),
                    "winner_judgment_call": local_round.get("winner_judgment_call"),
                    "requirement_constraint_analysis": local_round.get(
                        "requirement_constraint_analysis"
                    ),
                    "requirement_constraint_call": local_round.get(
                        "requirement_constraint_call"
                    ),
                    "scenario_family_size_before": local_round.get(
                        "scenario_family_size_before"
                    ),
                    "stimulus_family_size_before": local_round.get(
                        "stimulus_family_size_before"
                    ),
                    "scenario_history_size_before": local_round.get(
                        "scenario_history_size_before"
                    ),
                    "stimulus_history_size_before": local_round.get(
                        "stimulus_history_size_before"
                    ),
                    "scenario_history_size_after": local_round.get(
                        "scenario_history_size_after"
                    ),
                    "stimulus_history_size_after": local_round.get(
                        "stimulus_history_size_after"
                    ),
                    "used_scenario_family": local_round.get("used_scenario_family"),
                    "used_stimulus_family": local_round.get("used_stimulus_family"),
                }
            )
        resolved = requirement_round.get("resolved_under_targeted_stimulus")
        bundled_rounds.append(
            {
                "target_requirement_id": requirement_round.get("target_requirement_id"),
                "target_requirement": requirement_round.get("target_requirement"),
                "scenario": scenario if isinstance(scenario, dict) else None,
                "resolved_under_targeted_stimulus": resolved,
                "resolved": resolved,
                "num_local_rounds": len(bundled_local_rounds),
                "local_rounds": bundled_local_rounds,
                "rounds": bundled_local_rounds,
            }
        )
    return bundled_rounds


def _build_all_generated_constraints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every actually applied constraint into one easy-to-scan list."""

    requirement_rounds = payload.get("requirement_rounds")
    if not isinstance(requirement_rounds, list):
        return []

    all_constraints: list[dict[str, Any]] = []
    for requirement_round in requirement_rounds:
        if not isinstance(requirement_round, dict):
            continue
        scenario = requirement_round.get("scenario")
        target_requirement_id = requirement_round.get("target_requirement_id")
        target_requirement = requirement_round.get("target_requirement")
        local_rounds = requirement_round.get("rounds")
        if not isinstance(local_rounds, list):
            continue
        for local_round in local_rounds:
            if not isinstance(local_round, dict):
                continue
            constraint_text = str(local_round.get("new_constraint_applied") or "").strip()
            if not constraint_text:
                continue
            all_constraints.append(
                {
                    "constraint": constraint_text,
                    "target_requirement_id": target_requirement_id,
                    "target_requirement": target_requirement,
                    "scenario": scenario if isinstance(scenario, dict) else None,
                    "local_round_index": local_round.get("local_round_index"),
                    "stimulus_case": local_round.get("stimulus_case"),
                    "selected_candidate": local_round.get("selected_candidate"),
                    "winner_judgment": local_round.get("winner_judgment"),
                    "same_requirement_prior_constraints": local_round.get(
                        "same_requirement_prior_constraints", []
                    ),
                    "stop_reason": local_round.get("stop_reason"),
                }
            )
    return all_constraints


def write_task_index(trace_root: Path, tasks: list[TaskSpec]) -> Path:
    """Persist the scanned task inventory for debugging and reproducibility."""

    output_path = trace_root / "task_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([task.to_dict() for task in tasks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_task_content(task_output_dir: Path, task: TaskSpec) -> Path:
    """Copy the original task inputs beside artifacts for easier manual review."""

    content_dir = task_output_dir / "task_content"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    content_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(task.spec_path, content_dir / task.spec_path.name)
    shutil.copy2(task.testbench_path, content_dir / task.testbench_path.name)
    return content_dir


def write_candidate_artifacts(task_output_dir: Path, trace: PipelineTrace) -> Path:
    """Persist large candidate/verifier outputs outside of trace.json."""

    artifacts_dir = task_output_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _write_behavior_to_constraint_artifacts(artifacts_dir, trace)

    verification_by_candidate = {
        item.candidate_id: item for item in trace.verification_results
    }
    compile_by_candidate = {
        item.candidate_id: item for item in trace.compile_results
    }
    second_compile_by_candidate = {
        item.candidate_id: item for item in trace.second_compile_results
    }
    if trace.baseline_candidate is not None:
        baseline_dir = artifacts_dir / "baseline_verilog"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "candidate.v").write_text(
            trace.baseline_candidate.clean_verilog + "\n",
            encoding="utf-8",
        )
        verification = trace.baseline_verification_result
        if verification is not None:
            _write_verification_result_artifacts(
                candidate_dir=baseline_dir,
                verification=verification,
                task=trace.task,
            )
    final_cluster_candidate_id = _resolve_final_cluster_candidate_id(trace)
    _write_intermediate_cluster_verilog_rounds(artifacts_dir, trace)
    written_candidate_ids: set[str] = set()
    for candidate in trace.candidates:
        if candidate.candidate_id.startswith("generated_cluster_"):
            if candidate.candidate_id != final_cluster_candidate_id:
                continue
        candidate_dir = artifacts_dir / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate.v").write_text(candidate.clean_verilog + "\n", encoding="utf-8")
        written_candidate_ids.add(candidate.candidate_id)
        compile_result = compile_by_candidate.get(candidate.candidate_id)
        verification = verification_by_candidate.get(candidate.candidate_id)
        if compile_result is not None:
            _write_verification_result_artifacts(
                candidate_dir=candidate_dir,
                verification=compile_result,
                task=trace.task,
            )
        if verification is not None:
            _write_verification_result_artifacts(
                candidate_dir=candidate_dir,
                verification=verification,
                task=trace.task,
            )
    for verification in trace.verification_results:
        if verification.candidate_id in written_candidate_ids:
            continue
        candidate_dir = artifacts_dir / verification.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_verification_result_artifacts(
            candidate_dir=candidate_dir,
            verification=verification,
            task=trace.task,
        )
    if trace.second_compile_candidates:
        second_compile_dir = artifacts_dir / "second_compile_process"
        second_compile_dir.mkdir(parents=True, exist_ok=True)
        for candidate in trace.second_compile_candidates:
            candidate_dir = second_compile_dir / candidate.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            (candidate_dir / "candidate.v").write_text(candidate.clean_verilog + "\n", encoding="utf-8")
            compile_result = second_compile_by_candidate.get(candidate.candidate_id)
            if compile_result is not None:
                _write_verification_result_artifacts(
                    candidate_dir=candidate_dir,
                    verification=compile_result,
                    task=trace.task,
                )
            verification = verification_by_candidate.get(candidate.candidate_id)
            if verification is not None:
                _write_verification_result_artifacts(
                    candidate_dir=candidate_dir,
                    verification=verification,
                    task=trace.task,
                )
    return artifacts_dir


def _write_verification_result_artifacts(
    candidate_dir: Path,
    verification: Any,
    task: TaskSpec,
) -> None:
    """Persist durable verification diagnostics beside one candidate."""

    (candidate_dir / "compile_summary.txt").write_text(
        str(verification.compile_summary or "").strip() + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "evaluation_summary.txt").write_text(
        str(verification.evaluation_summary or "").strip() + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "verification_result.json").write_text(
        json.dumps(verification.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(task.testbench_path, candidate_dir / task.testbench_path.name)


def _resolve_final_cluster_candidate_id(trace: PipelineTrace) -> str | None:
    """Return the cluster candidate chosen for final compile/verification outputs."""

    if not trace.candidates:
        return None
    verification_ids = [item.candidate_id for item in trace.verification_results]
    for candidate_id in verification_ids:
        if candidate_id.startswith("generated_cluster_"):
            return candidate_id
    compile_ids = [item.candidate_id for item in trace.compile_results]
    for candidate_id in compile_ids:
        if candidate_id.startswith("generated_cluster_"):
            return candidate_id
    for candidate in trace.candidates:
        if candidate.candidate_id.startswith("generated_cluster_majority"):
            return candidate.candidate_id
    for candidate in trace.candidates:
        if candidate.candidate_id.startswith("generated_cluster_top1"):
            return candidate.candidate_id
    return None


def _write_intermediate_cluster_verilog_rounds(artifacts_dir: Path, trace: PipelineTrace) -> None:
    """Persist per-round dual cluster outputs for easier inspection."""

    constraint_trace = trace.constraint_trace or {}
    rounds = constraint_trace.get("llm_verilog_cluster_rounds") or []
    if not rounds:
        return

    intermediate_dir = artifacts_dir / "intermediate_verilog"
    wrote_any = False
    for default_index, round_bundle in enumerate(rounds, start=1):
        if not isinstance(round_bundle, dict):
            continue
        resolution = round_bundle.get("resolution") or {}
        selected_candidates = resolution.get("selected_candidates") or []
        if len(selected_candidates) < 2:
            continue

        round_index = int(round_bundle.get("round_index", default_index) or default_index)
        round_dir = intermediate_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for idx, selected in enumerate(selected_candidates[:2]):
            verilog_code = str(selected.get("verilog_code", "")).strip()
            if not verilog_code:
                continue
            label = "candidate_a" if idx == 0 else "candidate_b"
            (round_dir / f"{label}.v").write_text(verilog_code + "\n", encoding="utf-8")
        wrote_any = True

    if not wrote_any and intermediate_dir.exists():
        shutil.rmtree(intermediate_dir)


def _write_behavior_to_constraint_artifacts(artifacts_dir: Path, trace: PipelineTrace) -> None:
    constraint_trace = trace.constraint_trace or {}
    if not constraint_trace:
        return

    if trace.pipeline_variant == "requirements_constraint_selfplanning":
        constraint_dir = artifacts_dir / trace.pipeline_variant
    else:
        constraint_dir = artifacts_dir / "constraint_trace"
    constraint_dir.mkdir(parents=True, exist_ok=True)

    runs = constraint_trace.get("runs") or []
    if runs:
        (constraint_dir / "runs.json").write_text(
            json.dumps(runs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runs_dir = constraint_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for index, bundle in enumerate(runs, start=1):
            if not isinstance(bundle, dict):
                continue
            run_id = str(bundle.get("run_id", "")).strip() or f"run_{index:02d}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "initial_understanding.json").write_text(
                json.dumps(bundle.get("initial_understanding") or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (run_dir / "final_understanding.json").write_text(
                json.dumps(bundle.get("final_understanding") or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rounds = bundle.get("rounds") or []
            (run_dir / "rounds.json").write_text(
                json.dumps(rounds, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if not rounds:
                continue
            rounds_dir = run_dir / "rounds"
            rounds_dir.mkdir(parents=True, exist_ok=True)
            for round_index, round_bundle in enumerate(rounds, start=1):
                if not isinstance(round_bundle, dict):
                    continue
                round_id = int(round_bundle.get("round_index", round_index) or round_index)
                round_dir = rounds_dir / f"round_{round_id:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "understanding_before.json").write_text(
                    json.dumps(round_bundle.get("understanding_before") or {}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / "challenges.json").write_text(
                    json.dumps(round_bundle.get("challenges") or [], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / "judgments.json").write_text(
                    json.dumps(round_bundle.get("judgments") or [], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / "understanding_after.json").write_text(
                    json.dumps(round_bundle.get("understanding_after") or {}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / "round_summary.json").write_text(
                    json.dumps(
                        {
                            "round_index": round_bundle.get("round_index"),
                            "stop_reason": round_bundle.get("stop_reason", ""),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    final_understandings = constraint_trace.get("final_understandings") or []
    if final_understandings:
        (constraint_dir / "final_understandings.json").write_text(
            json.dumps(final_understandings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    grouped_constraint_candidates = constraint_trace.get("grouped_constraint_candidates") or []
    if grouped_constraint_candidates:
        (constraint_dir / "grouped_constraint_candidates.json").write_text(
            json.dumps(grouped_constraint_candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    semantic_clusters = constraint_trace.get("semantic_clusters") or []
    if semantic_clusters:
        (constraint_dir / "semantic_clusters.json").write_text(
            json.dumps(semantic_clusters, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    final_constraints = constraint_trace.get("final_constraints") or []
    if final_constraints:
        (constraint_dir / "final_constraints.json").write_text(
            json.dumps(final_constraints, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    requirements = constraint_trace.get("requirements") or []
    if requirements:
        (constraint_dir / "requirements.json").write_text(
            json.dumps(requirements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    constraint_groups = constraint_trace.get("constraint_groups") or {}
    if constraint_groups:
        (constraint_dir / "constraint_groups.json").write_text(
            json.dumps(constraint_groups, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    constraints = constraint_trace.get("constraints") or []
    if constraints:
        (constraint_dir / "constraints.json").write_text(
            json.dumps(constraints, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    self_planning = constraint_trace.get("self_planning") or []
    if self_planning:
        (constraint_dir / "self_planning.json").write_text(
            json.dumps(self_planning, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_run_summary(
    trace_root: Path,
    traces: list[PipelineTrace],
    run_timing: dict[str, Any] | None = None,
) -> Path:
    """Persist aggregate syntax/function pass statistics and failed task names."""

    summary = build_run_summary(traces, run_timing=run_timing)
    _augment_summary_with_run_completion(trace_root, summary, traces)
    output_path = trace_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    task_summary_path = trace_root / "task_summary.json"
    task_summary_path.write_text(
        json.dumps(_build_task_summary(traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    constraint_summary_path = trace_root / "constraint_summary.json"
    constraint_summary_path.write_text(
        json.dumps(_build_constraint_summary(traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    entropy_summary_path = trace_root / "entropy_summary.json"
    entropy_summary_path.write_text(
        json.dumps(_build_entropy_summary(traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _read_design_description(trace: PipelineTrace) -> str:
    try:
        return trace.task.spec_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_task_summary(traces: list[PipelineTrace]) -> list[dict[str, Any]]:
    """Summarize each task with requirement/scenario/stimulus/constraint round bundles."""

    task_rows: list[dict[str, Any]] = []
    for trace in traces:
        llm_payload = (trace.llm_io or {}).get("llm_verilog_cluster_disambiguation_constraints") or {}
        round_bundle = _build_requirement_round_bundle(llm_payload) if isinstance(llm_payload, dict) else []
        task_rows.append(
            {
                "pipeline_variant": trace.pipeline_variant,
                "task_id": trace.task.task_id,
                "task_name": trace.task.task_name,
                "design_description": _read_design_description(trace),
                "requirement_rounds": round_bundle,
            }
        )
    return task_rows


def _build_constraint_summary(traces: list[PipelineTrace]) -> list[dict[str, Any]]:
    """Summarize each task with only design description, requirement, and generated constraints."""

    task_rows: list[dict[str, Any]] = []
    for trace in traces:
        generated_constraints = _trace_generated_constraints(trace)
        if not generated_constraints:
            continue
        task_rows.append(
            {
                "pipeline_variant": trace.pipeline_variant,
                "task_id": trace.task.task_id,
                "task_name": trace.task.task_name,
                "design_description": _read_design_description(trace),
                "constraints": [
                    {
                        "target_requirement_id": item.get("target_requirement_id"),
                        "target_requirement": item.get("target_requirement"),
                        "constraint": item.get("constraint"),
                    }
                    for item in generated_constraints
                ],
            }
        )
    return task_rows


def _build_entropy_summary(traces: list[PipelineTrace]) -> dict[str, Any]:
    """Aggregate entropy trajectories across all tasks for plotting and analysis."""

    tasks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for trace in traces:
        llm_payload = (trace.llm_io or {}).get("llm_verilog_cluster_disambiguation_constraints") or {}
        cluster_payload = (trace.llm_io or {}).get("llm_verilog_cluster") or {}
        if not isinstance(llm_payload, dict):
            continue
        entropy_summary = _build_entropy_round_summary(
            llm_payload,
            cluster_payload if isinstance(cluster_payload, dict) else None,
        )
        task_row = {
            "pipeline_variant": trace.pipeline_variant,
            "task_id": trace.task.task_id,
            "task_name": trace.task.task_name,
            **entropy_summary,
        }
        tasks.append(task_row)
        for point_index, round_item in enumerate(entropy_summary.get("entropy_rounds", []), start=1):
            if not isinstance(round_item, dict):
                continue
            rows.append(
                {
                    "pipeline_variant": trace.pipeline_variant,
                    "task_id": trace.task.task_id,
                    "task_name": trace.task.task_name,
                    "point_index": point_index,
                    **round_item,
                }
            )

    return {
        "task_count": len(tasks),
        "tasks": tasks,
        "rows": rows,
    }


def build_run_summary(
    traces: list[PipelineTrace],
    run_timing: dict[str, Any] | None = None,
) -> dict:
    """Build aggregate summary from completed traces."""

    by_variant: dict[str, dict] = {}
    variant_traces: dict[str, list[PipelineTrace]] = {}

    for trace in traces:
        variant = trace.pipeline_variant
        task_id = trace.task.task_id
        verification_results = trace.verification_results
        syntax_pass = any(item.compile_returncode == 0 for item in verification_results)
        functional_pass = any(item.passed for item in verification_results)
        variant_traces.setdefault(variant, []).append(trace)

        bucket = by_variant.setdefault(
            variant,
            {
                "total_tasks": 0,
                "syntax_pass_count": 0,
                "functional_pass_count": 0,
                "Pass_CMB": 0,
                "Pass_SEQ": 0,
                "total_Mismatches": 0,
                "total_Mismatches_scope": "reported_mismatch_summaries_only",
                "reported_mismatch_summary_task_count": 0,
                "reported_mismatch_summary_coverage_rate": 0.0,
                "syntax_failed_tasks": [],
                "functional_failed_tasks": [],
                "constrainted_functional_failed_tasks": [],
                "tasks_without_reported_mismatches_summary": [],
            },
        )
        bucket["total_tasks"] += 1
        if _trace_has_reported_mismatches_summary(trace):
            bucket["reported_mismatch_summary_task_count"] += 1
            bucket["total_Mismatches"] += _trace_total_mismatches(trace)
        else:
            bucket["tasks_without_reported_mismatches_summary"].append(task_id)
        if syntax_pass:
            bucket["syntax_pass_count"] += 1
        else:
            bucket["syntax_failed_tasks"].append(task_id)

        if functional_pass:
            bucket["functional_pass_count"] += 1
            circuit_type = _read_circuit_type(trace)
            if circuit_type == "CMB":
                bucket["Pass_CMB"] += 1
            elif circuit_type == "SEQ":
                bucket["Pass_SEQ"] += 1
        else:
            bucket["functional_failed_tasks"].append(task_id)
            if _trace_generated_constraints(trace):
                bucket["constrainted_functional_failed_tasks"].append(task_id)

    for bucket in by_variant.values():
        total = bucket["total_tasks"]
        bucket["syntax_pass_rate"] = bucket["syntax_pass_count"] / total if total else 0.0
        bucket["functional_pass_rate"] = bucket["functional_pass_count"] / total if total else 0.0
        bucket["reported_mismatch_summary_coverage_rate"] = (
            bucket["reported_mismatch_summary_task_count"] / total if total else 0.0
        )

    for variant, traces_for_variant in variant_traces.items():
        by_variant[variant]["model_usage"] = _build_model_usage_summary(traces_for_variant)
        by_variant[variant]["total_elapsed_seconds"] = sum(
            max(0.0, trace.elapsed_seconds) for trace in traces_for_variant
        )
        challenge_stats = _build_challenge_stats(variant, traces_for_variant)
        if challenge_stats is not None:
            by_variant[variant]["challenge_stats"] = challenge_stats

    summary = {
        "variants": by_variant,
    }
    if run_timing:
        summary["run_timing"] = dict(run_timing)
    return summary


def _trace_missing_final_mismatches_summary(trace: PipelineTrace) -> bool:
    verification = _select_summary_verification_result(trace)
    if verification is None:
        return False
    return not _has_mismatches_summary_line(verification.evaluation_summary)


def _trace_has_reported_mismatches_summary(trace: PipelineTrace) -> bool:
    verification = _select_summary_verification_result(trace)
    if verification is None:
        return False
    return _has_mismatches_summary_line(verification.evaluation_summary)


def _has_mismatches_summary_line(text: str) -> bool:
    return re.search(r"Mismatches:\s*\d+\s+in\s+\d+\s+samples", text, flags=re.IGNORECASE) is not None


def _trace_total_mismatches(trace: PipelineTrace) -> int:
    verification = _select_summary_verification_result(trace)
    if verification is None:
        return 0
    parsed = _parse_mismatches_summary(verification.evaluation_summary)
    if parsed is None:
        return 0
    mismatch_count, _sample_count = parsed
    return mismatch_count


def _select_summary_verification_result(trace: PipelineTrace):
    runtime_results = [item for item in trace.verification_results if item.run_returncode is not None]
    if not runtime_results:
        return None

    final_cluster_candidate_id = _resolve_final_cluster_candidate_id(trace)
    if final_cluster_candidate_id:
        for item in runtime_results:
            if item.candidate_id == final_cluster_candidate_id:
                return item

    runtime_by_candidate = {item.candidate_id: item for item in runtime_results}
    for candidate in reversed(trace.candidates):
        match = runtime_by_candidate.get(candidate.candidate_id)
        if match is not None:
            return match

    return runtime_results[-1]


def _parse_mismatches_summary(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples",
        text or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_circuit_type(trace: PipelineTrace) -> str:
    circuit_type_path = trace.task.task_dir / "circuit_type.txt"
    try:
        return circuit_type_path.read_text(encoding="utf-8").strip().upper()
    except Exception:
        return ""


def _build_model_usage_summary(traces: list[PipelineTrace]) -> dict[str, int]:
    totals = {
        "total_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for trace in traces:
        _accumulate_model_usage(trace.llm_io, totals)
    return totals


def _trace_generated_constraints(trace: PipelineTrace) -> list[dict[str, Any]]:
    llm_payload = (trace.llm_io or {}).get("llm_verilog_cluster_disambiguation_constraints") or {}
    if not isinstance(llm_payload, dict):
        return []
    generated_constraints = _build_all_generated_constraints(llm_payload)
    if not isinstance(generated_constraints, list):
        return []
    return generated_constraints


def _accumulate_model_usage(node: Any, totals: dict[str, int]) -> None:
    if isinstance(node, dict):
        if _is_llm_call_record(node):
            totals["total_model_calls"] += 1
            usage = node.get("usage")
            if isinstance(usage, dict):
                totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            return
        for value in node.values():
            _accumulate_model_usage(value, totals)
        return
    if isinstance(node, list):
        for item in node:
            _accumulate_model_usage(item, totals)


def _is_llm_call_record(node: dict[str, Any]) -> bool:
    required_keys = {
        "backend",
        "mode",
        "system_prompt",
        "user_prompt",
        "raw_response",
        "parsed_response",
    }
    return required_keys.issubset(node.keys())


def _llm_timeout_retry_triggered(node: Any) -> bool:
    if isinstance(node, dict):
        if _is_llm_call_record(node):
            return bool(node.get("timeout_retry_triggered"))
        return any(_llm_timeout_retry_triggered(value) for value in node.values())
    if isinstance(node, list):
        return any(_llm_timeout_retry_triggered(item) for item in node)
    return False


def _augment_summary_with_run_completion(trace_root: Path, summary: dict, traces: list[PipelineTrace]) -> None:
    expected_task_ids: list[str] = []
    task_index_path = trace_root / "task_index.json"
    expected_total = None
    if task_index_path.exists():
        try:
            task_index = json.loads(task_index_path.read_text(encoding="utf-8"))
            expected_task_ids = [str(item.get("task_id") or item.get("task_name")) for item in task_index if isinstance(item, dict)]
            expected_total = len(expected_task_ids)
        except Exception:
            expected_total = None
            expected_task_ids = []

    variants = summary.get("variants", {})
    run_errors: list[Any] = []
    run_errors_path = trace_root / "run_errors.json"
    if run_errors_path.exists():
        try:
            run_errors = json.loads(run_errors_path.read_text(encoding="utf-8"))
        except Exception:
            run_errors = []

    known_variants = sorted(
        {
            str(trace.pipeline_variant).strip()
            for trace in traces
            if str(trace.pipeline_variant).strip()
        }
        | {
            str(item.get("branch_variant", "")).strip()
            for item in run_errors
            if isinstance(item, dict) and str(item.get("branch_variant", "")).strip()
        }
    )
    completed_task_ids_by_variant: dict[str, list[str]] = {}
    for variant in known_variants:
        completed_ids = sorted(
            {
                trace.task.task_id
                for trace in traces
                if trace.pipeline_variant == variant
            }
        )
        completed_task_ids_by_variant[variant] = completed_ids

    for variant in known_variants:
        bucket = variants.setdefault(
            variant,
            {
                "total_tasks": 0,
                "syntax_pass_count": 0,
                "functional_pass_count": 0,
                "Pass_CMB": 0,
                "Pass_SEQ": 0,
                "total_Mismatches": 0,
                "total_Mismatches_scope": "reported_mismatch_summaries_only",
                "reported_mismatch_summary_task_count": 0,
                "reported_mismatch_summary_coverage_rate": 0.0,
                "syntax_failed_tasks": [],
                "functional_failed_tasks": [],
                "constrainted_functional_failed_tasks": [],
                "tasks_without_reported_mismatches_summary": [],
                "syntax_pass_rate": 0.0,
                "functional_pass_rate": 0.0,
            },
        )
        if expected_task_ids:
            bucket["uncompleted_tasks"] = sorted(set(expected_task_ids) - set(completed_task_ids_by_variant[variant]))
        else:
            bucket["uncompleted_tasks"] = []

    completed = {variant: len(completed_task_ids_by_variant[variant]) for variant in known_variants}
    completion = {
        "expected_total_tasks_per_variant": expected_total,
        "completed_tasks_by_variant": completed,
        "is_complete_by_variant": {
            variant: (count == expected_total if expected_total is not None else None)
            for variant, count in completed.items()
        },
    }

    if run_errors_path.exists():
        completion["run_errors_file"] = "run_errors.json"
        completion["run_error_count"] = len(run_errors)
        completion["failed_tasks"] = sorted(
            {
                item.get("task_id") or item.get("task_name")
                for item in run_errors
                if isinstance(item, dict)
            }
        )
        errors_by_variant: dict[str, list[str]] = {}
        for variant in known_variants:
            errors_by_variant[variant] = sorted(
                {
                    item.get("task_id") or item.get("task_name")
                    for item in run_errors
                    if isinstance(item, dict) and item.get("branch_variant") == variant
                }
            )
            variants[variant]["runtime_error_tasks"] = errors_by_variant[variant]
        completion["failed_tasks_by_variant"] = errors_by_variant
    else:
        for variant in known_variants:
            variants[variant]["runtime_error_tasks"] = []

    summary["run_completion"] = completion


def _build_challenge_stats(variant: str, traces: list[PipelineTrace]) -> dict | None:
    return None
