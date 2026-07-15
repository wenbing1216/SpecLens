"""Reasoning stages for spec understanding, QA, summarization, and generation."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .llm import LLMBackend
from .models import (
    CandidateVerilog,
    SpecUnderstanding,
    TaskSpec,
)


DEFAULT_TOP_MODULE_NAME = "TopModule"
DIRECT_FEW_SHOTS_EXAMPLES = """Example 1: XOR gate
module TopModule(
    input  logic in0,
    input  logic in1,
    output logic out
);

    assign out = in0 ^ in1;

endmodule
Example 2: 8-bit registered incrementer
module TopModule(
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] in_,
    output logic [7:0] out
);

    // Sequential logic
    logic [7:0] reg_out;
    always @( posedge clk ) begin
        if ( reset )
        reg_out <= 0;
        else
        reg_out <= in_;
    end

    // Combinational logic
    logic [7:0] temp_wire;
    always @(*) begin
        temp_wire = reg_out + 1;
    end

    // Structural connections
    assign out = temp_wire;

endmodule"""


def understand_spec(
    task: TaskSpec,
    spec_text: str,
    planner_backend: LLMBackend,
) -> tuple[SpecUnderstanding, dict[str, Any]]:
    """Module 3: derive understanding through focused prompt stages."""

    explicit_requirements, explicit_record = extract_explicit_requirements(task, spec_text, planner_backend)
    ambiguities: list[str] = []
    self_planning, planning_record = extract_self_planning(
        task,
        spec_text,
        planner_backend,
    )
    understanding = SpecUnderstanding(
        explicit_requirements=explicit_requirements,
        ambiguities=ambiguities,
        self_planning=self_planning,
    )
    llm_io = {
        "understanding": {
            "explicit_requirements": explicit_record,
            "ambiguities": [],
            "self_planning": planning_record,
        }
    }
    return understanding, llm_io



# If the spec gives an example, keep it as an example-like requirement rather than upgrading it into a broader mandatory rule.


def extract_explicit_requirements(task: TaskSpec, spec_text: str, backend: LLMBackend) -> tuple[list[str], dict[str, Any]]:
    """Extract only requirements explicitly stated in the spec, split into interface and behavior groups."""

    system_prompt = (
        "Extract only explicit Verilog-relevant requirements from the spec. "
        "Do not infer or fill gaps. "
        "Do not normalize the spec into stricter engineering language than the spec itself uses. "
        "Keep the extracted requirements at the same semantic level as the spec."
    )
    user_prompt = f"""
Return JSON with keys:
- interface_requirements: list[str]
- behavior_requirements: list[str]

Only include information directly stated in the spec.
Prefer atomic requirements.
Do not silently repair awkward wording, inconsistent wording, or likely spec mistakes.
Do not infer canonical reset styles, overflow formulas, timing rules, interface corrections, or implementation policies unless explicitly stated.


Grouping rules:
- interface_requirements should include only explicit interface/module requirements such as module name, ports, widths, and explicitly stated clock/reset interface form
- behavior_requirements should include only explicit externally observable functional behavior requirements from the spec, expressed in terms of input conditions, clocked behavior when relevant, and outputs rather than structural descriptors, implementation style, or generic circuit classification
- keep both groups at the same semantic level as the spec
- do not turn behavior_requirements into implementation steps, helper mechanisms, scenarios, or testbench instructions
- do not promote generic structural or timing descriptors such as "sequential circuit", "one bit of memory", or similar circuit-class labels into standalone behavior_requirements unless they directly state an externally observable functional rule
- If the specification includes a waveform table, timing diagram, or simulation waveform as the primary behavioral description, first determine from the specification whether the sequential logic is triggered on the positive or negative clock edge, then represent the waveform-described behavior as only one single coherent behavior requirement by describing all triggered clock-edge rows whose output is not x together with the immediately adjacent non-triggered clock rows that provide the pre-edge input context, and do not also extract separate behavior_requirements for individual rows, timestamps, or local waveform fragments.
- For waveform-driven sequential specifications, when describing that only one coherent behavior requirement, read the input for each triggered clock-edge row from the immediately adjacent non-triggered row that represents the input just before that triggered edge, and read the output as the observable output shown on the triggered clock-edge row itself.

Spec:
{spec_text}
""".strip()
    record = backend.complete_json_record(system_prompt=system_prompt, user_prompt=user_prompt)
    data = record["parsed_response"]
    interface_requirements = _as_string_list(data.get("interface_requirements"))
    behavior_requirements = _as_string_list(data.get("behavior_requirements"))
    all_requirements = list(interface_requirements)
    all_requirements.extend(behavior_requirements)
    record["parsed_interface_requirements"] = interface_requirements
    record["parsed_behavior_requirements"] = behavior_requirements
    record["parsed_all_requirements"] = all_requirements
    return all_requirements, record


def extract_self_planning(
    task: TaskSpec,
    spec_text: str,
    backend: LLMBackend,
) -> tuple[list[str], dict[str, Any]]:
    """Generate a self-planning decomposition for implementing the final Verilog."""

    system_prompt = (
        "First, understand the requirements above and give reasoning steps in natural language to implement this RTL design. "
        "In addition, give advice to avoid syntax errors during Verilog generation. "
        "Do not generate Verilog code yet."
    )
    user_prompt = f"""
Return a short natural-language planning note for this module.
Use short lines or bullet-like lines when helpful.

Spec:
{spec_text}
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw_text = str(record.get("parsed_response", "")).strip()

    planning_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if line:
            planning_lines.append(line)

    if not planning_lines and raw_text:
        planning_lines = [raw_text]

    record["parsed_self_planning"] = planning_lines
    return planning_lines, record


def generate_candidate(
    task: TaskSpec,
    spec_text: str,
    understanding: SpecUnderstanding,
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Module 9 + 10: generate a single candidate and extract clean Verilog."""

    system_prompt = (
        "You are now a professional Verilog coding expert. Generate Verilog code based on the information below."
    )
    user_prompt = f"""
Below is the original specification, followed by the extracted understanding sections that may help implementation.

Spec:
{spec_text}

Explicit requirements:
{json.dumps(understanding.explicit_requirements, ensure_ascii=False, indent=2)}

Self-planning:
{json.dumps(understanding.self_planning, ensure_ascii=False, indent=2)}

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated from spec + understanding sections.",
    ), record


def generate_candidate_from_spec_only(
    task: TaskSpec,
    spec_text: str,
    backend: LLMBackend,
    additional_guidance: str | None = None,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Generate a single candidate directly from the specification only."""

    system_prompt = (
        "Generate Verilog code directly from the specification."
    )
    guidance_section = ""
    if additional_guidance:
        guidance_section = f"""

Additional hardware-knowledge reminders:
{additional_guidance}
""".rstrip()

    user_prompt = f"""
Generate Verilog code directly from the following specification.

Spec:
{spec_text}
{guidance_section}

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate_from_spec_only",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated directly from spec only.",
    ), record


def generate_candidate_from_spec_selfplanning_style(
    task: TaskSpec,
    spec_text: str,
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Generate a single candidate directly from the specification with a self-planning-style prompt."""

    system_prompt = (
        "First, understand the requirements above and give reasoning steps in natural language to implement this RTL design. "
        "In addition, give advice to avoid syntax errors during Verilog generation."
    )
    user_prompt = f"""
Generate Verilog code directly from the following specification.

Spec:
{spec_text}

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate_from_spec_selfplanning_style",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated directly from spec with a self-planning-style prompt.",
    ), record


def generate_candidate_from_spec_few_shots(
    task: TaskSpec,
    spec_text: str,
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Generate a single candidate directly from the specification with two few-shot examples."""

    system_prompt = "Generate Verilog code directly from the specification."
    user_prompt = f"""
Generate Verilog code directly from the following specification.

Here are two reference examples:
{DIRECT_FEW_SHOTS_EXAMPLES}

Spec:
{spec_text}

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate_from_spec_few_shots",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated directly from spec with two few-shot examples.",
    ), record


def generate_candidate_from_spec_with_constraints(
    task: TaskSpec,
    spec_text: str,
    constraints: list[str],
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """You are now a professional Verilog coding expert. Generate a single candidate directly from the spec plus external constraints."""

    system_prompt = (
        "You are now a professional Verilog coding expert. Generate Verilog code from the specification while "
        "following the provided design constraints."
    )
    constraints_text = json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "[]"
    user_prompt = f"""
Generate Verilog code from the following specification and design constraints.


Spec:
{spec_text}

Constraints:
{constraints_text}

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
If the specification does not explicitly name the top-level module, name it {DEFAULT_TOP_MODULE_NAME}.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate_from_spec_with_constraints",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated directly from spec plus external constraints.",
    ), record


def generate_candidate_with_constraints(
    task: TaskSpec,
    spec_text: str,
    constraints: list[str],
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Generate a single candidate from spec plus plain Socratic-derived constraint strings."""

    system_prompt = (
        "You are now a professional Verilog coding expert. Generate Verilog code based on the specification and derived functional constraints."
    )
    user_prompt = f"""
Below is the original specification, followed by functional constraints derived from repeated spec-grounded Socratic understanding.

Spec:
{spec_text}

Functional constraints:
{json.dumps(constraints or [], ensure_ascii=False, indent=2)}

Rules:
- implement the specified behavior faithfully
- use the constraints only to clarify functional logic already supported by the spec
- do not add extra functionality beyond the specification
- do not turn open points into invented hard requirements

Return the final answer as a fenced ```verilog``` block containing only the Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
If the specification does not explicitly name the top-level module, name it {DEFAULT_TOP_MODULE_NAME}.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{task.task_id}:generate_candidate_with_constraints",
        fallback_text=raw,
    )
    return CandidateVerilog(
        candidate_id="generated",
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes="Generated from spec + derived functional constraints.",
    ), record

def extract_verilog_module(text: str) -> str:
    """Extract Verilog, preferring the last fenced block and preserving multiple modules."""

    fenced = re.findall(r"```(?:verilog|systemverilog)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced[-1]] if fenced else [text]
    pattern = re.compile(r"\bmodule\b.*?\bendmodule\b", flags=re.DOTALL)
    for candidate in candidates:
        matches = list(pattern.finditer(candidate))
        if matches:
            start = matches[0].start()
            end = matches[-1].end()
            return candidate[start:end].strip()
    for candidate in candidates:
        stripped = candidate.strip()
        if stripped.startswith("module ") and stripped.endswith("endmodule"):
            return stripped
    return ""


def extract_verilog_module_with_warning(text: str, context: str, fallback_text: str = "") -> str:
    """Extract Verilog and emit a non-blocking terminal warning if extraction fails."""

    extracted = extract_verilog_module(text)
    if extracted.strip():
        return extracted.strip()
    print(
        f"[safe_pipeline_156 warning] failed to extract Verilog module in {context}; continuing with fallback.",
        file=sys.stderr,
        flush=True,
    )
    fallback = fallback_text.strip()
    if fallback:
        return fallback
    return text.strip()


def guess_module_name(verilog_text: str) -> str | None:
    """Guess the module identifier from the first module declaration."""

    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", verilog_text)
    if match:
        return match.group(1)
    return None


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
