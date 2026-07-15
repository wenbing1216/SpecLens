from __future__ import annotations

"""
This module implements a Verilog-candidate clustering pipeline driven by a natural-language
specification. The semantic public input is a single `spec` string, while callers may optionally
override the default Verilog-generation prompts and a small set of control hyperparameters. The
pipeline first asks an LLM to generate multiple Verilog candidates, parses and clusters their
interfaces, selects the majority interface plus a representative circuit type/edge/reset
convention, then asks the LLM to generate verification scenarios and concrete stimulus cases
under that shared interface. Clock-like and reset-like signals are inferred from the majority
candidate interface/code so later driver generation can treat them specially without requiring
the caller to annotate ports manually.

The generated stimulus is executed against each candidate through an auto-generated Verilog
driver, producing `TBout` traces via `iverilog` + `vvp`. Combinational designs run all cases in
one simulation. Sequential designs run one simulator instance per case: case boundaries clear
non-clock/non-reset inputs, reset defaults inactive during normal cycles, and the driver
automatically appends one reset-check tail at the end of each case instead of inserting a hidden
reset preamble at the start. Those traces are compared to build functional clusters. The final
output is intentionally compact for downstream reuse: either a single representative Verilog, or
two representative Verilogs together with their differing input and output trajectories under the
same stimulus set.
"""

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
import ast
import json
import re
import subprocess
from importlib import import_module

from .entropy import semantic_entropy_from_counts

DEFAULT_NUM_CANDIDATES = 8
DEFAULT_MAX_STIMULI = 5  # LLMs can struggle to keep quality high when asked for many cases, so we cap the count and rely on careful scenario design to hit coverage targets.
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 600
DEFAULT_TOP_MODULE_NAME = "TopModule"
DEFAULT_STIMULUS_REFINEMENT_ROUNDS = 3
STIMULUS_REFINEMENT_ENTROPY_GAIN_TOLERANCE = 1e-9


def _get_runtime_monitor() -> Any | None:
    """Avoid a hard import edge while still allowing deep-phase heartbeat diagnostics."""

    try:
        module = import_module("realbench_safe_pipeline_RTLLM.runtime_monitor")
    except Exception:
        return None
    getter = getattr(module, "get_active_run_monitor", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


VERILOG_SYSTEM_PROMPT = (
    "You are an expert at writing Verilog code. "
    "You generate complete, synthesizable Verilog modules wrapped in ```verilog fences."
)

VERILOG_USER_PROMPT = """Write a complete Verilog module for the specification below.

Specification:
{spec}

Required top-level module name:
{top_module_name}

Requirements:
- Return complete Verilog code.
- The top-level module must be named exactly `{top_module_name}`.
- You may define helper modules if needed.
- The top-level module must use ANSI-style port declarations.
- Wrap the final code in ```verilog fences.
- Keep the top-level interface faithful to the specification.
- For pulse-detection tasks like a continuous 0->1->0 stream, do not implement the output pulse as a clock-edge-registered `data_out` assignment inside the detection state transition itself; instead make the pulse externally visible in the same observable completion cycle by deriving `data_out` from previously sampled history together with the current input level.
- For serial-to-parallel or byte-assembly tasks, when the final accepted serial bit completes a word, the completed parallel word and its completion indication must follow the specification's boundary-cycle semantics directly. Do not hide the completion event behind an extra delayed stage or an effectively unobservable internal pulse unless the specification explicitly requires that delay.
- For parallel-to-serial tasks, once a parallel word is accepted for transmission, the externally visible serial stream must begin with the correct first payload bit under the specification's stated timing, without inserting a spurious startup bit, skipping the first payload bit, or shifting the whole stream by one cycle unless the specification explicitly requires such staging.
- For width-conversion tasks that accumulate multiple narrower inputs into one wider output, the output-valid event belongs to the boundary at which the final required input piece has been accepted according to the specification. Do not add an extra output-register delay after the aggregation boundary unless the specification explicitly requires that additional cycle.
- If the specification does not explicitly name the top-level module, use `{top_module_name}`.
"""






TOP_MODULE_NAME_SYSTEM_PROMPT = (
    "You extract the required Verilog top-level module name from a specification."
)

TOP_MODULE_NAME_USER_PROMPT = """Extract the required top-level Verilog module name from the specification below.

Specification:
{spec}

Return only JSON with this exact schema:
{{"top_module_name": "<exact module name from the specification>"}}

Rules:
- Use the exact top module name specified in the problem/specification.
- If the specification does not explicitly name the top-level module, return {{"top_module_name": "TopModule"}}.
- top_module_name must be exactly one valid Verilog identifier.
- Do not include explanation, markdown, or code fences.
"""






SEQ_CONTROL_SYSTEM_PROMPT = (
    "You extract sequential control signal information from a Verilog specification."
)

SEQ_CONTROL_USER_PROMPT = """Extract sequential control information from the specification and majority interface below.

Specification:
{spec}

Majority interface:
{interface_signature}

Return valid JSON wrapped in triple single quotes:
'''
{{
  "clock_name": "<clock input name>",
  "clock_edge": "posedge or negedge",
  "reset_name": "<reset input name or null>",
  "reset_active_level": 0 or 1,
  "reset_style": "async or sync or none"
}}
'''

Rules:
- Return only the JSON payload wrapped in triple single quotes: '''...'''
- Use only input port names that appear in the majority interface.
- `clock_name` must be the exact clock input name in the majority interface.
- `clock_edge` must be either "posedge" or "negedge".
- If there is no reset signal, still return `clock_name` and `clock_edge`, and use:
  {{
  "reset_name": null,
  "reset_active_level": null,
  "reset_style": "none"
  }}
- If there is a reset signal:
  - `reset_name` must be the exact reset input name in the majority interface.
  - `reset_active_level` must be 1 for active-high reset or 0 for active-low reset.
  - `reset_style` must be "sync" or "async".
- Do not explain.
"""






STIMULUS_SYSTEM_PROMPT = (
    "You generate structured simulation stimuli for Verilog modules."
)


SEQ_STIMULUS_USER_PROMPT = """Given this Verilog specification and majority interface, {case_count_instruction} sequential stimulus cases.

Specification:
{spec}

Majority interface:
{interface_signature}

Circuit type hint:
{circuit_type}

Sequential control contract:
{seq_control_blob}

Verification scenarios:
{scenario_blob}

Return valid JSON in this shape:
'''
[
  {{
    "name": "case_name",
    "description": "what this independent temporal testcase covers",
    "reason_based_on_spec": "brief explanation of why this input sequence was chosen based on the specification",
    "cycles": [
      {{
        "inputs": {{"signal_name": value}},
        "sample": true
      }}
    ]
  }}
]
'''

Definitions:
- A stimulus case is one complete independent temporal testcase.
- Each verification scenario must be realized as exactly one corresponding stimulus case.
- Each stimulus case will be simulated independently from other cases.
- Do not rely on state carrying over from one case to another.
- Each cycle entry represents one logical clock cycle within that case.
- For sequential behavior, include multiple cycles whenever the scenario depends on state evolution, delayed effects, counting, rollover, pattern accumulation, or any behavior that cannot be exercised in a single cycle.
- Do not create a separate case for every single input value if those values are meant to form one temporal sequence.

Rules:
- Return only the JSON payload wrapped in triple single quotes: '''...'''
- Use only ports from the majority interface.
- Never include output ports inside "inputs".
- Put only ordinary non-clock, non-reset input values in "inputs".
- Never drive the clock manually in "inputs"; the driver will generate the clock.
- Do not include reset inputs unless the scenario genuinely needs reset to initialize or verify reset-sensitive behavior.
- If reset is used, keep it explicit and minimal; do not sprinkle reset pulses throughout unrelated cycles.
- If a reset signal exists, the driver will append one reset-check step at the end of each case.
- Generate only input stimulus cycles; do not include expected outputs, answers, or implementation commentary.
- Provide `reason_based_on_spec` for each testcase to explain why the input sequence is justified by the specification.
- Within one sequential case, inputs are stateful across cycles: any omitted non-clock/non-reset input keeps its previous cycle value.
- Explicitly assign a non-clock/non-reset input in a cycle only when you want to change it or return it to 0.
- Values must be JSON booleans, JSON integers, or Verilog-style numeric strings such as "4'b1010", "8'hFF", "1'b0". Do not use words like "high", "low", "true_signal", or symbolic expressions.
- The driver will sample normal cycles after the active clock edge and `#1`.
- Cover hold behavior, one-step transitions, and multi-cycle state evolution when applicable.
- Each stimulus case must fully cover the single verification scenario it is derived from; if that scenario describes a multi-cycle event or long-horizon behavior, include enough cycles for that behavior to actually occur and be sampled in simulation.
"""





CMB_STIMULUS_USER_PROMPT = """Given this Verilog specification and majority interface, {case_count_instruction} combinational stimulus cases.

Specification:
{spec}

Majority interface:
{interface_signature}

Circuit type hint:
{circuit_type}

Verification scenarios:
{scenario_blob}

Return valid JSON in this shape:
'''
[
  {{
    "name": "case_name",
    "description": "what this case covers",
    "reason_based_on_spec": "brief explanation of why this input test was chosen based on the specification",
    "steps": [
      {{
        "sets": {{"signal_name": value}},
        "sample": true
      }}
    ]
  }}
]
'''

Definitions:
- A stimulus case is one independent combinational verification testcase or testing theme.
- Each step is one independent input vector application.
- A case may contain one or more steps when those input vectors belong to the same testing theme.
- For small truth tables, related input vectors may be grouped into one case.
- Each stimulus case must concretely cover the input conditions described in its verification scenario; do not describe a condition in the scenario unless the case includes explicit input vectors that exercise it.

Rules:
- Return only the JSON payload wrapped in triple single quotes: '''...'''
- Use only ports from the majority interface.
- Include output sampling points with "sample": true.
- Never include output ports inside "sets".
- Provide `reason_based_on_spec` for each testcase to explain why the input pattern is justified by the specification.
- Values must be JSON booleans, JSON integers, or Verilog-style numeric strings such as "4'b1010", "8'hFF", "1'b0". Do not use words like "high", "low", "true_signal", or symbolic expressions.
- Each step represents one input vector application followed by optional sampling.
- The driver will wait a fixed `#10` time units after applying each combinational input vector before sampling outputs.
- When the scenario concerns a Boolean condition, selector behavior, priority rule, or don't-care-sensitive logic, include the specific input vectors that distinguish nearby or easily confused cases, not just representative easy examples.
- If no explicit count is imposed, generate at most 8 non-redundant high-value stimulus cases.
"""


STIMULUS_REFINEMENT_SYSTEM_PROMPT = (
    "You refine structured Verilog simulation stimuli to improve scenario coverage."
)


STIMULUS_REFINEMENT_USER_PROMPT = """Given this Verilog specification, target verification scenario, current stimulus case, and the current output trajectory produced under that stimulus, first decide whether the current stimulus already covers the target scenario well enough. If coverage is already adequate, return the original stimulus case unchanged. Only refine the stimulus when coverage is incomplete.

Specification:
{spec}

Majority interface:
{interface_signature}

Circuit type hint:
{circuit_type}

Sequential control contract:
{seq_control_blob}

Target verification scenario (including target requirement context when available):
{scenario_blob}

Current stimulus case:
{stimulus_case_blob}

Current top-one-cluster output trajectory:
{output_trajectory_blob}

Return valid JSON wrapped in triple single quotes using exactly one stimulus case object:
'''
{schema_blob}
'''

Rules:
- First judge whether the current output trajectory already appears to begin in the starting state or precondition expected by the target scenario and target requirement.
- If the current stimulus does not appear to reach that required starting state, add only the necessary prefix input steps needed to drive the design into that starting state before testing the target behavior.
- After the starting state is established, check whether the remaining stimulus adequately covers the target scenario.
- If the current stimulus already reaches the correct starting state and already adequately covers the target scenario, return the original stimulus case unchanged in the same JSON schema.
- Return only the JSON payload wrapped in triple single quotes: '''...'''
- Preserve the original testing goal and the key event order of the current stimulus case.
- Improve coverage only when needed by making the stimulus more specific, more complete, or longer.
- Judge coverage based on the full observed output trajectory, not just whether the starting state is reached. Check whether the current stimulus fully exercises the target scenario's required behavior, including the needed precondition, the key triggering event, and the observable consequence or follow-up behavior. If any of these are missing, refine the stimulus to fill the missing coverage.- Use the output trajectory only as coverage feedback about what the current stimulus appears to exercise; do not treat it as authoritative truth and do not try to hard-code outputs into the stimulus.
- Use only ports from the majority interface.
- Never include output ports in the stimulus inputs/sets.
- For sequential circuits, never drive the clock manually in the stimulus inputs; the driver will generate the clock.
- For sequential circuits, do not include reset inputs unless they are genuinely needed for the scenario.
- Keep the refined stimulus focused on the same target scenario rather than changing to a different behavior goal.
- If the current stimulus already adequately covers the target scenario, return the original stimulus case unchanged instead of inventing extra steps.
- Values must be JSON booleans, JSON integers, or Verilog-style numeric strings such as "4'b1010", "8'hFF", "1'b0".
"""


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
        temperature: float = 0.8,
    ) -> List[str]:
        """Abstract LLM entrypoint so the pipeline can swap real and demo backends."""
        """Return n string responses from an LLM."""


@dataclass
class Port:
    direction: str
    name: str
    width_text: str = ""
    signed: bool = False
    net_type: str = ""

    def normalized_width(self) -> str:
        """Canonicalize width text so interface comparisons ignore spacing noise."""
        return _normalize_whitespace(self.width_text)


@dataclass
class InterfaceInfo:
    module_name: str
    ports: List[Port]
    raw_header: str
    parameter_defaults: Dict[str, str]
    signature_text: str
    parse_success: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def input_ports(self) -> List[Port]:
        """Expose input ports directly because later stimulus generation only drives inputs."""
        return [port for port in self.ports if port.direction == "input"]

    @property
    def output_ports(self) -> List[Port]:
        """Expose output ports directly because later trace comparison only reads outputs."""
        return [port for port in self.ports if port.direction == "output"]


@dataclass
class VerilogCandidate:
    candidate_id: int
    raw_response: str
    verilog_code: str
    interface: InterfaceInfo
    inferred_circuit_type: str
    interface_cluster_id: Optional[str] = None
    is_majority_interface: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfaceCluster:
    cluster_id: str
    signature_text: str
    candidates: List[VerilogCandidate]
    inferred_circuit_types: Dict[str, int]


@dataclass
class StimulusStep:
    sets: Dict[str, Any]
    delay: int = 10
    sample: bool = False
    sample_phase: str = "after_clock"


@dataclass
class StimulusCase:
    name: str
    description: str
    steps: List[StimulusStep]
    reason_based_on_spec: str = ""


@dataclass
class VerificationScenario:
    name: str
    goal: str
    story: str
    target_requirement_id: Optional[int] = None
    target_requirement: str = ""



@dataclass
class SimulationRecord:
    candidate_id: int
    interface_cluster_id: Optional[str]
    simulation_dir: Optional[str]
    compile_ok: bool
    run_ok: bool
    tbout_path: Optional[str]
    trace: List[Dict[str, Any]]
    functional_signature: Optional[Tuple[Tuple[Any, ...], ...]]
    error: str = ""


@dataclass
class FunctionalCluster:
    cluster_id: str
    score: int
    candidate_ids: List[int]
    representative_candidate_id: Optional[int] = None


@dataclass
class ResetInfo:
    signal_name: Optional[str]
    style: str
    active_level: int = 1


@dataclass
class SeqControlContract:
    clock_name: str
    clock_edge: str
    reset_name: Optional[str] = None
    reset_active_level: Optional[int] = None
    reset_style: str = "none"


class MultiClockSeqControlContractError(RuntimeError):
    """Raised when the seq-control extractor returns multiple clock/reset domains."""

    def __init__(self, payload: list[Any]):
        super().__init__("seq control payload indicates multiple clock/reset domains")
        self.payload = payload


def remove_comments(verilog_code: str) -> str:
    """Strip Verilog comments so downstream regex parsing sees only structural code. 因为后面很多解析逻辑是靠 正则和字符串扫描 在读 Verilog，不是真正的 Verilog 语法解析器。"""
    code_without_single_comments = re.sub(r"//.*", "", verilog_code)
    return re.sub(r"/\*[\s\S]*?\*/", "", code_without_single_comments)



def extract_verilog_code(text: str) -> str:
    """Recover Verilog code from an LLM response that may contain prose or one/multiple fences."""
    fenced_blocks = re.findall(
        r"```(?:verilog|systemverilog|sv)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if fenced_blocks:
        return "\n\n".join(
            block.strip()
            for block in fenced_blocks
            if block.strip()
        )

    start = re.search(r"\bmodule\b", text, re.IGNORECASE)
    end_matches = list(re.finditer(r"\bendmodule\b", text, re.IGNORECASE))
    if start and end_matches:
        return text[start.start() : end_matches[-1].end()].strip()

    return text.strip()


def _extract_triple_single_quoted_text(text: str) -> str:
    """Extract the payload inside triple single quotes for lightweight structured prompts."""
    match = re.search(r"'''[\r\n\s]*(.*?)[\r\n\s]*'''", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _is_valid_verilog_identifier(name: str) -> bool:
    """Validate that an extracted module name is a single legal Verilog identifier."""
    return bool(re.fullmatch(r"[A-Za-z_]\w*", name.strip()))


def _normalize_whitespace(text: str) -> str:
    """Make text comparisons robust by collapsing arbitrary spacing into a stable form."""
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_json_loads(text: str) -> Any:
    """Parse LLM-emitted structured text through a tolerant JSON extraction pipeline."""
    def _try_load(candidate: str) -> Any:
        """Validate one candidate payload and repair common near-JSON formatting mistakes."""
        candidate = candidate.strip()
        if not candidate:
            raise json.JSONDecodeError("empty candidate", candidate, 0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as first_error:
            cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
            cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pythonish = re.sub(r"\btrue\b", "True", cleaned, flags=re.IGNORECASE)
                pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
                pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
                try:
                    return ast.literal_eval(pythonish)
                except (ValueError, SyntaxError):
                    raise first_error

    def _extract_balanced_json_snippet(source: str) -> str | None:
        """Pull out the first balanced object/array when the response contains extra wrapper text."""
        start = None
        opening = ""
        closing = ""
        for index, char in enumerate(source):
            if char == "[":
                start = index
                opening = "["
                closing = "]"
                break
            if char == "{":
                start = index
                opening = "{"
                closing = "}"
                break
        if start is None:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        return None

    candidates: List[str] = []
    triple_quoted = re.search(r"'''[\r\n\s]*(.*?)[\r\n\s]*'''", text, re.DOTALL)
    if triple_quoted:
        candidates.append(triple_quoted.group(1))
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    balanced = _extract_balanced_json_snippet(text)
    if balanced:
        candidates.append(balanced)
    candidates.append(text)

    deduped_candidates: List[str] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen_candidates:
            continue
        seen_candidates.add(normalized)
        deduped_candidates.append(candidate)

    last_error: json.JSONDecodeError | None = None
    for candidate in deduped_candidates:
        try:
            return _try_load(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("unable to find JSON payload", text, 0)


def _split_top_level_commas(text: str) -> List[str]:
    """Split module port lists safely without breaking on commas nested inside widths or defaults."""
    parts: List[str] = []
    current: List[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0

    for char in text:
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(depth_paren - 1, 0)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket = max(depth_bracket - 1, 0)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(depth_brace - 1, 0)

        if char == "," and depth_paren == depth_bracket == depth_brace == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching_paren(text: str, open_index: int) -> int:
    """Find the matching closing parenthesis so parameter and port blocks can be skipped safely."""
    depth = 0
    for idx in range(open_index, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_module_header(
    verilog_code: str,
    target_module_name: Optional[str] = None,
) -> Tuple[str, str, str, List[str]]:
    """Locate module name, optional parameter block, and raw port header before finer parsing."""
    clean_code = remove_comments(verilog_code)
    warnings: List[str] = []
    if target_module_name:
        match = re.search(
            rf"\bmodule\s+({re.escape(target_module_name)})\b",
            clean_code,
        )
        if not match:
            return (
                target_module_name,
                "",
                "",
                [f"target top module `{target_module_name}` not found"],
            )
    else:
        match = re.search(r"\bmodule\s+([A-Za-z_]\w*)", clean_code)
    if not match:
        return "", "", "", ["module declaration not found"]

    module_name = match.group(1)
    param_header = ""
    pos = match.end()
    while pos < len(clean_code) and clean_code[pos].isspace():
        pos += 1

    if pos < len(clean_code) and clean_code[pos] == "#":
        pos += 1
        while pos < len(clean_code) and clean_code[pos].isspace():
            pos += 1
        if pos >= len(clean_code) or clean_code[pos] != "(":
            return module_name, "", "", ["module parameter list not found after #"]
        param_end = _find_matching_paren(clean_code, pos)
        if param_end == -1:
            return module_name, "", "", ["module parameter list not closed"]
        param_header = clean_code[pos : param_end + 1]
        pos = param_end + 1

    while pos < len(clean_code) and clean_code[pos].isspace():
        pos += 1
    if pos >= len(clean_code) or clean_code[pos] != "(":
        return module_name, "", param_header, ["module port list not found"]

    start = pos
    end = _find_matching_paren(clean_code, start)
    if end == -1:
        return module_name, "", param_header, ["module port list not closed"]

    port_block = clean_code[start + 1 : end]
    header = clean_code[start : end + 1]
    if re.search(r"\binput\b|\boutput\b|\binout\b", port_block) is None:
        warnings.append("non-ANSI or directionless port list detected")
    return module_name, header, param_header, warnings


def _parse_module_parameter_defaults(param_header: str) -> Dict[str, str]:
    """Extract simple parameter default expressions so generated drivers can mirror module widths."""
    if not param_header:
        return {}
    block = param_header[param_header.find("(") + 1 : param_header.rfind(")")]
    defaults: Dict[str, str] = {}
    for entry in _split_top_level_commas(block):
        normalized = _normalize_whitespace(entry)
        if "=" not in normalized:
            continue
        left, right = normalized.split("=", 1)
        left = re.sub(r"^(parameter|localparam)\b", "", left.strip()).strip()
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", left)
        if not name_match:
            continue
        defaults[name_match.group(1)] = right.strip()
    return defaults


def parse_verilog_interface(
    verilog_code: str,
    target_module_name: Optional[str] = None,
) -> InterfaceInfo:
    """Turn a Verilog module header into structured port metadata for clustering and driving."""
    module_name, raw_header, param_header, warnings = _extract_module_header(
        verilog_code,
        target_module_name=target_module_name,
    )
    parameter_defaults = _parse_module_parameter_defaults(param_header)
    if not raw_header:
        return InterfaceInfo(
            module_name=module_name or "unknown_module",
            ports=[],
            raw_header=raw_header,
            parameter_defaults=parameter_defaults,
            signature_text="",
            parse_success=False,
            warnings=warnings,
        )

    port_block = raw_header[raw_header.find("(") + 1 : raw_header.rfind(")")]
    entries = _split_top_level_commas(port_block)
    ports: List[Port] = []
    last_direction: Optional[str] = None
    last_width = ""
    last_signed = False
    last_net_type = ""

    for entry in entries:
        normalized = _normalize_whitespace(entry)
        if not normalized:
            continue

        direction_match = re.match(r"^(input|output|inout)\b", normalized)
        if direction_match:
            last_direction = direction_match.group(1)
            rest = normalized[direction_match.end() :].strip()
            signed_match = re.search(r"\bsigned\b", rest)
            last_signed = bool(signed_match)
            rest = re.sub(r"\bsigned\b", "", rest).strip()

            net_match = re.match(r"^(wire|reg|logic)\b", rest)
            if net_match:
                last_net_type = net_match.group(1)
                rest = rest[net_match.end() :].strip()
            else:
                last_net_type = ""

            width_match = re.match(r"^(\[[^\]]+\])", rest)
            if width_match:
                last_width = width_match.group(1)
                rest = rest[width_match.end() :].strip()
            else:
                last_width = ""
            name = rest.split("=")[0].strip()
        else:
            name = normalized.split("=")[0].strip()

        if not last_direction or not name:
            warnings.append(f"unparsed port entry: {normalized}")
            continue

        name = name.rstrip(");")
        ports.append(
            Port(
                direction=last_direction,
                name=name,
                width_text=last_width,
                signed=last_signed,
                net_type=last_net_type,
            )
        )

    parse_success = len(ports) > 0
    signature_lines = [
        f"{port.direction} {port.net_type} {'signed ' if port.signed else ''}{port.normalized_width()} {port.name}".strip()
        for port in sorted(
            ports,
            key=lambda port: (
                port.direction,
                port.name,
                port.normalized_width(),
                "signed" if port.signed else "unsigned",
                port.net_type,
            ),
        )
    ]
    signature_text = "\n".join(
        line.replace("  ", " ").strip() for line in signature_lines
    )

    return InterfaceInfo(
        module_name=module_name,
        ports=ports,
        raw_header=raw_header,
        parameter_defaults=parameter_defaults,
        signature_text=signature_text,
        parse_success=parse_success,
        warnings=warnings,
    )


def infer_circuit_type(verilog_code: str, interface: Optional[InterfaceInfo] = None) -> str:
    """Classify a candidate as SEQ or CMB so later testcase and driver templates can branch."""
    clean_code = remove_comments(verilog_code)
    if re.search(r"\b(posedge|negedge)\b", clean_code):
        return "SEQ"
    if re.search(r"\balways_ff\b", clean_code):
        return "SEQ"
    if re.search(r"\balways_latch\b", clean_code):
        return "CMB"
    if re.search(r"\balways_comb\b|assign\b|@\s*\(\s*\*\s*\)", clean_code):
        return "CMB"
    if interface and _candidate_clock_names(interface):
        return "SEQ"
    return "UNKNOWN"


def _name_tokens(name: str) -> List[str]:
    """Split a signal name into lowercase tokens so embedded rst/clk markers remain detectable."""
    return [token for token in re.split(r"[_\W]+", name.lower()) if token]


def _candidate_clock_names(interface: InterfaceInfo) -> List[str]:
    """Rank likely clock inputs. Any input containing clk/clock is treated as clock-like."""
    ranked: List[Tuple[int, str]] = []
    for port in interface.input_ports:
        lower = port.name.lower()
        score = 0
        if lower in {"clk", "clock"}:
            score += 10
        elif "clock" in lower:
            score += 8
        elif "clk" in lower:
            score += 8
        if score > 0:
            ranked.append((-score, port.name))
    ranked.sort()
    return [name for _, name in ranked]


def _candidate_reset_names(interface: InterfaceInfo) -> List[str]:
    """Rank likely reset inputs so semantic parsing can prefer real reset signals over generic inputs."""
    ranked: List[Tuple[int, str]] = []
    for port in interface.input_ports:
        lower = port.name.lower()
        tokens = _name_tokens(port.name)
        if "clk" in lower or "clock" in lower:
            continue
        score = 0
        is_reset_like = (
            "reset" in lower
            or "rst" in tokens
            or lower.startswith("rst")
            or lower.startswith("arst")
            or lower.startswith("areset")
        )
        if is_reset_like:
            score += 4
            if lower.startswith("a") or lower.startswith("arst") or lower.startswith("areset"):
                score += 1
            if lower.endswith("_n") or lower.endswith("n") or lower.endswith("_ni"):
                score += 1
        if score > 0:
            ranked.append((-score, port.name))
    ranked.sort()
    return [name for _, name in ranked]


def _infer_reset_active_level_from_if(clean_code: str, reset_name: str) -> Optional[int]:
    """Infer active-high vs active-low reset from common synchronous `if (...)` guard patterns."""
    name = re.escape(reset_name)
    active_low_patterns = [
        rf"\bif\s*\(\s*!\s*{name}\s*\)",
        rf"\bif\s*\(\s*~\s*{name}\s*\)",
        rf"\bif\s*\(\s*{name}\s*(?:==|===)\s*(?:1'?b0|1'?d0|0)\s*\)",
        rf"\bif\s*\(\s*(?:1'?b0|1'?d0|0)\s*(?:==|===)\s*{name}\s*\)",
    ]
    active_high_patterns = [
        rf"\bif\s*\(\s*{name}\s*\)",
        rf"\bif\s*\(\s*{name}\s*(?:==|===)\s*(?:1'?b1|1'?d1|1)\s*\)",
        rf"\bif\s*\(\s*(?:1'?b1|1'?d1|1)\s*(?:==|===)\s*{name}\s*\)",
    ]

    if any(re.search(pattern, clean_code) for pattern in active_low_patterns):
        return 0
    if any(re.search(pattern, clean_code) for pattern in active_high_patterns):
        return 1
    return None


def detect_reset_info(verilog_code: str, interface: InterfaceInfo) -> ResetInfo:
    """Infer reset semantics from always blocks first, then fall back to naming conventions if needed."""
    clean_code = remove_comments(verilog_code)
    reset_candidates = _candidate_reset_names(interface)
    if not reset_candidates:
        return ResetInfo(signal_name=None, style="none", active_level=1)

    always_headers = re.findall(
        r"\balways(?:_ff|_comb|_latch)?\s*@\s*\((.*?)\)",
        clean_code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for reset_name in reset_candidates:
        async_match = next(
            (
                header
                for header in always_headers
                if re.search(rf"\b(?:posedge|negedge)\s+{re.escape(reset_name)}\b", header)
            ),
            None,
        )
        if async_match is not None:
            active_level = 0 if re.search(rf"\bnegedge\s+{re.escape(reset_name)}\b", async_match) else 1
            return ResetInfo(signal_name=reset_name, style="async", active_level=active_level)

    for reset_name in reset_candidates:
        active_level = _infer_reset_active_level_from_if(clean_code, reset_name)
        if active_level is not None:
            return ResetInfo(signal_name=reset_name, style="sync", active_level=active_level)

    fallback_name = reset_candidates[0]
    fallback_active_level = 0 if fallback_name.lower().endswith("_n") else 1
    return ResetInfo(signal_name=fallback_name, style="sync", active_level=fallback_active_level)


def detect_sample_edge(verilog_code: str, interface: InterfaceInfo) -> str:
    """Infer the sampling edge only from recognized clock inputs, not from unrelated reset edges."""
    clean_code = remove_comments(verilog_code)
    clock_names = _candidate_clock_names(interface)
    for clock_name in clock_names:
        if re.search(rf"\bnegedge\s+{re.escape(clock_name)}\b", clean_code):
            return "negedge"
        if re.search(rf"\bposedge\s+{re.escape(clock_name)}\b", clean_code):
            return "posedge"
    return "posedge"


def normalize_interface_signature(interface: InterfaceInfo) -> str:
    """Reduce an interface to a stable order-insensitive signature for majority-interface clustering."""
    if not interface.parse_success:
        return ""
    parts = []
    for port in sorted(
        interface.ports,
        key=lambda port: (
            port.direction,
            port.name,
            port.normalized_width(),
            "signed" if port.signed else "unsigned",
        ),
    ):
        parts.append(
            "|".join(
                [
                    port.direction,
                    port.name,
                    port.normalized_width(),
                    "signed" if port.signed else "unsigned",
                ]
            )
        )
    return "\n".join(parts)


def output_signature_from_trace(
    trace: Sequence[Dict[str, Any]],
    output_port_names: Sequence[str],
) -> Tuple[Tuple[Any, ...], ...]:
    """Project a full trace into a compact output signature used for coarse behavior tracking."""
    signature: List[Tuple[Any, ...]] = []
    for entry in filter_trace_rows_without_unknown_outputs(trace, output_port_names):
        row: List[Any] = [entry.get("case"), entry.get("step")]
        for port_name in output_port_names:
            row.append(entry.get(port_name))
        signature.append(tuple(row))
    return tuple(signature)


def compact_input_trajectory(
    trace: Sequence[Dict[str, Any]],
    input_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Summarize the actual simulated input evolution so auto-appended reset tails are visible too."""
    input_name_set = set(input_names)
    compact_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(trace):
        inputs = {
            key: value
            for key, value in row.items()
            if key in input_name_set
        }
        compact_rows.append(
            {
                "index": index,
                "inputs": inputs,
            }
        )
    return compact_rows


def compact_output_trajectory(
    trace: Sequence[Dict[str, Any]],
    input_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Summarize a trace as only output evolution so final reports stay focused and compact."""
    meta_keys = {"case", "step", "cycle", "clk", "clock"}
    input_name_set = set(input_names)
    compact_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(trace):
        outputs = {
            key: value
            for key, value in row.items()
            if key not in meta_keys and key not in input_name_set
        }
        compact_rows.append(
            {
                "index": index,
                "outputs": outputs,
            }
        )
    return compact_rows


def output_value_has_unknown(value: Any) -> bool:
    """Treat any textual output value containing x as unknown for clustering/comparison."""

    if isinstance(value, str):
        return "x" in value.lower()
    return False


def row_has_unknown_outputs(
    row: Mapping[str, Any],
    output_port_names: Sequence[str],
) -> bool:
    """Whether one sampled row contains unknown output bits on any compared output port."""

    for port_name in output_port_names:
        if output_value_has_unknown(row.get(port_name)):
            return True
    return False


def filter_trace_rows_without_unknown_outputs(
    trace: Sequence[Dict[str, Any]],
    output_port_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Drop sampled rows whose compared outputs contain unknown bits."""

    return [
        row
        for row in trace
        if not row_has_unknown_outputs(row, output_port_names)
    ]


def _trace_row_key(row: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    """Stable per-row key for pairwise masking and comparison."""

    return (row.get("case"), row.get("step"), row.get("cycle"))


def mask_pairwise_unknown_output_rows(
    left_trace: Sequence[Dict[str, Any]],
    right_trace: Sequence[Dict[str, Any]],
    output_port_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove any sampled row where either candidate shows unknown outputs on compared ports."""

    masked_keys: set[Tuple[Any, Any, Any]] = set()
    for trace in (left_trace, right_trace):
        for row in trace:
            if row_has_unknown_outputs(row, output_port_names):
                masked_keys.add(_trace_row_key(row))

    filtered_left = [row for row in left_trace if _trace_row_key(row) not in masked_keys]
    filtered_right = [row for row in right_trace if _trace_row_key(row) not in masked_keys]
    return filtered_left, filtered_right


def summarize_pairwise_trace_difference(
    left_trace: Sequence[Dict[str, Any]],
    right_trace: Sequence[Dict[str, Any]],
    output_port_names: Sequence[str],
) -> Dict[str, Any]:
    """Summarize differing rows and the first few observable output divergences."""

    left_by_key = {_trace_row_key(row): row for row in left_trace}
    right_by_key = {_trace_row_key(row): row for row in right_trace}
    differing_row_indices: List[Any] = []
    earliest_difference: Optional[Dict[str, Any]] = None
    differing_outputs_by_row: List[Dict[str, Any]] = []

    common_keys = sorted(set(left_by_key) & set(right_by_key), key=lambda item: (item[0], item[1], item[2]))
    for row_key in common_keys:
        left_row = left_by_key[row_key]
        right_row = right_by_key[row_key]
        output_differences: Dict[str, Dict[str, Any]] = {}
        for port_name in output_port_names:
            left_value = left_row.get(port_name)
            right_value = right_row.get(port_name)
            if left_value != right_value:
                output_differences[port_name] = {
                    "candidate_a": left_value,
                    "candidate_b": right_value,
                }
        if not output_differences:
            continue

        row_index = left_row.get("step", right_row.get("step"))
        differing_row_indices.append(row_index)
        if earliest_difference is None:
            earliest_difference = {
                "row_index": row_index,
                "cycle": left_row.get("cycle", right_row.get("cycle")),
            }
        if len(differing_outputs_by_row) < 3:
            differing_outputs_by_row.append(
                {
                    "row_index": row_index,
                    "cycle": left_row.get("cycle", right_row.get("cycle")),
                    "differences": output_differences,
                }
            )

    return {
        "differing_row_indices": differing_row_indices,
        "earliest_difference": earliest_difference,
        "differing_outputs_by_row": differing_outputs_by_row,
    }


def _is_clock_like_name(name: str) -> bool:
    """Recognize clock-like names so human-facing summaries can hide timing signals."""
    lowered = name.strip().lower()
    return "clk" in lowered or "clock" in lowered


def trace_io_rows(
    trace: Sequence[Dict[str, Any]],
    interface: InterfaceInfo,
    circuit_type: str,
) -> Dict[Tuple[int, int], Tuple[Any, ...]]:
    """Convert a raw trace into comparable per-timepoint input/output rows for clustering."""
    clock_names = set(_candidate_clock_names(interface))
    input_ports = [port.name for port in interface.input_ports if port.name not in clock_names]
    output_ports = [port.name for port in interface.output_ports]
    time_field = "cycle" if circuit_type == "SEQ" else "step"
    rows: Dict[Tuple[int, int], Tuple[Any, ...]] = {}
    for entry in trace:
        case_idx = entry.get("case")
        time_idx = entry.get(time_field)
        if not isinstance(case_idx, int) or not isinstance(time_idx, int):
            continue
        row: List[Any] = []
        for port_name in input_ports:
            row.append(entry.get(port_name))
        for port_name in output_ports:
            row.append(entry.get(port_name))
        rows[(case_idx, time_idx)] = tuple(row)
    return rows


def _rows_compatible_ignoring_missing_keys(
    left_rows: Mapping[Tuple[int, int], Tuple[Any, ...]],
    right_rows: Mapping[Tuple[int, int], Tuple[Any, ...]],
) -> bool:
    """Treat rows as compatible when every commonly defined sampled row matches."""

    for row_key in set(left_rows) & set(right_rows):
        if left_rows[row_key] != right_rows[row_key]:
            return False
    return True


def case_signature_from_trace(
    case_trace: Sequence[Dict[str, Any]],
    output_port_names: Sequence[str],
) -> Tuple[Tuple[Any, ...], ...]:
    """Build a per-testcase output signature so one testcase can form its own mini cluster view."""
    signature: List[Tuple[Any, ...]] = []
    for entry in filter_trace_rows_without_unknown_outputs(case_trace, output_port_names):
        row: List[Any] = [entry.get("step"), entry.get("cycle")]
        for port_name in output_port_names:
            row.append(entry.get(port_name))
        signature.append(tuple(row))
    return tuple(signature)



def infer_majority_circuit_type_from_cluster(
    cluster: InterfaceCluster,
) -> Tuple[str, Optional[int]]:
    """Infer majority circuit type from candidates inside the selected majority interface cluster."""
    counts: Dict[str, int] = {}

    for candidate in cluster.candidates:
        circuit_type = candidate.inferred_circuit_type
        counts[circuit_type] = counts.get(circuit_type, 0) + 1

    known_counts = {
        circuit_type: count
        for circuit_type, count in counts.items()
        if circuit_type in {"SEQ", "CMB"}
    }

    if known_counts:
        # Tie-breaker: prefer SEQ over CMB because missing a sequential clock is more damaging
        priority = {"SEQ": 2, "CMB": 1}
        majority_type = max(
            known_counts,
            key=lambda circuit_type: (
                known_counts[circuit_type],
                priority[circuit_type],
            ),
        )
    else:
        majority_type = "UNKNOWN"

    source_candidate_id = next(
        (
            candidate.candidate_id
            for candidate in cluster.candidates
            if candidate.inferred_circuit_type == majority_type
        ),
        None,
    )

    return majority_type, source_candidate_id






class VerilogStimulusClusterPipeline:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        shared_cache: Optional[dict[str, Any]] = None,
        max_candidate_workers: int = 2,
        verilog_temperature: float = 0.9,
        stimulus_temperature: float = 0.4,
        iverilog_bin: str = "iverilog",
        vvp_bin: str = "vvp",
        subprocess_timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
        verilog_system_prompt: Optional[str] = None,
        verilog_user_prompt: Optional[str] = None,
    ) -> None:
        """Capture all reusable generation/simulation knobs in one pipeline object."""
        self.llm_client = llm_client
        self.max_candidate_workers = max(1, int(max_candidate_workers or 1))
        self.verilog_temperature = verilog_temperature
        self.stimulus_temperature = stimulus_temperature
        self.iverilog_bin = iverilog_bin
        self.vvp_bin = vvp_bin
        self.subprocess_timeout_seconds = subprocess_timeout_seconds
        self.verilog_system_prompt = verilog_system_prompt or VERILOG_SYSTEM_PROMPT
        self.verilog_user_prompt = verilog_user_prompt or VERILOG_USER_PROMPT
        self.shared_cache = shared_cache if shared_cache is not None else {}
        self.shared_cache.setdefault("top_module_name", None)
        self.shared_cache.setdefault("seq_control_contracts", {})
        self._last_top_module_name_cache_hit = False
        self._last_seq_control_contract_cache_hit = False

    def _reset_call_cache_flags(self) -> None:
        """Track whether this resolve_spec call reused task-level cached metadata."""
        self._last_top_module_name_cache_hit = False
        self._last_seq_control_contract_cache_hit = False

    def extract_top_module_name_from_spec(self, spec: str) -> str:
        """Resolve the required top module name from the specification so helper modules do not confuse parsing."""
        cached_name = self.shared_cache.get("top_module_name")
        if isinstance(cached_name, str) and _is_valid_verilog_identifier(cached_name):
            self._last_top_module_name_cache_hit = True
            return cached_name
        response = self.llm_client.generate(
            system_prompt=TOP_MODULE_NAME_SYSTEM_PROMPT,
            user_prompt=TOP_MODULE_NAME_USER_PROMPT.format(spec=spec),
            n=1,
            temperature=0.0,
        )[0]
        try:
            payload = _safe_json_loads(response)
        except json.JSONDecodeError as exc:
            snippet = response[:400].replace("\n", "\\n")
            raise RuntimeError(
                f"top module name JSON parse failed: {exc}. raw response prefix: {snippet}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"top module name payload must be a JSON object, got: {type(payload).__name__}")
        name = str(payload.get("top_module_name", "")).strip()
        if not _is_valid_verilog_identifier(name):
            name = DEFAULT_TOP_MODULE_NAME
        self.shared_cache["top_module_name"] = name
        self._last_top_module_name_cache_hit = False
        return name

    def extract_seq_control_contract_from_spec(
        self,
        spec: str,
        majority_interface: InterfaceInfo,
    ) -> SeqControlContract:
        """Extract clock/reset contract from the specification after the design is known to be sequential."""
        interface_signature = normalize_interface_signature(majority_interface)
        contract_cache = self.shared_cache.setdefault("seq_control_contracts", {})
        cached_contract = contract_cache.get(interface_signature)
        if isinstance(cached_contract, SeqControlContract):
            self._last_seq_control_contract_cache_hit = True
            return cached_contract
        response = self.llm_client.generate(
            system_prompt=SEQ_CONTROL_SYSTEM_PROMPT,
            user_prompt=SEQ_CONTROL_USER_PROMPT.format(
                spec=spec,
                interface_signature=interface_signature,
            ),
            n=1,
            temperature=0.0,
        )[0]

        try:
            payload = _safe_json_loads(response)
        except json.JSONDecodeError as exc:
            snippet = response[:1200].replace("\n", "\\n")
            raise RuntimeError(
                f"seq control JSON parse failed: {exc}. raw response prefix: {snippet}"
            ) from exc

        if isinstance(payload, list):
            if payload and all(isinstance(item, dict) for item in payload):
                raise MultiClockSeqControlContractError(payload)
            raise RuntimeError(f"seq control payload must be a JSON object, got: {type(payload).__name__}")

        if not isinstance(payload, dict):
            raise RuntimeError(f"seq control payload must be a JSON object, got: {type(payload).__name__}")

        input_names = {port.name for port in majority_interface.input_ports}

        clock_name = str(payload.get("clock_name", "")).strip()
        if not _is_valid_verilog_identifier(clock_name):
            raise RuntimeError(f"invalid clock_name in seq control contract: {clock_name!r}")
        if clock_name not in input_names:
            raise RuntimeError(f"clock_name `{clock_name}` not found in majority interface inputs")

        clock_edge = str(payload.get("clock_edge", "posedge")).strip().lower()
        if clock_edge not in {"posedge", "negedge"}:
            raise RuntimeError(f"invalid clock_edge in seq control contract: {clock_edge!r}")

        raw_reset_name = payload.get("reset_name")
        if raw_reset_name is None:
            reset_name = None
        else:
            reset_name = str(raw_reset_name).strip()
            if reset_name.lower() in {"", "none", "null", "no_reset", "no reset"}:
                reset_name = None

        reset_style = str(payload.get("reset_style", "none")).strip().lower()
        raw_reset_active_level = payload.get("reset_active_level")

        if reset_name is None:
            contract = SeqControlContract(
                clock_name=clock_name,
                clock_edge=clock_edge,
                reset_name=None,
                reset_active_level=None,
                reset_style="none",
            )
            contract_cache[interface_signature] = contract
            self._last_seq_control_contract_cache_hit = False
            return contract

        if not _is_valid_verilog_identifier(reset_name):
            raise RuntimeError(f"invalid reset_name in seq control contract: {reset_name!r}")
        if reset_name not in input_names:
            raise RuntimeError(f"reset_name `{reset_name}` not found in majority interface inputs")

        if reset_style not in {"sync", "async"}:
            raise RuntimeError(f"invalid reset_style in seq control contract: {reset_style!r}")

        if isinstance(raw_reset_active_level, bool):
            reset_active_level = 1 if raw_reset_active_level else 0
        elif isinstance(raw_reset_active_level, int) and raw_reset_active_level in {0, 1}:
            reset_active_level = raw_reset_active_level
        elif isinstance(raw_reset_active_level, str) and raw_reset_active_level.strip() in {"0", "1"}:
            reset_active_level = int(raw_reset_active_level.strip())
        else:
            raise RuntimeError(
                f"invalid reset_active_level in seq control contract: {raw_reset_active_level!r}"
            )

        contract = SeqControlContract(
            clock_name=clock_name,
            clock_edge=clock_edge,
            reset_name=reset_name,
            reset_active_level=reset_active_level,
            reset_style=reset_style,
        )
        contract_cache[interface_signature] = contract
        self._last_seq_control_contract_cache_hit = False
        return contract

    def seq_control_contract_to_runtime(
        self,
        contract: SeqControlContract,
    ) -> Tuple[str, str, ResetInfo]:
        """Convert extracted seq control contract into driver runtime parameters."""
        if contract.reset_name is None:
            reset_info = ResetInfo(signal_name=None, style="none", active_level=1)
        else:
            reset_info = ResetInfo(
                signal_name=contract.reset_name,
                style=contract.reset_style,
                active_level=int(contract.reset_active_level),
            )

        return contract.clock_name, contract.clock_edge, reset_info    



    def generate_candidates(
        self,
        spec: str,
        *,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        verilog_system_prompt: Optional[str] = None,
        verilog_user_prompt: Optional[str] = None,
        top_module_name: Optional[str] = None,
    ) -> List[VerilogCandidate]:
        """Generate several RTL candidates so later stages can compare interface and behavior diversity."""
        system_prompt = verilog_system_prompt or self.verilog_system_prompt
        user_prompt_template = verilog_user_prompt or self.verilog_user_prompt
        expected_top_module_name = top_module_name or self.extract_top_module_name_from_spec(spec)
        rendered_user_prompt = user_prompt_template.format(
            spec=spec,
            top_module_name=expected_top_module_name,
        )
        monitor = _get_runtime_monitor()
        if monitor is not None:
            monitor.update(
                phase="generate_candidates_start",
                details={
                    "num_candidates": num_candidates,
                    "spec_chars": len(spec),
                    "top_module_name": expected_top_module_name,
                },
            )

        def _generate_one_candidate(candidate_id: int) -> VerilogCandidate:
            if monitor is not None:
                monitor.heartbeat(
                    phase="generate_one_candidate_start",
                    details={
                        "candidate_id": candidate_id,
                        "num_candidates": num_candidates,
                        "spec_chars": len(spec),
                    },
                )
            responses = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=rendered_user_prompt,
                n=1,
                temperature=self.verilog_temperature,
            )
            raw_response = responses[0]
            verilog_code = extract_verilog_code(raw_response)
            interface = parse_verilog_interface(
                verilog_code,
                target_module_name=expected_top_module_name,
            )
            return VerilogCandidate(
                candidate_id=candidate_id,
                raw_response=raw_response,
                verilog_code=verilog_code,
                interface=interface,
                inferred_circuit_type=infer_circuit_type(verilog_code, interface),
                metadata={"expected_top_module_name": expected_top_module_name},
            )

        if num_candidates <= 1:
            return [_generate_one_candidate(0)]

        max_workers = max(1, min(num_candidates, self.max_candidate_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_generate_one_candidate, range(num_candidates)))

    def cluster_interfaces(self, candidates: Sequence[VerilogCandidate]) -> List[InterfaceCluster]:
        """Group candidates by normalized port interface before any shared stimulus is generated."""
        buckets: Dict[str, List[VerilogCandidate]] = {}
        for candidate in candidates:
            signature = normalize_interface_signature(candidate.interface)
            if not signature:
                signature = f"__parse_failed__:{candidate.candidate_id}"
            buckets.setdefault(signature, []).append(candidate)

        def _cluster_sort_key(item: Tuple[str, List[VerilogCandidate]]) -> Tuple[int, bool, str]:
            signature, grouped = item
            is_parse_failed = signature.startswith("__parse_failed__")
            return (-len(grouped), is_parse_failed, signature)

        clusters: List[InterfaceCluster] = []
        for index, (signature, grouped) in enumerate(
            sorted(buckets.items(), key=_cluster_sort_key),
            start=1,
        ):
            cluster_id = f"interface_cluster_{index}"
            counts: Dict[str, int] = {}
            for candidate in grouped:
                counts[candidate.inferred_circuit_type] = counts.get(candidate.inferred_circuit_type, 0) + 1
                candidate.interface_cluster_id = cluster_id
            clusters.append(
                InterfaceCluster(
                    cluster_id=cluster_id,
                    signature_text=signature,
                    candidates=list(grouped),
                    inferred_circuit_types=counts,
                )
            )
        return clusters

    
    def select_majority_interface(
        self,
        clusters: Sequence[InterfaceCluster],
    ) -> Tuple[Optional[InterfaceCluster], Optional[InterfaceInfo], str, Optional[int]]:
        """Choose the dominant parse-success interface cluster as the shared contract."""
        if not clusters:
            return None, None, "UNKNOWN", None

        valid_clusters = [
            cluster
            for cluster in clusters
            if not cluster.signature_text.startswith("__parse_failed__")
        ]
        candidate_clusters = valid_clusters or list(clusters)

        majority_cluster = max(
            candidate_clusters,
            key=lambda cluster: (len(cluster.candidates), cluster.cluster_id),
        )

        for candidate in majority_cluster.candidates:
            candidate.is_majority_interface = True

        # Use the first candidate only for the interface object.
        # Circuit type is inferred by voting across the whole majority interface cluster.
        interface_source_candidate = majority_cluster.candidates[0]
        majority_circuit_type, majority_circuit_type_source_candidate_id = (
            infer_majority_circuit_type_from_cluster(majority_cluster)
        )

        return (
            majority_cluster,
            interface_source_candidate.interface,
            majority_circuit_type,
            majority_circuit_type_source_candidate_id,
        )




    def build_interface_report(
        self,
        candidates: Sequence[VerilogCandidate],
        interface_clusters: Sequence[InterfaceCluster],
        majority_cluster: Optional[InterfaceCluster],
        majority_interface: Optional[InterfaceInfo],
        majority_circuit_type: str,
        majority_circuit_type_source_candidate_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serialize interface analysis so downstream tools can audit majority choice and outliers."""
        return {
            "majority_interface_cluster_id": majority_cluster.cluster_id if majority_cluster else None,
            "majority_circuit_type": majority_circuit_type,
            "majority_circuit_type_source_candidate_id": majority_circuit_type_source_candidate_id,
            "majority_interface": _interface_to_jsonable(majority_interface),
            "interface_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": len(cluster.candidates),
                    "signature_text": cluster.signature_text,
                    "candidate_ids": [candidate.candidate_id for candidate in cluster.candidates],
                    "inferred_circuit_types": cluster.inferred_circuit_types,
                }
                for cluster in interface_clusters
            ],
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "module_name": candidate.interface.module_name,
                    "interface_cluster_id": candidate.interface_cluster_id,
                    "is_majority_interface": candidate.is_majority_interface,
                    "parse_success": candidate.interface.parse_success,
                    "signature_text": normalize_interface_signature(candidate.interface),
                    "inferred_circuit_type": candidate.inferred_circuit_type,
                    "warnings": candidate.interface.warnings,
                    "ports": [_port_to_jsonable(port) for port in candidate.interface.ports],
                }
                for candidate in candidates
            ],
        }

    def _build_case_count_instruction(self, num_cases: Optional[int]) -> str:
        """Translate optional testcase limits into plain-English prompt instructions."""
        if num_cases is None:
            return (
                f"create up to {DEFAULT_MAX_STIMULI} non-redundant, high-value, high-coverage"
            )
        return f"create {num_cases}"

    def save_interface_report(self, report: Dict[str, Any], path: str | Path) -> None:
        """Persist interface analysis for offline inspection and reuse by other tools."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_json_file(self, payload: Any, path: str | Path) -> str:
        """Write one JSON artifact to disk and return its absolute path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(output_path.resolve())

    def _coerce_verification_scenarios(
        self,
        scenarios: Sequence[VerificationScenario | Mapping[str, Any]] | None,
    ) -> List[VerificationScenario]:
        """Normalize external scenario payloads into VerificationScenario objects."""

        if not scenarios:
            return []
        normalized: List[VerificationScenario] = []
        for index, item in enumerate(scenarios, start=1):
            if isinstance(item, VerificationScenario):
                normalized.append(item)
                continue
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"external verification scenario must be an object, got: {type(item).__name__}"
                )
            raw_target_requirement_id = item.get("target_requirement_id")
            target_requirement_id: Optional[int] = None
            if raw_target_requirement_id is not None and str(raw_target_requirement_id).strip():
                try:
                    target_requirement_id = int(raw_target_requirement_id)
                except (TypeError, ValueError):
                    target_requirement_id = None
            normalized.append(
                VerificationScenario(
                    name=str(item.get("name", "")).strip() or f"scenario_{index}",
                    goal=str(item.get("goal", "")).strip(),
                    story=str(item.get("story", "")).strip(),
                    target_requirement_id=target_requirement_id,
                    target_requirement=str(item.get("target_requirement", "")).strip(),
                )
            )
        return normalized

    def _coerce_stimulus_cases(
        self,
        cases: Sequence[StimulusCase | Mapping[str, Any]] | None,
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        *,
        clock_name: Optional[str] = None,
        reset_name: Optional[str] = None,
    ) -> List[StimulusCase]:
        """Normalize external stimulus payloads into StimulusCase objects."""

        if not cases:
            return []
        normalized: List[StimulusCase] = []
        for index, item in enumerate(cases, start=1):
            if isinstance(item, StimulusCase):
                normalized.append(item)
                continue
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"external stimulus case must be an object, got: {type(item).__name__}"
                )
            raw_steps = item.get("steps")
            raw_cycles = item.get("cycles")
            if isinstance(raw_steps, list):
                raw_step_list = raw_steps
            elif isinstance(raw_cycles, list):
                raw_step_list = raw_cycles
            else:
                raise RuntimeError("external stimulus case must contain `steps` or `cycles` as a list")

            steps: List[StimulusStep] = []
            for step in raw_step_list:
                if not isinstance(step, Mapping):
                    raise RuntimeError(
                        f"external stimulus step must be an object, got: {type(step).__name__}"
                    )
                raw_sets = step.get("sets", step.get("inputs", {}))
                if not isinstance(raw_sets, Mapping):
                    raise RuntimeError(
                        f"external stimulus step sets/inputs must be an object, got: {type(raw_sets).__name__}"
                    )
                steps.append(
                    StimulusStep(
                        sets=self._normalize_stimulus_sets(
                            dict(raw_sets),
                            majority_interface,
                            majority_circuit_type,
                            clock_name=clock_name,
                            reset_name=reset_name,
                        ),
                        delay=int(step.get("delay", 0 if majority_circuit_type == "SEQ" else 10) or 0),
                        sample=self._resolve_sample_flag(step),
                        sample_phase=str(step.get("sample_phase", "auto" if majority_circuit_type == "SEQ" else "after_delay")).strip()
                        or ("auto" if majority_circuit_type == "SEQ" else "after_delay"),
                    )
                )
            normalized.append(
                StimulusCase(
                    name=str(item.get("name", "")).strip() or f"case_{index}",
                    description=str(item.get("description", "")).strip(),
                    steps=steps,
                    reason_based_on_spec=str(item.get("reason_based_on_spec", "")).strip(),
                )
            )
        return normalized

    def _stimulus_case_to_jsonable(
        self,
        stimulus_case: StimulusCase,
    ) -> Dict[str, Any]:
        """Convert one stimulus case into the JSON style expected by prompts and reports."""

        return {
            "name": stimulus_case.name,
            "description": stimulus_case.description,
            "reason_based_on_spec": stimulus_case.reason_based_on_spec,
            "steps": [
                {
                    "sets": dict(step.sets),
                    "delay": step.delay,
                    "sample": step.sample,
                    "sample_phase": step.sample_phase,
                }
                for step in stimulus_case.steps
            ],
        }

    def _stimulus_case_to_refinement_payload(
        self,
        stimulus_case: StimulusCase,
        majority_circuit_type: str,
    ) -> Dict[str, Any]:
        """Render one case in the same schema family that stimulus-generation prompts use."""

        if majority_circuit_type == "SEQ":
            return {
                "name": stimulus_case.name,
                "description": stimulus_case.description,
                "reason_based_on_spec": stimulus_case.reason_based_on_spec,
                "cycles": [
                    {
                        "inputs": dict(step.sets),
                        "sample": step.sample,
                        "sample_phase": step.sample_phase,
                    }
                    for step in stimulus_case.steps
                ],
            }
        return {
            "name": stimulus_case.name,
            "description": stimulus_case.description,
            "reason_based_on_spec": stimulus_case.reason_based_on_spec,
            "steps": [
                {
                    "sets": dict(step.sets),
                    "sample": step.sample,
                    "sample_phase": step.sample_phase,
                }
                for step in stimulus_case.steps
            ],
        }

    def _serialize_stimulus_cases(
        self,
        stimulus_cases: Sequence[StimulusCase],
    ) -> List[Dict[str, Any]]:
        """Serialize stimulus cases once so result-building logic stays consistent."""

        return [self._stimulus_case_to_jsonable(case) for case in stimulus_cases]

    def _extract_top_one_cluster_output_trajectory(
        self,
        resolution_output: Mapping[str, Any],
        majority_interface: InterfaceInfo,
    ) -> List[Dict[str, Any]]:
        """Use the top cluster representative trace as feedback for stimulus refinement."""

        mode = str(resolution_output.get("mode", "")).strip()
        top_trace: Sequence[Dict[str, Any]] = []
        if mode == "single_majority_verilog":
            candidate_trace = resolution_output.get("selected_trace", [])
            if isinstance(candidate_trace, list):
                top_trace = candidate_trace
        elif mode == "top_two_verilog_clusters":
            selected_candidates = resolution_output.get("selected_candidates", [])
            if (
                isinstance(selected_candidates, list)
                and selected_candidates
                and isinstance(selected_candidates[0], Mapping)
            ):
                candidate_trace = selected_candidates[0].get("trace", [])
                if isinstance(candidate_trace, list):
                    top_trace = candidate_trace

        input_names = [port.name for port in majority_interface.input_ports]
        return compact_output_trajectory(top_trace, input_names) if top_trace else []

    def _simulate_and_cluster_candidates(
        self,
        *,
        candidates: Sequence[VerilogCandidate],
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        stimulus_cases: Sequence[StimulusCase],
        workdir: str | Path,
        sample_edge: str,
        clock_name: Optional[str],
        reset_info: ResetInfo,
    ) -> Tuple[List[SimulationRecord], List[FunctionalCluster], Dict[str, Any]]:
        """Run one fixed-candidate simulation+clustering pass for the current shared stimulus."""

        simulation_records = self.simulate_candidates(
            candidates,
            majority_interface,
            majority_circuit_type,
            stimulus_cases,
            workdir,
            sample_edge=sample_edge,
            clock_name=clock_name,
            reset_info=reset_info,
        )
        functional_clusters = self.cluster_functional_results(
            simulation_records,
            majority_interface,
            majority_circuit_type,
        )
        resolution = self.build_resolution_output(
            candidates,
            stimulus_cases,
            simulation_records,
            functional_clusters,
        )
        return simulation_records, functional_clusters, resolution

    def _refine_stimulus_case_with_cluster_feedback(
        self,
        *,
        spec: str,
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        scenario: VerificationScenario,
        stimulus_case: StimulusCase,
        top_output_trajectory: Sequence[Dict[str, Any]],
        clock_name: Optional[str],
        reset_name: Optional[str],
    ) -> Tuple[StimulusCase, bool]:
        """Strengthen one stimulus case using the current top-cluster trajectory as feedback."""

        seq_control_blob = ""
        if majority_circuit_type == "SEQ":
            seq_control_blob = json.dumps(
                {
                    "clock_name": clock_name,
                    "reset_name": reset_name,
                    "rule": "Do not include clock_name or reset_name in normal cycle inputs.",
                },
                indent=2,
                ensure_ascii=False,
            )

        schema_blob = (
            json.dumps(
                {
                    "name": "case_name",
                    "description": "what this temporal testcase covers",
                    "reason_based_on_spec": "why this testcase is justified by the specification",
                    "cycles": [{"inputs": {"signal_name": 0}, "sample": True}],
                },
                indent=2,
                ensure_ascii=False,
            )
            if majority_circuit_type == "SEQ"
            else json.dumps(
                {
                    "name": "case_name",
                    "description": "what this testcase covers",
                    "reason_based_on_spec": "why this testcase is justified by the specification",
                    "steps": [{"sets": {"signal_name": 0}, "sample": True}],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        response = self.llm_client.generate(
            system_prompt=STIMULUS_REFINEMENT_SYSTEM_PROMPT,
            user_prompt=STIMULUS_REFINEMENT_USER_PROMPT.format(
                spec=spec,
                interface_signature=normalize_interface_signature(majority_interface),
                circuit_type=majority_circuit_type,
                seq_control_blob=seq_control_blob,
                scenario_blob=json.dumps(asdict(scenario), indent=2, ensure_ascii=False),
                stimulus_case_blob=json.dumps(
                    self._stimulus_case_to_refinement_payload(
                        stimulus_case,
                        majority_circuit_type,
                    ),
                    indent=2,
                    ensure_ascii=False,
                ),
                output_trajectory_blob=json.dumps(
                    list(top_output_trajectory),
                    indent=2,
                    ensure_ascii=False,
                ),
                schema_blob=schema_blob,
            ),
            n=1,
            temperature=self.stimulus_temperature,
        )[0]

        try:
            payload = _safe_json_loads(response)
        except json.JSONDecodeError as exc:
            snippet = response[:1200].replace("\n", "\\n")
            raise RuntimeError(
                f"stimulus refinement JSON parse failed: {exc}. raw response prefix: {snippet}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"stimulus refinement payload must be a JSON object, got: {type(payload).__name__}"
            )

        refined_cases = self._coerce_stimulus_cases(
            [payload],
            majority_interface,
            majority_circuit_type,
            clock_name=clock_name,
            reset_name=reset_name,
        )
        if not refined_cases:
            raise RuntimeError("stimulus refinement returned no usable stimulus case")
        refined_case = refined_cases[0]
        original_payload = self._stimulus_case_to_refinement_payload(
            stimulus_case,
            majority_circuit_type,
        )
        refined_payload = self._stimulus_case_to_refinement_payload(
            refined_case,
            majority_circuit_type,
        )
        return refined_case, refined_payload != original_payload

    def run_fixed_candidate_resolution(
        self,
        *,
        spec: str,
        candidates: Sequence[VerilogCandidate],
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        verification_scenarios: Sequence[VerificationScenario],
        stimulus_cases: Sequence[StimulusCase],
        workdir: str | Path,
        sample_edge: str,
        clock_name: Optional[str],
        reset_info: ResetInfo,
        stimulus_refinement_rounds: int = DEFAULT_STIMULUS_REFINEMENT_ROUNDS,
    ) -> Dict[str, Any]:
        """Keep candidates fixed while iteratively refining the shared stimulus with cluster feedback."""

        current_stimulus_cases = list(stimulus_cases)
        simulation_records, functional_clusters, resolution = self._simulate_and_cluster_candidates(
            candidates=candidates,
            majority_interface=majority_interface,
            majority_circuit_type=majority_circuit_type,
            stimulus_cases=current_stimulus_cases,
            workdir=workdir,
            sample_edge=sample_edge,
            clock_name=clock_name,
            reset_info=reset_info,
        )

        refinement_history: List[Dict[str, Any]] = []
        if (
            stimulus_refinement_rounds > 0
            and verification_scenarios
            and len(verification_scenarios) == len(current_stimulus_cases)
        ):
            for refinement_round_index in range(1, stimulus_refinement_rounds + 1):
                entropy_before = _semantic_entropy_from_functional_clusters(functional_clusters)
                top_output_trajectory = self._extract_top_one_cluster_output_trajectory(
                    resolution,
                    majority_interface,
                )
                if not top_output_trajectory:
                    break

                input_stimulus_cases = self._serialize_stimulus_cases(current_stimulus_cases)
                refined_stimulus_cases: List[StimulusCase] = []
                refinement_applied = False
                for scenario, stimulus_case in zip(
                    verification_scenarios,
                    current_stimulus_cases,
                ):
                    refined_case, changed = self._refine_stimulus_case_with_cluster_feedback(
                        spec=spec,
                        majority_interface=majority_interface,
                        majority_circuit_type=majority_circuit_type,
                        scenario=scenario,
                        stimulus_case=stimulus_case,
                        top_output_trajectory=top_output_trajectory,
                        clock_name=clock_name,
                        reset_name=reset_info.signal_name,
                    )
                    refinement_applied = refinement_applied or changed
                    refined_stimulus_cases.append(refined_case)

                if not refinement_applied:
                    refinement_history.append(
                        {
                            "round_index": refinement_round_index,
                            "entropy_before": entropy_before,
                            "entropy_after": entropy_before,
                            "top_one_cluster_output_trajectory": top_output_trajectory,
                            "input_stimulus_cases": input_stimulus_cases,
                            "refined_stimulus_cases": input_stimulus_cases,
                            "refinement_applied": False,
                            "stop_reason": "refinement_returned_identical_stimulus",
                            "resolution_mode_after_refinement": resolution.get("mode"),
                            "functional_clusters_after_refinement": summarize_functional_clusters(
                                functional_clusters
                            ),
                        }
                    )
                    break

                current_stimulus_cases = refined_stimulus_cases
                simulation_records, functional_clusters, resolution = self._simulate_and_cluster_candidates(
                    candidates=candidates,
                    majority_interface=majority_interface,
                    majority_circuit_type=majority_circuit_type,
                    stimulus_cases=current_stimulus_cases,
                    workdir=workdir,
                    sample_edge=sample_edge,
                    clock_name=clock_name,
                    reset_info=reset_info,
                )
                entropy_after = _semantic_entropy_from_functional_clusters(functional_clusters)
                entropy_increased = (
                    entropy_before is not None
                    and entropy_after is not None
                    and entropy_after
                    > entropy_before + STIMULUS_REFINEMENT_ENTROPY_GAIN_TOLERANCE
                )
                refinement_history.append(
                    {
                        "round_index": refinement_round_index,
                        "entropy_before": entropy_before,
                        "entropy_after": entropy_after,
                        "top_one_cluster_output_trajectory": top_output_trajectory,
                        "input_stimulus_cases": input_stimulus_cases,
                        "refined_stimulus_cases": self._serialize_stimulus_cases(current_stimulus_cases),
                        "refinement_applied": True,
                        "stop_reason": (
                            "entropy_increased_after_refinement"
                            if entropy_increased
                            else "refinement_applied_and_reclustered"
                        ),
                        "resolution_mode_after_refinement": resolution.get("mode"),
                        "functional_clusters_after_refinement": summarize_functional_clusters(
                            functional_clusters
                        ),
                    }
                )
                if entropy_increased:
                    break

        return {
            "stimulus_cases": current_stimulus_cases,
            "simulation_records": simulation_records,
            "functional_clusters": functional_clusters,
            "resolution": resolution,
            "stimulus_refinement": {
                "enabled": True,
                "rounds_configured": stimulus_refinement_rounds,
                "rounds_executed": len(refinement_history),
                "history": refinement_history,
            },
        }

    def _build_final_output_from_full_result(self, full_result: Dict[str, Any]) -> Dict[str, Any]:
        """Derive the compact final-verilog payload from a full resolve_spec result."""
        stimulus_cases = [
            StimulusCase(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                steps=[
                    StimulusStep(
                        sets=dict(step.get("sets", {})),
                        delay=int(step.get("delay", 0)),
                        sample=self._resolve_sample_flag(step),
                        sample_phase=str(step.get("sample_phase", "auto")),
                    )
                    for step in item.get("steps", [])
                ],
                reason_based_on_spec=str(item.get("reason_based_on_spec", "")),
            )
            for item in full_result.get("stimulus_cases", [])
        ]
        majority_interface = full_result.get("interface_report", {}).get("majority_interface", {})
        all_input_names = [
            str(port.get("name"))
            for port in majority_interface.get("ports", [])
            if str(port.get("direction", "")).lower() == "input" and str(port.get("name", "")).strip()
        ]
        display_input_names = [name for name in all_input_names if not _is_clock_like_name(name)]
        return self.build_final_verilog_output(
            stimulus_cases=stimulus_cases,
            resolution_output=full_result.get("resolution", {}),
            input_names=all_input_names,
            display_input_names=display_input_names,
        )

    def persist_resolve_spec_artifacts(
        self,
        full_result: Dict[str, Any],
        artifact_dir: str | Path,
    ) -> Dict[str, str]:
        """Save resolve_spec outputs into one declared artifact directory."""
        output_dir = Path(artifact_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        final_output = self._build_final_output_from_full_result(full_result)

        artifact_files = {
            "interface_report": self._save_json_file(
                full_result.get("interface_report", {}),
                output_dir / "interface_report.json",
            ),
            "verification_scenarios": self._save_json_file(
                full_result.get("verification_scenarios", []),
                output_dir / "verification_scenarios.json",
            ),
            "stimulus_cases": self._save_json_file(
                full_result.get("stimulus_cases", []),
                output_dir / "stimulus_cases.json",
            ),
            "stimulus_refinement": self._save_json_file(
                full_result.get("stimulus_refinement", {}),
                output_dir / "stimulus_refinement.json",
            ),
            "simulation_records": self._save_json_file(
                full_result.get("simulation_records", []),
                output_dir / "simulation_records.json",
            ),
            "functional_clusters": self._save_json_file(
                full_result.get("functional_clusters", []),
                output_dir / "functional_clusters.json",
            ),
            "resolution": self._save_json_file(
                full_result.get("resolution", {}),
                output_dir / "resolution.json",
            ),
            "final_verilog_output": self._save_json_file(
                final_output,
                output_dir / "final_verilog_output.json",
            ),
        }
        artifact_files["full_result"] = self._save_json_file(
            {
                **full_result,
                "artifact_dir": str(output_dir.resolve()),
                "artifact_files": artifact_files,
            },
            output_dir / "full_result.json",
        )
        artifact_files["status"] = self._save_json_file(
            {
                "artifact_dir": str(output_dir.resolve()),
                "artifact_files": artifact_files,
            },
            output_dir / "status.json",
        )
        return artifact_files

    def generate_stimulus_cases(
        self,
        spec: str,
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        scenarios: Sequence[VerificationScenario],
        *,
        num_cases: Optional[int] = None,
        clock_name: Optional[str] = None,
        reset_name: Optional[str] = None,
    ) -> List[StimulusCase]:
        """Turn high-level scenarios into concrete sequential or combinational stimuli."""
        stimulus_prompt = (
            SEQ_STIMULUS_USER_PROMPT
            if majority_circuit_type == "SEQ"
            else CMB_STIMULUS_USER_PROMPT
        )

        scenario_blob = json.dumps(
            [asdict(scenario) for scenario in scenarios],
            indent=2,
            ensure_ascii=False,
        )

        seq_control_blob = ""
        if majority_circuit_type == "SEQ":
            seq_control_blob = json.dumps(
                {
                    "clock_name": clock_name,
                    "reset_name": reset_name,
                    "rule": "Do not include clock_name or reset_name in normal cycle inputs.",
                },
                indent=2,
                ensure_ascii=False,
            )

        response = self.llm_client.generate(
            system_prompt=STIMULUS_SYSTEM_PROMPT,
            user_prompt=stimulus_prompt.format(
                spec=spec,
                case_count_instruction=self._build_case_count_instruction(num_cases),
                interface_signature=normalize_interface_signature(majority_interface),
                circuit_type=majority_circuit_type,
                scenario_blob=scenario_blob,
                seq_control_blob=seq_control_blob,
            ),
            n=1,
            temperature=self.stimulus_temperature,
        )[0]

        try:
            payload = _safe_json_loads(response)
        except json.JSONDecodeError as exc:
            snippet = response[:1200].replace("\n", "\\n")
            raise RuntimeError(
                f"stimulus JSON parse failed: {exc}. raw response prefix: {snippet}"
            ) from exc

        if not isinstance(payload, list):
            raise RuntimeError(
                f"stimulus payload must be a JSON list, got: {type(payload).__name__}"
            )

        cases: List[StimulusCase] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"each stimulus case must be a JSON object, got: {type(item).__name__}"
                )

            if majority_circuit_type == "SEQ":
                raw_cycles = item.get("cycles", item.get("steps", []))
                if not isinstance(raw_cycles, list):
                    raise RuntimeError(
                        f"`cycles` must be a JSON list, got: {type(raw_cycles).__name__}"
                    )

                steps: List[StimulusStep] = []
                for cycle in raw_cycles:
                    if not isinstance(cycle, Mapping):
                        raise RuntimeError(
                            f"each sequential cycle must be a JSON object, got: {type(cycle).__name__}"
                        )

                    raw_sets = cycle.get("inputs", cycle.get("sets", {}))
                    if not isinstance(raw_sets, Mapping):
                        raise RuntimeError(
                            f"`inputs` must be a JSON object, got: {type(raw_sets).__name__}"
                        )

                    steps.append(
                        StimulusStep(
                            sets=self._normalize_stimulus_sets(
                                dict(raw_sets),
                                majority_interface,
                                majority_circuit_type,
                                clock_name=clock_name,
                                reset_name=reset_name,
                            ),
                            delay=0,
                            sample=self._resolve_sample_flag(cycle),
                            sample_phase=str(cycle.get("sample_phase", "auto")).strip() or "auto",
                        )
                    )
            else:
                raw_steps = item.get("steps", [])
                if not isinstance(raw_steps, list):
                    raise RuntimeError(
                        f"`steps` must be a JSON list, got: {type(raw_steps).__name__}"
                    )

                steps = []
                for step in raw_steps:
                    if not isinstance(step, Mapping):
                        raise RuntimeError(
                            f"each combinational step must be a JSON object, got: {type(step).__name__}"
                        )

                    raw_sets = step.get("sets", {})
                    if not isinstance(raw_sets, Mapping):
                        raise RuntimeError(
                            f"`sets` must be a JSON object, got: {type(raw_sets).__name__}"
                        )

                    steps.append(
                        StimulusStep(
                            sets=self._normalize_stimulus_sets(
                                dict(raw_sets),
                                majority_interface,
                                majority_circuit_type,
                            ),
                            delay=10,
                            sample=self._resolve_sample_flag(step),
                            sample_phase="after_delay",
                        )
                    )

            cases.append(
                StimulusCase(
                    name=str(item.get("name", "")).strip() or f"case_{len(cases) + 1}",
                    description=str(item.get("description", "")).strip(),
                    steps=steps,
                    reason_based_on_spec=str(item.get("reason_based_on_spec", "")).strip(),
                )
            )

        return cases
    def _resolve_sample_flag(self, step_like: Mapping[str, Any]) -> bool:
        """Default missing sampling flags to True so omitted metadata does not erase all observability."""
        if "sample" not in step_like:
            return True
        value = step_like.get("sample")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        if isinstance(value, int):
            return value != 0
        return bool(value)

    def build_driver_verilog(
        self,
        interface: InterfaceInfo,
        stimulus_cases: Sequence[StimulusCase],
        *,
        candidate_module_name: str,
        circuit_type: str = "UNKNOWN",
        clock_name: Optional[str] = None,
        reset_info: Optional[ResetInfo] = None,
        sample_edge: str = "posedge",
        timescale: str = "1ns/1ps",
        case_index_base: int = 0,
    ) -> str:
        """Compile structured stimulus into a runnable Verilog driver tailored to one interface."""
        input_ports = interface.input_ports
        output_ports = interface.output_ports
        effective_reset_info = reset_info or ResetInfo(None, "none")

        lines: List[str] = [f"`timescale {timescale}", "module __auto_tb_driver;"]
        for param_name, default_expr in interface.parameter_defaults.items():
            lines.append(f"localparam {param_name} = {default_expr};")
        for port in input_ports:
            width = f" {port.normalized_width()}" if port.normalized_width() else ""
            lines.append(f"reg{width} {port.name};")
        for port in output_ports:
            width = f" {port.normalized_width()}" if port.normalized_width() else ""
            lines.append(f"wire{width} {port.name};")

        lines.append("integer file;")
        lines.append("integer case_idx;")
        lines.append("integer step_idx;")
        lines.append("integer cycle_idx;")

        clk_port = None

        if circuit_type == "SEQ":
            clock_names = set(_candidate_clock_names(interface))
            if clock_name:
                clock_names.add(clock_name)

            if clock_name:
                clk_port = next((port for port in input_ports if port.name == clock_name), None)

            if clk_port is None:
                clk_port = next((port for port in input_ports if port.name in clock_names), None)

            if clk_port:
                lines.append("initial begin")
                lines.append(f"    {clk_port.name} = 0;")
                lines.append("    forever #5 " + clk_port.name + " = ~" + clk_port.name + ";")
                lines.append("end")

        inst_ports = ", ".join(f".{port.name}({port.name})" for port in interface.ports)
        param_override = ""
        if interface.parameter_defaults:
            param_pairs = ", ".join(f".{name}({name})" for name in interface.parameter_defaults)
            param_override = f" #({param_pairs})"
        lines.append(f"{candidate_module_name}{param_override} dut ({inst_ports});")
        lines.append("")
        lines.append("initial begin")
        for port in input_ports:
            if clk_port and port.name == clk_port.name:
                continue
            if port.name == effective_reset_info.signal_name:
                lines.append(f"    {port.name} = {self._reset_deassert_literal(effective_reset_info)};")
            else:
                lines.append(f"    {port.name} = 0;")
        lines.append('    file = $fopen("TBout.txt", "w");')
        lines.append("    case_idx = 0;")
        lines.append("    step_idx = 0;")
        lines.append("    cycle_idx = 0;")
        if circuit_type == "SEQ" and clk_port:
            lines.append("    #1;")

        if circuit_type == "SEQ":
            lines.extend(
                self._build_seq_stimulus_body(
                    interface,
                    stimulus_cases,
                    clk_port,
                    effective_reset_info,
                    sample_edge=sample_edge,
                    case_index_base=case_index_base,
                )
            )
        else:
            lines.extend(self._build_cmb_stimulus_body(interface, stimulus_cases))
        lines.append("    $fclose(file);")
        lines.append("    $finish;")
        lines.append("end")
        lines.append("endmodule")
        return "\n".join(lines) + "\n"


    def simulate_candidates(
        self,
        candidates: Sequence[VerilogCandidate],
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
        stimulus_cases: Sequence[StimulusCase],
        workdir: str | Path,
        sample_edge: str = "posedge",
        clock_name: Optional[str] = None,
        reset_info: Optional[ResetInfo] = None,
    ) -> List[SimulationRecord]:
        """Run all interface-compatible candidates under one shared stimulus set and collect traces."""
        output_dir = Path(workdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not majority_interface.parse_success:
            return [
                SimulationRecord(
                    candidate_id=candidate.candidate_id,
                    interface_cluster_id=candidate.interface_cluster_id,
                    simulation_dir=None,
                    compile_ok=False,
                    run_ok=False,
                    tbout_path=None,
                    trace=[],
                    functional_signature=None,
                    error="majority interface parse failed",
                )
                for candidate in candidates
            ]

        if any(port.direction == "inout" for port in majority_interface.ports):
            return [
                SimulationRecord(
                    candidate_id=candidate.candidate_id,
                    interface_cluster_id=candidate.interface_cluster_id,
                    simulation_dir=None,
                    compile_ok=False,
                    run_ok=False,
                    tbout_path=None,
                    trace=[],
                    functional_signature=None,
                    error="inout ports are not supported by the generated driver",
                )
                for candidate in candidates
            ]

        results: List[SimulationRecord] = []
        majority_signature = normalize_interface_signature(majority_interface)
        output_port_names = [port.name for port in majority_interface.output_ports]
        shared_reset_info = reset_info or ResetInfo(None, "none")

        for candidate in candidates:
            candidate_dir = output_dir / f"candidate_{candidate.candidate_id:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            if normalize_interface_signature(candidate.interface) != majority_signature:
                results.append(
                    SimulationRecord(
                        candidate_id=candidate.candidate_id,
                        interface_cluster_id=candidate.interface_cluster_id,
                        simulation_dir=str(candidate_dir),
                        compile_ok=False,
                        run_ok=False,
                        tbout_path=None,
                        trace=[],
                        functional_signature=None,
                        error="interface differs from majority interface",
                    )
                )
                continue

            if majority_circuit_type == "SEQ":
                results.append(
                    self._simulate_seq_candidate_per_case(
                        candidate=candidate,
                        majority_interface=majority_interface,
                        stimulus_cases=stimulus_cases,
                        candidate_dir=candidate_dir,
                        sample_edge=sample_edge,
                        clock_name=clock_name,
                        reset_info=shared_reset_info,
                    )
                )
                continue

            dut_path = candidate_dir / "DUT.v"
            driver_path = candidate_dir / "driver.v"
            dut_path.write_text(candidate.verilog_code, encoding="utf-8")
            candidate_module_name = str(
                candidate.metadata.get("expected_top_module_name", candidate.interface.module_name)
            )
            driver_path.write_text(
                self.build_driver_verilog(
                    majority_interface,
                    stimulus_cases,
                    candidate_module_name=candidate_module_name,
                    circuit_type=majority_circuit_type,
                    clock_name=clock_name,
                    reset_info=shared_reset_info,
                    sample_edge=sample_edge,
                ),
                encoding="utf-8",
            )

            compile_ok, run_ok, error = self._run_iverilog(candidate_dir)
            tbout_path = candidate_dir / "TBout.txt"
            trace = parse_tbout(tbout_path) if tbout_path.exists() else []
            if compile_ok and run_ok and not trace:
                error = "no output trace generated; candidate excluded from functional clustering"
            signature = (
                output_signature_from_trace(trace, output_port_names)
                if trace
                else None
            )

            results.append(
                SimulationRecord(
                    candidate_id=candidate.candidate_id,
                    interface_cluster_id=candidate.interface_cluster_id,
                    simulation_dir=str(candidate_dir),
                    compile_ok=compile_ok,
                    run_ok=run_ok,
                    tbout_path=str(tbout_path) if tbout_path.exists() else None,
                    trace=trace,
                    functional_signature=signature,
                    error=error,
                )
            )
        return results


        
    def _simulate_seq_candidate_per_case(
        self,
        *,
        candidate: VerilogCandidate,
        majority_interface: InterfaceInfo,
        stimulus_cases: Sequence[StimulusCase],
        candidate_dir: Path,
        sample_edge: str,
        clock_name: Optional[str],
        reset_info: ResetInfo,
    ) -> SimulationRecord:
        """Run one sequential simulation per testcase so each case starts from a fresh simulator instance."""
        all_trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        all_compile_ok = True
        all_run_ok = True
        output_port_names = [port.name for port in majority_interface.output_ports]

        for case_index, stimulus_case in enumerate(stimulus_cases):
            case_dir = candidate_dir / f"case_{case_index:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)

            dut_path = case_dir / "DUT.v"
            driver_path = case_dir / "driver.v"
            dut_path.write_text(candidate.verilog_code, encoding="utf-8")
            candidate_module_name = str(
                candidate.metadata.get("expected_top_module_name", candidate.interface.module_name)
            )
            driver_code = self.build_driver_verilog(
                majority_interface,
                [stimulus_case],
                candidate_module_name=candidate_module_name,
                circuit_type="SEQ",
                clock_name=clock_name,
                reset_info=reset_info,
                sample_edge=sample_edge,
                case_index_base=case_index,
            )
            driver_path.write_text(driver_code, encoding="utf-8")

            compile_ok, run_ok, error = self._run_iverilog(case_dir)
            tbout_path = case_dir / "TBout.txt"
            case_trace = parse_tbout(tbout_path) if tbout_path.exists() else []
            for row in case_trace:
                row["case"] = case_index
            all_trace.extend(case_trace)

            if not compile_ok:
                all_compile_ok = False
            if not run_ok:
                all_run_ok = False
            if error:
                errors.append(f"case_{case_index:03d}: {error}")

        signature = output_signature_from_trace(all_trace, output_port_names) if all_trace else None
        if all_compile_ok and all_run_ok and not all_trace:
            errors.append("no output trace generated")

        return SimulationRecord(
            candidate_id=candidate.candidate_id,
            interface_cluster_id=candidate.interface_cluster_id,
            simulation_dir=str(candidate_dir),
            compile_ok=all_compile_ok,
            run_ok=all_run_ok,
            tbout_path=None,
            trace=all_trace,
            functional_signature=signature,
            error="; ".join(errors),
        )

    def cluster_functional_results(
        self,
        simulation_records: Sequence[SimulationRecord],
        majority_interface: InterfaceInfo,
        majority_circuit_type: str,
    ) -> List[FunctionalCluster]:
        """Cluster candidates by identical observed input/output trace signatures under shared stimuli."""
        valid_records = [
            record
            for record in simulation_records
            if record.compile_ok and record.run_ok and record.trace
        ]
        output_port_names = [port.name for port in majority_interface.output_ports]
        cluster_states: List[Dict[str, Any]] = []
        sorted_records = sorted(valid_records, key=lambda record: record.candidate_id)
        for record in sorted_records:
            filtered_trace = filter_trace_rows_without_unknown_outputs(record.trace, output_port_names)
            rows = trace_io_rows(filtered_trace, majority_interface, majority_circuit_type)
            placed = False
            for cluster_state in cluster_states:
                member_rows = cluster_state["member_rows"]
                if all(
                    _rows_compatible_ignoring_missing_keys(rows, existing_rows)
                    for existing_rows in member_rows
                ):
                    cluster_state["candidate_ids"].append(record.candidate_id)
                    member_rows.append(rows)
                    placed = True
                    break
            if not placed:
                cluster_states.append(
                    {
                        "candidate_ids": [record.candidate_id],
                        "member_rows": [rows],
                    }
                )

        cluster_states.sort(key=lambda item: (-len(item["candidate_ids"]), min(item["candidate_ids"])))

        clusters: List[FunctionalCluster] = []
        for index, cluster_state in enumerate(cluster_states, start=1):
            candidate_ids = sorted(cluster_state["candidate_ids"])
            representative_candidate_id = min(candidate_ids) if candidate_ids else None
            clusters.append(
                FunctionalCluster(
                    cluster_id=f"functional_cluster_{index}",
                    score=len(candidate_ids),
                    candidate_ids=candidate_ids,
                    representative_candidate_id=representative_candidate_id,
                )
            )
        return clusters

    def build_resolution_output(
        self,
        candidates: Sequence[VerilogCandidate],
        stimulus_cases: Sequence[StimulusCase],
        simulation_records: Sequence[SimulationRecord],
        functional_clusters: Sequence[FunctionalCluster],
    ) -> Dict[str, Any]:
        """Convert raw clustering into a decision-oriented result that picks one or two representatives."""
        candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
        valid_records = {
            record.candidate_id: record
            for record in simulation_records
            if record.compile_ok and record.run_ok and record.trace
        }
        excluded_candidates = [
            {
                "candidate_id": record.candidate_id,
                "interface_cluster_id": record.interface_cluster_id,
                "error": record.error,
                "compile_ok": record.compile_ok,
                "run_ok": record.run_ok,
            }
            for record in simulation_records
            if record.candidate_id not in valid_records
        ]
        total_valid = len(valid_records)
        if not functional_clusters or total_valid == 0:
            return {
                "mode": "no_valid_verilog",
                "excluded_candidates": excluded_candidates,
                "functional_clusters": summarize_functional_clusters(functional_clusters),
            }

        top_cluster = functional_clusters[0]
        semantic_entropy = _semantic_entropy_from_functional_clusters(functional_clusters)
        if semantic_entropy is not None and abs(semantic_entropy) <= 1e-12:
            representative_id = top_cluster.representative_candidate_id or top_cluster.candidate_ids[0]
            representative_candidate = candidate_map[representative_id]
            representative_record = valid_records[representative_id]
            return {
                "mode": "single_majority_verilog",
                "semantic_entropy": semantic_entropy,
                "selected_candidate_id": representative_id,
                "selected_verilog": representative_candidate.verilog_code,
                "selected_trace": representative_record.trace,
                "functional_clusters": summarize_functional_clusters(functional_clusters),
                "excluded_candidates": excluded_candidates,
            }

        selected_clusters = list(functional_clusters[:2])
        selected_candidates: List[Dict[str, Any]] = []
        for cluster in selected_clusters:
            representative_id = cluster.representative_candidate_id or cluster.candidate_ids[0]
            representative_candidate = candidate_map[representative_id]
            representative_record = valid_records[representative_id]
            selected_candidates.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_score": cluster.score,
                    "cluster_size": len(cluster.candidate_ids),
                    "cluster_ratio": len(cluster.candidate_ids) / total_valid,
                    "candidate_id": representative_id,
                    "verilog_code": representative_candidate.verilog_code,
                    "trace": representative_record.trace,
                }
            )

        differing_cases = []
        if len(selected_candidates) == 2:
            representative_candidate = candidate_map[selected_candidates[0]["candidate_id"]]
            differing_cases = self._build_differing_case_evidence(
                valid_records[selected_candidates[0]["candidate_id"]],
                valid_records[selected_candidates[1]["candidate_id"]],
                stimulus_cases,
                [port.name for port in representative_candidate.interface.output_ports],
            )

        return {
            "mode": "top_two_verilog_clusters",
            "semantic_entropy": semantic_entropy,
            "selected_candidates": selected_candidates,
            "differing_cases": differing_cases,
            "functional_clusters": summarize_functional_clusters(functional_clusters),
            "excluded_candidates": excluded_candidates,
        }

    def build_final_verilog_output(
        self,
        stimulus_cases: Sequence[StimulusCase],
        resolution_output: Dict[str, Any],
        input_names: Optional[Sequence[str]] = None,
        display_input_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compress the full decision result into the minimal file intended for downstream reuse."""
        final_output: Dict[str, Any] = {
            "single_verilog": None,
            "dual_verilog_candidates": None,
        }

        mode = resolution_output.get("mode")
        if mode == "single_majority_verilog":
            final_output["single_verilog"] = {
                "verilog_code": resolution_output.get("selected_verilog"),
            }
            return final_output

        if mode == "top_two_verilog_clusters":
            selected_candidates = resolution_output.get("selected_candidates", [])
            if len(selected_candidates) >= 2:
                candidate_a = selected_candidates[0]
                candidate_b = selected_candidates[1]
                differing_cases = resolution_output.get("differing_cases", [])
                different_output_trajectories: List[Dict[str, Any]] = []
                effective_input_names = list(input_names or [])
                effective_display_input_names = list(
                    display_input_names
                    if display_input_names is not None
                    else [name for name in effective_input_names if not _is_clock_like_name(name)]
                )
                for item in differing_cases:
                    case_index = int(item.get("case_index", -1))
                    candidate_a_trace = item.get("candidate_a_trace", [])
                    candidate_b_trace = item.get("candidate_b_trace", [])
                    input_trace_source = candidate_a_trace or candidate_b_trace
                    input_trajectory = compact_input_trajectory(
                        input_trace_source,
                        effective_display_input_names,
                    )
                    different_output_trajectories.append(
                        {
                            "case_index": case_index,
                            "case_name": item.get("case_name"),
                            "case_description": item.get("case_description"),
                            "case_reason_based_on_spec": item.get("case_reason_based_on_spec", ""),
                            "input_trajectory": input_trajectory,
                            "candidate_a_output_trajectory": compact_output_trajectory(
                                candidate_a_trace,
                                effective_input_names,
                            ),
                            "candidate_b_output_trajectory": compact_output_trajectory(
                                candidate_b_trace,
                                effective_input_names,
                            ),
                        }
                    )

                final_output["dual_verilog_candidates"] = {
                    "candidate_a": {
                        "verilog_code": candidate_a.get("verilog_code"),
                    },
                    "candidate_b": {
                        "verilog_code": candidate_b.get("verilog_code"),
                    },
                    "different_output_trajectories": different_output_trajectories,
                }
            return final_output

        return final_output

    def _build_differing_case_evidence(
        self,
        left_record: SimulationRecord,
        right_record: SimulationRecord,
        stimulus_cases: Sequence[StimulusCase],
        output_port_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Isolate only the testcase slices where two representative candidates disagree."""
        left_by_case = group_trace_by_case(left_record.trace)
        right_by_case = group_trace_by_case(right_record.trace)
        case_name_map = {index: case.name for index, case in enumerate(stimulus_cases)}
        case_description_map = {index: case.description for index, case in enumerate(stimulus_cases)}
        case_reason_map = {index: case.reason_based_on_spec for index, case in enumerate(stimulus_cases)}
        differing_cases: List[Dict[str, Any]] = []
        for case_idx in sorted(set(left_by_case) | set(right_by_case)):
            left_trace = left_by_case.get(case_idx, [])
            right_trace = right_by_case.get(case_idx, [])
            masked_left_trace, masked_right_trace = mask_pairwise_unknown_output_rows(
                left_trace,
                right_trace,
                output_port_names,
            )
            if masked_left_trace == masked_right_trace:
                continue
            differing_cases.append(
                {
                    "case_index": case_idx,
                    "case_name": case_name_map.get(case_idx, f"case_{case_idx}"),
                    "case_description": case_description_map.get(case_idx, ""),
                    "case_reason_based_on_spec": case_reason_map.get(case_idx, ""),
                    "candidate_a_id": left_record.candidate_id,
                    "candidate_a_trace": masked_left_trace,
                    "candidate_b_id": right_record.candidate_id,
                    "candidate_b_trace": masked_right_trace,
                    "trajectory_difference_summary": summarize_pairwise_trace_difference(
                        masked_left_trace,
                        masked_right_trace,
                        output_port_names,
                    ),
                }
            )
        return differing_cases

    def analyze_interfaces(
        self,
        spec: str,
        *,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        report_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run just the candidate/interface-analysis half of the pipeline when simulation is not needed."""
        candidates = self.generate_candidates(
            spec,
            num_candidates=num_candidates,
        )
        interface_clusters = self.cluster_interfaces(candidates)
        (
            majority_cluster,
            majority_interface,
            majority_circuit_type,
            majority_circuit_type_source_candidate_id,
        ) = self.select_majority_interface(interface_clusters)
        report = self.build_interface_report(
            candidates,
            interface_clusters,
            majority_cluster,
            majority_interface,
            majority_circuit_type,
            majority_circuit_type_source_candidate_id,
        )
        if report_path is not None:
            self.save_interface_report(report, report_path)
        return report

    def resolve_spec(
        self,
        spec: str,
        *,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        num_stimuli: Optional[int] = DEFAULT_MAX_STIMULI,
        external_verification_scenarios: Optional[Sequence[VerificationScenario | Mapping[str, Any]]] = None,
        external_stimulus_cases: Optional[Sequence[StimulusCase | Mapping[str, Any]]] = None,
        workdir: str | Path,
        artifact_dir: Optional[str | Path] = None,
        enable_stimulus_refinement: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full end-to-end pipeline and return every intermediate artifact plus resolution."""
        self._reset_call_cache_flags()
        candidates = self.generate_candidates(
            spec,
            num_candidates=num_candidates,
        )
        interface_clusters = self.cluster_interfaces(candidates)
        (
            majority_cluster,
            majority_interface,
            majority_circuit_type,
            majority_circuit_type_source_candidate_id,
        ) = self.select_majority_interface(interface_clusters)
        report = self.build_interface_report(
            candidates,
            interface_clusters,
            majority_cluster,
            majority_interface,
            majority_circuit_type,
            majority_circuit_type_source_candidate_id,
        )
        if majority_cluster is None or majority_interface is None:
            result = {
                "mode": "no_majority_interface",
                "interface_report": report,
                "functional_clusters": [],
                "excluded_candidates": [],
            }
            if artifact_dir is not None:
                resolved_artifact_dir = str(Path(artifact_dir).resolve())
                artifact_files = self.persist_resolve_spec_artifacts(result, resolved_artifact_dir)
                result["artifact_dir"] = resolved_artifact_dir
                result["artifact_files"] = artifact_files
            return result

        clock_name_hint: Optional[str] = None
        sample_edge_hint = "posedge"
        majority_reset_info = ResetInfo(None, "none")
        seq_control_contract: Optional[SeqControlContract] = None
        seq_control_fallback_reason: Optional[str] = None

        if majority_circuit_type == "SEQ":
            try:
                seq_control_contract = self.extract_seq_control_contract_from_spec(
                    spec,
                    majority_interface,
                )
                clock_name_hint, sample_edge_hint, majority_reset_info = (
                    self.seq_control_contract_to_runtime(seq_control_contract)
                )
            except MultiClockSeqControlContractError:
                # Multi-clock / multi-reset tasks do not fit the current single-domain
                # seq-control schema. Fall back to domain-agnostic stimulus handling
                # instead of failing the entire task.
                seq_control_fallback_reason = "multiclock_seq_control_contract"

        if external_verification_scenarios is None:
            raise RuntimeError(
                "resolve_spec now requires external_verification_scenarios in the active requirements_constraint_selfplanning flow."
            )
        verification_scenarios = self._coerce_verification_scenarios(external_verification_scenarios)

        if external_stimulus_cases is not None:
            stimulus_cases = self._coerce_stimulus_cases(
                external_stimulus_cases,
                majority_interface,
                majority_circuit_type,
                clock_name=clock_name_hint,
                reset_name=majority_reset_info.signal_name,
            )
        else:
            stimulus_cases = self.generate_stimulus_cases(
                spec,
                majority_interface,
                majority_circuit_type,
                verification_scenarios,
                num_cases=num_stimuli,
                clock_name=clock_name_hint,
                reset_name=majority_reset_info.signal_name,
            )

        resolution_bundle = self.run_fixed_candidate_resolution(
            spec=spec,
            candidates=candidates,
            majority_interface=majority_interface,
            majority_circuit_type=majority_circuit_type,
            verification_scenarios=verification_scenarios,
            stimulus_cases=stimulus_cases,
            workdir=workdir,
            sample_edge=sample_edge_hint,
            clock_name=clock_name_hint,
            reset_info=majority_reset_info,
            stimulus_refinement_rounds=(
                DEFAULT_STIMULUS_REFINEMENT_ROUNDS if enable_stimulus_refinement else 0
            ),
        )
        stimulus_cases = list(resolution_bundle.get("stimulus_cases", []))
        simulation_records = list(resolution_bundle.get("simulation_records", []))
        functional_clusters = list(resolution_bundle.get("functional_clusters", []))
        resolution = dict(resolution_bundle.get("resolution", {}))
        result = {
            "interface_report": report,
            "verification_scenarios": [asdict(scenario) for scenario in verification_scenarios],
            "verification_scenarios_source": "external",
            "cache_usage": {
                "top_module_name_cache_hit": self._last_top_module_name_cache_hit,
                "seq_control_contract_cache_hit": self._last_seq_control_contract_cache_hit,
            },
            "seq_control_fallback_reason": seq_control_fallback_reason,
            "clock_name": clock_name_hint,
            "sample_edge": sample_edge_hint,
            "reset_info": asdict(majority_reset_info),
            "seq_control_contract": asdict(seq_control_contract) if seq_control_contract else None,
            "stimulus_cases": self._serialize_stimulus_cases(stimulus_cases),
            "stimulus_cases_source": (
                "external" if external_stimulus_cases is not None else "generated"
            ),
            "stimulus_refinement": resolution_bundle.get("stimulus_refinement", {}),
            "simulation_records": [
                {
                    "candidate_id": record.candidate_id,
                    "interface_cluster_id": record.interface_cluster_id,
                    "simulation_dir": record.simulation_dir,
                    "compile_ok": record.compile_ok,
                    "run_ok": record.run_ok,
                    "tbout_path": record.tbout_path,
                    "error": record.error,
                    "trace": record.trace,
                }
                for record in simulation_records
            ],
            "functional_clusters": summarize_functional_clusters(functional_clusters),
            "resolution": resolution,
        }
        if artifact_dir is not None:
            resolved_artifact_dir = str(Path(artifact_dir).resolve())
            artifact_files = self.persist_resolve_spec_artifacts(result, resolved_artifact_dir)
            result["artifact_dir"] = resolved_artifact_dir
            result["artifact_files"] = artifact_files
        return result

    def resolve_spec_to_verilog_choice(
        self,
        spec: str,
        *,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        num_stimuli: Optional[int] = DEFAULT_MAX_STIMULI,
        workdir: str | Path,
        artifact_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Expose only the final one-verilog or two-verilog decision view for downstream callers."""
        full_result = self.resolve_spec(
            spec,
            num_candidates=num_candidates,
            num_stimuli=num_stimuli,
            workdir=workdir,
            artifact_dir=artifact_dir,
        )
        final_output = self._build_final_output_from_full_result(full_result)
        if "artifact_dir" in full_result:
            final_output["artifact_dir"] = full_result["artifact_dir"]
        if "artifact_files" in full_result:
            final_output["artifact_files"] = full_result["artifact_files"]
        return final_output

    def _build_sample_format(self, interface: InterfaceInfo) -> str:
        """Build the `$fdisplay` format string once so all TBout rows share one stable schema."""
        fields = ["case=%0d", "step=%0d", "cycle=%0d"]
        for port in interface.ports:
            fields.append(f"{port.name}=%b")
        return ", ".join(fields)

    def _build_sample_args(self, interface: InterfaceInfo) -> str:
        """Build the `$fdisplay` argument list that matches the generated sample format exactly."""
        args = ["case_idx", "step_idx", "cycle_idx"]
        args.extend(port.name for port in interface.ports)
        return ", ".join(args)


    def _normalize_stimulus_sets(
        self,
        sets: Dict[str, Any],
        interface: InterfaceInfo,
        circuit_type: str,
        *,
        clock_name: Optional[str] = None,
        reset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Filter LLM stimulus fields down to legal driven inputs under the chosen interface."""
        valid_inputs = {port.name for port in interface.input_ports}

        blocked_names: set[str] = set()
        if circuit_type == "SEQ":
            if clock_name:
                blocked_names.add(clock_name)
            else:
                blocked_names.update(_candidate_clock_names(interface))
            # Reset is usually driver-managed, but some sequential scenarios need an
            # explicit reset pulse to initialize or verify reset-sensitive behavior.
            # Allow the concrete reset signal to pass through when the stimulus
            # generator provides it explicitly. If we do not know the reset name,
            # keep blocking reset-like guesses to avoid accidental misuse.
            if not reset_name:
                blocked_names.update(_candidate_reset_names(interface))

        normalized: Dict[str, Any] = {}
        for key, value in sets.items():
            if key not in valid_inputs:
                continue
            if key in blocked_names:
                continue
            normalized[key] = value

        return normalized   


    def _build_cmb_stimulus_body(
        self,
        interface: InterfaceInfo,
        stimulus_cases: Sequence[StimulusCase],
    ) -> List[str]:
        """Render combinational stimulus as clean input vectors, a fixed settle delay, and samples."""
        lines: List[str] = []
        sample_fmt = self._build_sample_format(interface)
        sample_args = self._build_sample_args(interface)
        for case_index, case in enumerate(stimulus_cases):
            lines.append(f"    // {case.name}: {case.description}")
            lines.append(f"    case_idx = {case_index};")
            for step_index, step in enumerate(case.steps):
                lines.append(f"    step_idx = {step_index};")
                lines.append(f"    cycle_idx = {step_index};")
                for port in interface.input_ports:
                    lines.append(f"    {port.name} = 0;")
                for signal_name, value in step.sets.items():
                    lines.append(f"    {signal_name} = {self._verilog_literal(value)};")
                lines.append("    #10;")
                if step.sample:
                    lines.append(f'    $fdisplay(file, "{sample_fmt}", {sample_args});')
        return lines

    def _build_seq_stimulus_body(
        self,
        interface: InterfaceInfo,
        stimulus_cases: Sequence[StimulusCase],
        clk_port: Optional[Port],
        reset_info: ResetInfo,
        sample_edge: str = "posedge",
        case_index_base: int = 0,
    ) -> List[str]:
        """Render sequential stimulus as edge-aligned cycles, then append one reset-check tail per case."""
        lines: List[str] = []
        sample_fmt = self._build_sample_format(interface)
        sample_args = self._build_sample_args(interface)
        if clk_port is None:
            lines.extend(self._build_cmb_stimulus_body(interface, stimulus_cases))
            return lines

        active_edge = "posedge" if sample_edge == "posedge" else "negedge"
        inactive_edge = "negedge" if sample_edge == "posedge" else "posedge"
        for local_case_index, case in enumerate(stimulus_cases):
            global_case_index = case_index_base + local_case_index
            lines.append(f"    // {case.name}: {case.description}")
            lines.append(f"    case_idx = {global_case_index};")
            lines.extend(self._build_case_input_clear(interface, clk_port, reset_info))
            if reset_info.signal_name:
                lines.append(f"    {reset_info.signal_name} = {self._reset_deassert_literal(reset_info)};")
            for step_index, step in enumerate(case.steps):
                if step_index > 0:
                    lines.append(f"    @({inactive_edge} {clk_port.name});")
                lines.append(f"    step_idx = {step_index};")
                lines.append(f"    cycle_idx = {step_index};")
                for signal_name, value in step.sets.items():
                    lines.append(f"    {signal_name} = {self._verilog_literal(value)};")
                lines.append(f"    @({active_edge} {clk_port.name});")
                lines.append("    #1;")
                if step.sample:
                    lines.append(f'    $fdisplay(file, "{sample_fmt}", {sample_args});')
            if case.steps:
                lines.append(f"    @({inactive_edge} {clk_port.name});")
            lines.extend(
                self._build_seq_reset_check_tail(
                    interface=interface,
                    clk_port=clk_port,
                    reset_info=reset_info,
                    active_edge=active_edge,
                    inactive_edge=inactive_edge,
                    step_index=len(case.steps),
                    sample_fmt=sample_fmt,
                    sample_args=sample_args,
                )
            )
        return lines

    def _build_seq_reset_check_tail(
        self,
        *,
        interface: InterfaceInfo,
        clk_port: Port,
        reset_info: ResetInfo,
        active_edge: str,
        inactive_edge: str,
        step_index: int,
        sample_fmt: str,
        sample_args: str,
    ) -> List[str]:
        """Append one reset-check sample at case end so reset behavior is still observed without polluting normal cycles."""
        lines: List[str] = []
        if not reset_info.signal_name:
            return lines
        lines.append("    // auto reset check tail")
        lines.append(f"    step_idx = {step_index};")
        lines.append(f"    cycle_idx = {step_index};")
        for port in interface.input_ports:
            if port.name == clk_port.name:
                continue
            if port.name == reset_info.signal_name:
                continue
            lines.append(f"    {port.name} = 0;")
        lines.append(f"    {reset_info.signal_name} = {self._reset_assert_literal(reset_info)};")
        if reset_info.style == "async":
            lines.append("    #1;")
            lines.append(f'    $fdisplay(file, "{sample_fmt}", {sample_args});')
        else:
            lines.append(f"    @({active_edge} {clk_port.name});")
            lines.append("    #1;")
            lines.append(f'    $fdisplay(file, "{sample_fmt}", {sample_args});')
        return lines

    def _build_case_input_clear(
        self,
        interface: InterfaceInfo,
        clk_port: Optional[Port],
        reset_info: ResetInfo,
    ) -> List[str]:
        """Clear non-clock, non-reset inputs at each case boundary so testcase inputs do not leak across cases."""
        lines: List[str] = []
        for port in interface.input_ports:
            if clk_port and port.name == clk_port.name:
                continue
            if port.name == reset_info.signal_name:
                continue
            lines.append(f"    {port.name} = 0;")
        return lines

    def _reset_assert_literal(self, reset_info: ResetInfo) -> str:
        """Emit the active reset literal derived from semantic reset analysis."""
        return "1'b1" if reset_info.active_level else "1'b0"

    def _reset_deassert_literal(self, reset_info: ResetInfo) -> str:
        """Emit the inactive reset literal so drivers do not accidentally start in reset."""
        return "1'b0" if reset_info.active_level else "1'b1"


    def _verilog_literal(self, value: Any) -> str:
        """Convert JSON/Python-side stimulus values into Verilog-safe literals for generated drivers."""
        if isinstance(value, bool):
            return "1'b1" if value else "1'b0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            cleaned = value.strip()
            lower = cleaned.lower()
            if lower in {"x", "1'bx", "'bx"}:
                return "1'bx"
            if lower in {"z", "1'bz", "'bz"}:
                return "1'bz"
            if lower.startswith("0b"):
                cleaned = "'b" + cleaned[2:]
                lower = cleaned.lower()
            if lower.startswith(("0h", "0x")):
                cleaned = "'h" + cleaned[2:]
                lower = cleaned.lower()
            if lower.startswith("0d"):
                cleaned = "'d" + cleaned[2:]
                lower = cleaned.lower()
            if cleaned.startswith("'") or "'" in cleaned:
                if re.fullmatch(r"\d*'[bBdDhHoO][0-9a-fA-F_xXzZ]+", cleaned):
                    return cleaned
                raise ValueError(f"unsafe Verilog literal: {cleaned}")
            if re.fullmatch(r"-?\d+", cleaned):
                return cleaned
            raise ValueError(f"unsafe Verilog literal: {cleaned}")
        raise TypeError(f"unsupported stimulus value: {value!r}")

    def _run_iverilog(self, candidate_dir: Path) -> Tuple[bool, bool, str]:
        """Compile and execute one DUT/driver pair, returning success flags plus any error text."""
        compile_cmd = [
            self.iverilog_bin,
            "-g2012",
            "-s",
            "__auto_tb_driver",
            "-o",
            "run.vvp",
            "DUT.v",
            "driver.v",
        ]
        try:
            compile_proc = subprocess.run(
                compile_cmd,
                cwd=candidate_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.subprocess_timeout_seconds,
            )
        except FileNotFoundError:
            return False, False, f"{self.iverilog_bin} not found"
        except subprocess.TimeoutExpired:
            return False, False, "compile timeout"

        if compile_proc.returncode != 0:
            return False, False, (compile_proc.stderr or compile_proc.stdout).strip()

        try:
            run_proc = subprocess.run(
                [self.vvp_bin, "run.vvp"],
                cwd=candidate_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.subprocess_timeout_seconds,
            )
        except FileNotFoundError:
            return True, False, f"{self.vvp_bin} not found"
        except subprocess.TimeoutExpired:
            return True, False, "simulation timeout"

        if run_proc.returncode != 0:
            return True, False, (run_proc.stderr or run_proc.stdout).strip()
        return True, True, ""


def parse_tbout(path: str | Path) -> List[Dict[str, Any]]:
    """Parse driver-emitted TBout rows back into structured Python dictionaries."""
    tbout_path = Path(path)
    if not tbout_path.exists():
        return []

    trace: List[Dict[str, Any]] = []
    for line in tbout_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry: Dict[str, Any] = {}
        for item in [part.strip() for part in line.split(",")]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            normalized_key = key.strip()
            entry[normalized_key] = _parse_tbout_value(value.strip(), normalized_key)
        trace.append(entry)
    return trace


def summarize_functional_clusters(clusters: Sequence[FunctionalCluster]) -> List[Dict[str, Any]]:
    """Convert cluster objects into plain JSON-friendly summaries for reports and files."""
    return [
        {
            "cluster_id": cluster.cluster_id,
            "score": cluster.score,
            "size": len(cluster.candidate_ids),
            "candidate_ids": cluster.candidate_ids,
            "representative_candidate_id": cluster.representative_candidate_id,
        }
        for cluster in clusters
    ]


def _semantic_entropy_from_functional_clusters(
    clusters: Sequence[FunctionalCluster],
) -> Optional[float]:
    """Compute semantic entropy from current cluster sizes for stimulus-refinement decisions."""

    cluster_sizes = [len(cluster.candidate_ids) for cluster in clusters if len(cluster.candidate_ids) > 0]
    if not cluster_sizes:
        return None
    return float(semantic_entropy_from_counts(cluster_sizes))


def group_trace_by_case(trace: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group one candidate trace by testcase so per-case disagreement can be inspected easily."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for entry in trace:
        case_idx = entry.get("case")
        if isinstance(case_idx, int):
            grouped.setdefault(case_idx, []).append(entry)
    return grouped


def summarize_per_test_case_clusters(
    simulation_records: Sequence[SimulationRecord],
    output_port_names: Sequence[str],
) -> Dict[str, Any]:
    """Build a testcase-by-testcase clustering view to show where candidates diverge or collapse."""
    per_case_candidates: Dict[int, List[Dict[str, Any]]] = {}
    unsimulated: List[Dict[str, Any]] = []

    for record in simulation_records:
        if record.functional_signature is None or not record.trace:
            unsimulated.append(
                {
                    "candidate_id": record.candidate_id,
                    "interface_cluster_id": record.interface_cluster_id,
                    "error": record.error,
                    "compile_ok": record.compile_ok,
                    "run_ok": record.run_ok,
                }
            )
            continue

        by_case = group_trace_by_case(record.trace)
        for case_idx, case_trace in by_case.items():
            per_case_candidates.setdefault(case_idx, []).append(
                {
                    "candidate_id": record.candidate_id,
                    "interface_cluster_id": record.interface_cluster_id,
                    "trace": case_trace,
                    "signature": case_signature_from_trace(case_trace, output_port_names),
                }
            )

    result: Dict[str, Any] = {"cases": [], "unsimulated_candidates": unsimulated}
    for case_idx in sorted(per_case_candidates):
        case_entries = per_case_candidates[case_idx]
        buckets: Dict[Tuple[Tuple[Any, ...], ...], List[Dict[str, Any]]] = {}
        for item in case_entries:
            buckets.setdefault(item["signature"], []).append(item)

        case_clusters: List[Dict[str, Any]] = []
        for cluster_index, (signature, items) in enumerate(
            sorted(buckets.items(), key=lambda item: (-len(item[1]), str(item[0]))),
            start=1,
        ):
            representative_trace = items[0]["trace"]
            case_clusters.append(
                {
                    "cluster_id": f"case_{case_idx}_cluster_{cluster_index}",
                    "size": len(items),
                    "candidate_ids": sorted(item["candidate_id"] for item in items),
                    "interface_cluster_ids": sorted({item["interface_cluster_id"] for item in items}),
                    "output_signature": [list(row) for row in signature],
                    "trace": representative_trace,
                }
            )

        result["cases"].append(
            {
                "case_index": case_idx,
                "num_clusters": len(case_clusters),
                "clusters": case_clusters,
            }
        )
    return result


def _parse_tbout_value(value: str, key: Optional[str] = None) -> Any:
    """Recover integers for indexing fields while leaving signal values as exact emitted strings."""
    if key in {"case", "step", "cycle"} and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _port_to_jsonable(port: Port) -> Dict[str, Any]:
    """Serialize a port dataclass without losing detailed parse metadata."""
    return asdict(port)


def _interface_to_jsonable(interface: Optional[InterfaceInfo]) -> Optional[Dict[str, Any]]:
    """Serialize an interface object so reports can persist parse results and port details."""
    if interface is None:
        return None
    return {
        "module_name": interface.module_name,
        "raw_header": interface.raw_header,
        "parameter_defaults": dict(interface.parameter_defaults),
        "signature_text": normalize_interface_signature(interface),
        "parse_success": interface.parse_success,
        "warnings": interface.warnings,
        "ports": [_port_to_jsonable(port) for port in interface.ports],
    }
