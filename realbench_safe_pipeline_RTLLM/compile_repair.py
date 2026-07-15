"""Compile-time Verilog repair helpers.

This file isolates the "Verilog syntax / compile repair" stage from the rest
of the pipeline.

Primary interface
-----------------
repair_candidate(candidate, verification_summary, backend)

Inputs:
- candidate: CandidateVerilog
  The current compile-failed Verilog candidate. The most important fields are
  `candidate.clean_verilog` and `candidate.candidate_id`.
- verification_summary: str
  The compile-time failure summary returned by the compile-only checker.
- backend: LLMBackend
  The backend used to ask the model to repair the Verilog.

Outputs:
- repaired: CandidateVerilog
  A new repaired candidate, typically with candidate_id like
  `generated_repair_1`, `generated_repair_2`, ...
- record: dict[str, Any]
  Full LLM I/O record for trace/debugging, including:
  - prompt/response
  - verilog_before
  - verilog_after
  - compile_summary

Pipeline position
-----------------
This module is called only after a candidate fails compile/elaboration.
It does not decide functional correctness; it only tries to repair
syntax/compile-time issues using the current code plus compile error text.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from .llm import LLMBackend
from .models import CandidateVerilog


def _mask_verilog_comments(text: str) -> str:
    """Replace comment contents with spaces while preserving string length."""

    def _replace_block(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if char == "\n" else " " for char in value)

    masked = re.sub(r"/\*.*?\*/", _replace_block, text, flags=re.DOTALL)
    masked = re.sub(r"//.*?$", lambda match: " " * len(match.group(0)), masked, flags=re.MULTILINE)
    return masked


def repair_candidate(
    candidate: CandidateVerilog,
    verification_summary: str,
    backend: LLMBackend,
) -> tuple[CandidateVerilog, dict[str, Any]]:
    """Ask the backend to repair a compile-failed candidate."""

    system_prompt = (
        "You are an expert at writing Verilog code. "
        "Repair the Verilog using only the current code and compile-time failure information."
    )
    user_prompt = f"""
Original candidate:
{candidate.clean_verilog}

Compile-time failure information:
{verification_summary}

Return the final answer as a fenced ```verilog``` block containing only the repaired Verilog/SystemVerilog module code.
Do not include any prose before or after the fenced code block.
""".strip()
    record = backend.complete_text_record(system_prompt=system_prompt, user_prompt=user_prompt)
    raw = str(record["raw_response"])
    clean = extract_verilog_module_with_warning(
        raw,
        context=f"{candidate.candidate_id}:repair_candidate",
        fallback_text=candidate.clean_verilog,
    )
    repaired = CandidateVerilog(
        candidate_id=_next_repair_candidate_id(candidate.candidate_id),
        raw_response=raw,
        clean_verilog=clean,
        module_name=guess_module_name(clean),
        generation_notes=f"Repaired from {candidate.candidate_id} after verification failure.",
    )
    record["verilog_before"] = candidate.clean_verilog
    record["verilog_after"] = repaired.clean_verilog
    record["compile_summary"] = verification_summary
    return repaired, record


def extract_verilog_module(text: str) -> str:
    """Extract Verilog, preferring the last fenced block and preserving multiple modules."""

    fenced = re.findall(r"```(?:verilog|systemverilog)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced[-1]] if fenced else [text]
    for candidate in candidates:
        masked_candidate = _mask_verilog_comments(candidate)
        pattern = re.compile(r"(^|\n)\s*module\b.*?\bendmodule\b", flags=re.DOTALL)
        matches = list(pattern.finditer(masked_candidate))
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


def _next_repair_candidate_id(candidate_id: str) -> str:
    match = re.fullmatch(r"generated(?:_repair_(\d+))?", candidate_id)
    if not match:
        return f"{candidate_id}_repair_1"
    current = match.group(1)
    next_index = 1 if current is None else int(current) + 1
    return f"generated_repair_{next_index}"
