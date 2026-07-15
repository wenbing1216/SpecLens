"""LLM backends, `.env` loading, and provider wiring."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class LLMBackend(Protocol):
    """Simple interface so model providers can be swapped cleanly."""

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return plain-text completion output."""

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return JSON-like structured output."""

    def complete_text_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return a structured record for one text completion call."""

    def complete_json_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return a structured record for one JSON completion call."""


@dataclass(slots=True)
class LLMCallRecord:
    """Serializable record for one LLM invocation."""

    backend: str
    mode: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_response: Any
    usage: dict[str, Any] | None = None
    elapsed_seconds: float | None = None
    timeout_retry_triggered: bool = False
    timeout_retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderBundle:
    """Concrete backend assignment for each pipeline stage family."""

    planner_backend: LLMBackend
    answerer_backend: LLMBackend
    reviewer_backend: LLMBackend
    generator_backend: LLMBackend
    repair_backend: LLMBackend


@dataclass(slots=True)
class StubBackend:
    """Fallback backend for local dry runs without real API calls."""

    name: str = "stub"

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_text_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["raw_response"]

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.complete_json_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["parsed_response"]

    def complete_text_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del system_prompt
        if "endmodule" in user_prompt:
            raw = user_prompt
        else:
            raw = (
                f"Stub backend `{self.name}` is active. Replace it with a real provider "
                "to produce model-driven outputs."
            )
        return LLMCallRecord(
            backend=self.name,
            mode="text",
            system_prompt="",
            user_prompt=user_prompt,
            raw_response=raw,
            parsed_response=raw,
            usage=None,
            elapsed_seconds=0.0,
        ).to_dict()

    def complete_json_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del system_prompt
        parsed = {
            "backend": self.name,
            "note": "No external LLM configured.",
            "echo_preview": user_prompt[:400],
        }
        return LLMCallRecord(
            backend=self.name,
            mode="json",
            system_prompt="",
            user_prompt=user_prompt,
            raw_response=json.dumps(parsed, ensure_ascii=False, indent=2),
            parsed_response=parsed,
            usage=None,
            elapsed_seconds=0.0,
        ).to_dict()


@dataclass(slots=True)
class GeminiBackend:
    """Gemini REST backend using `generateContent`."""

    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    timeout_sec: int = 2000
    max_retries: int = 5
    _last_timeout_retry_count: int = 0

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_text_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["raw_response"]

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.complete_json_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["parsed_response"]

    def complete_text_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "text/plain",
                "temperature": self.temperature,
            },
        }
        data = self._post_generate_content(payload)
        raw = _extract_gemini_text(data)
        return LLMCallRecord(
            backend=f"Gemini:{self.model}",
            mode="text",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw,
            parsed_response=raw,
            usage=_extract_gemini_usage(data),
            elapsed_seconds=time.perf_counter() - started_at,
            timeout_retry_triggered=self._last_timeout_retry_count > 0,
            timeout_retry_count=self._last_timeout_retry_count,
        ).to_dict()

    def complete_json_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
            },
        }
        data = self._post_generate_content(payload)
        text = _extract_gemini_text(data)
        parsed = _parse_json_object(text, provider=f"Gemini ({self.model})")
        return LLMCallRecord(
            backend=f"Gemini:{self.model}",
            mode="json",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=text,
            parsed_response=parsed,
            usage=_extract_gemini_usage(data),
            elapsed_seconds=time.perf_counter() - started_at,
            timeout_retry_triggered=self._last_timeout_retry_count > 0,
            timeout_retry_count=self._last_timeout_retry_count,
        ).to_dict()

    def _post_generate_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        data, timeout_retry_count = _post_json_with_retry(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
            provider=f"Gemini ({self.model})",
            max_retries=self.max_retries,
        )
        self._last_timeout_retry_count = timeout_retry_count
        return data


@dataclass(slots=True)
class OpenAIResponsesBackend:
    """OpenAI backend using the Chat Completions API."""

    api_key: str
    model: str = "o3-mini"
    reasoning_effort: str = "medium"
    temperature: float = 0.7
    timeout_sec: int = 2000
    max_retries: int = 5
    _last_timeout_retry_count: int = 0

    def _is_reasoning_family(self) -> bool:
        return self.model.startswith("o") or self.model.startswith("gpt-5")

    def _uses_developer_role(self) -> bool:
        return self._is_reasoning_family()

    def _supports_reasoning_effort(self) -> bool:
        return self._is_reasoning_family()

    def _supports_custom_temperature(self) -> bool:
        return not self._is_reasoning_family()

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_text_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["raw_response"]

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.complete_json_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )["parsed_response"]

    def complete_text_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        response, timeout_retry_count = self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=None,
        )
        raw = _extract_chat_completion_text(response)
        return LLMCallRecord(
            backend=f"OpenAI:{self.model}",
            mode="text",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw,
            parsed_response=raw,
            usage=_extract_chat_completion_usage(response),
            elapsed_seconds=time.perf_counter() - started_at,
            timeout_retry_triggered=timeout_retry_count > 0,
            timeout_retry_count=timeout_retry_count,
        ).to_dict()

    def complete_json_record(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        response, timeout_retry_count = self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )
        text = _extract_chat_completion_text(response)
        parsed = _parse_json_object(text, provider=f"OpenAI ({self.model})")
        return LLMCallRecord(
            backend=f"OpenAI:{self.model}",
            mode="json",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=text,
            parsed_response=parsed,
            usage=_extract_chat_completion_usage(response),
            elapsed_seconds=time.perf_counter() - started_at,
            timeout_retry_triggered=timeout_retry_count > 0,
            timeout_retry_count=timeout_retry_count,
        ).to_dict()

    def _create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None,
    ) -> tuple[Any, int]:
        if OpenAI is None:
            raise RuntimeError(
                "The `openai` Python package is required for OpenAI Chat Completions. "
                "Install it with `pip install openai`."
            )

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_sec)
        instruction_role = "developer" if self._uses_developer_role() else "system"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": instruction_role, "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._supports_reasoning_effort():
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self._supports_custom_temperature():
            kwargs["temperature"] = self.temperature
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        timeout_retry_count = 0
        for attempt in range(self.max_retries + 1):
            try:
                self._last_timeout_retry_count = timeout_retry_count
                return client.chat.completions.create(**kwargs), timeout_retry_count
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if _is_timeout_like_exception(exc):
                    timeout_retry_count += 1
                if attempt >= self.max_retries:
                    break
                _sleep_before_retry(f"OpenAI ({self.model})", attempt, "chat completion error")

        assert last_error is not None
        raise RuntimeError(
            f"OpenAI ({self.model}) chat completion failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


def load_dotenv_file(env_path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Load simple `KEY=VALUE` pairs from `.env` into process environment."""

    path = Path(env_path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def build_provider_bundle(
    env_path: str | Path = ".env",
    allow_stub_fallback: bool = False,
    provider: str | None = None,
) -> ProviderBundle:
    """Create the concrete provider mapping used by the pipeline."""

    # Treat the requested `.env` file as the source of truth for each fresh CLI
    # launch so updated API keys replace any stale inherited environment values.
    load_dotenv_file(env_path=env_path, override=True)
    selected_provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).strip().lower() or "gemini"
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    openai_model = os.environ.get("OPENAI_MODEL", "o3-mini").strip() or "o3-mini"
    openai_reasoning_effort = (
        os.environ.get("OPENAI_REASONING_EFFORT", "medium").strip().lower() or "medium"
    )

    if selected_provider not in {"gemini", "openai"}:
        raise RuntimeError(
            f"Unsupported provider '{selected_provider}'. Expected one of: gemini, openai."
        )

    required_keys: list[str] = []
    if selected_provider == "gemini" and not gemini_key:
        required_keys.append("GEMINI_API_KEY")
    if selected_provider == "openai" and not openai_key:
        required_keys.append("OPENAI_API_KEY")

    if required_keys:
        if allow_stub_fallback:
            stub = StubBackend()
            return ProviderBundle(
                planner_backend=stub,
                answerer_backend=stub,
                reviewer_backend=stub,
                generator_backend=stub,
                repair_backend=stub,
            )
        raise RuntimeError(
            "Missing API keys in environment or .env: " + ", ".join(required_keys)
        )

    gemini = GeminiBackend(api_key=gemini_key, model=gemini_model)
    openai = OpenAIResponsesBackend(
        api_key=openai_key,
        model=openai_model,
        reasoning_effort=openai_reasoning_effort,
    )
    active = gemini if selected_provider == "gemini" else openai
    return ProviderBundle(
        planner_backend=active,
        answerer_backend=active,
        reviewer_backend=active,
        generator_backend=active,
        repair_backend=active,
    )


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini response contains no candidates: {data}")
    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not texts:
        raise RuntimeError(f"Gemini response contains no text parts: {data}")
    return "\n".join(texts).strip()


def _extract_gemini_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    return {
        "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
        "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
        "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
        "raw_usage": usage,
    }


def _extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError(f"OpenAI chat completion contains no choices: {response}")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise RuntimeError(f"OpenAI chat completion contains no text content: {response}")


def _extract_chat_completion_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    raw_usage = _serialize_openai_usage(usage)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "raw_usage": raw_usage,
    }


def _serialize_openai_usage(usage: Any) -> dict[str, Any]:
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dumped
    raw: dict[str, Any] = {}
    for name in dir(usage):
        if name.startswith("_"):
            continue
        try:
            value = getattr(usage, name)
        except Exception:
            continue
        if callable(value):
            continue
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump()
            except Exception:
                value = str(value)
        elif not isinstance(value, (str, int, float, bool, dict, list, type(None))):
            value = str(value)
        raw[name] = value
    return raw


def _parse_json_object(text: str, provider: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]
    if candidate.endswith("```"):
        candidate = candidate[:-3]
    candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired_candidate = _repair_common_json_escaping(candidate)
        try:
            parsed = json.loads(repaired_candidate)
        except json.JSONDecodeError as repaired_exc:
            raise RuntimeError(
                f"{provider} returned non-JSON content when JSON was required.\n"
                f"Raw response:\n{candidate}"
            ) from repaired_exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{provider} returned JSON that is not an object: {parsed!r}")
    return parsed


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _repair_common_json_escaping(text: str) -> str:
    """Escape stray backslashes often produced in math-heavy model outputs."""

    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _raise_for_status_with_body(response: requests.Response, provider: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        raise RuntimeError(
            f"{provider} API request failed with HTTP {response.status_code}.\n{body}"
        ) from exc


def _post_json_with_retry(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: int,
    provider: str,
    max_retries: int,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    timeout_retry_count = 0

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )
            _raise_for_status_with_body(response, provider=provider)
            return response.json(), timeout_retry_count
        except requests.Timeout as exc:
            last_error = exc
            timeout_retry_count += 1
            if attempt >= max_retries:
                break
            _sleep_before_retry(provider, attempt, "request timeout")
        except requests.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if not _is_retryable_status(status_code) or attempt >= max_retries:
                break
            _sleep_before_retry(provider, attempt, f"HTTP {status_code}")
        except RuntimeError as exc:
            last_error = exc
            status_code = _extract_status_code_from_runtime_error(exc)
            if not _is_retryable_status(status_code) or attempt >= max_retries:
                break
            reason = f"HTTP {status_code}" if status_code is not None else "transient runtime error"
            _sleep_before_retry(provider, attempt, reason)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{provider} API request failed without a captured exception.")


def _is_timeout_like_exception(exc: Exception) -> bool:
    """Best-effort detection for provider timeouts across SDKs."""

    if isinstance(exc, TimeoutError):
        return True
    name = exc.__class__.__name__.lower()
    if "timeout" in name or "timedout" in name:
        return True
    message = str(exc).lower()
    timeout_markers = (
        "timeout",
        "timed out",
        "read timeout",
        "connect timeout",
        "request timeout",
    )
    return any(marker in message for marker in timeout_markers)


def _sleep_before_retry(provider: str, attempt: int, reason: str) -> None:
    delay_sec = min(20.0, (2 ** attempt) + random.uniform(0.0, 1.0))
    print(
        f"{provider} request failed with {reason}; retrying in {delay_sec:.1f}s "
        f"(attempt {attempt + 1})."
    )
    time.sleep(delay_sec)


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504, 520}


def _extract_status_code_from_runtime_error(exc: RuntimeError) -> int | None:
    match = re.search(r"HTTP\s+(\d{3})", str(exc))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
