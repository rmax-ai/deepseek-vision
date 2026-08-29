"""Deterministic case evaluators (pure functions, no LLM required).

Only ``llm_judge`` inspects model text; the rest compare structured output
against the case's expected facts.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel

from deepseek_vision.models import VerificationResult
from deepseek_vision.usage import UsageTracker

from .cases import SCHEMAS, VisionEvalCase


def _render_text(result: dict) -> str:
    """Concatenate every textual surface of a result for substring checks."""
    parts: list[str] = []
    data = result.get("data")
    if isinstance(data, BaseModel):
        parts.append(json.dumps(data.model_dump(mode="json"), default=str))
    elif data is not None:
        parts.append(json.dumps(data, default=str))
    synthesis = result.get("synthesis")
    if synthesis:
        parts.append(str(synthesis))
    for obs in result.get("observations", []):
        parts.append(str(obs.get("text", "")))
    return "\n".join(parts)


def _values_match(expected: Any, actual: Any) -> bool:
    """Match expected vs actual with float tolerance and string contains."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(expected), float(actual), rel_tol=1e-2, abs_tol=1e-2
        )
    if isinstance(expected, str):
        expected_l = expected.lower()
        actual_l = str(actual).lower()
        return expected_l in actual_l or actual_l in expected_l
    return expected == actual


def structured_exact(
    result: dict, case: VisionEvalCase
) -> tuple[bool, str]:
    """Data parses as the declared schema and every expected field matches.

    Floats are compared with 1e-2 relative tolerance (pytest.approx-like);
    strings match case-insensitively by containment.
    """
    data = result.get("data")
    if data is None:
        return False, "no structured data in result"

    schema_name = case.output_schema_name
    if schema_name and schema_name in SCHEMAS:
        model_type = SCHEMAS[schema_name]
        if not isinstance(data, model_type):
            return (
                False,
                f"data is {type(data).__name__}, expected {model_type.__name__}",
            )
        data_dict = data.model_dump()
    elif isinstance(data, BaseModel):
        data_dict = data.model_dump()
    elif isinstance(data, dict):
        data_dict = data
    else:
        return False, "data is neither a dict nor a model"

    mismatches: list[str] = []
    for key, want in (case.expected or {}).items():
        got = data_dict.get(key)
        if not _values_match(want, got):
            mismatches.append(f"{key}: expected {want!r}, got {got!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "all expected fields matched"


def llm_judge(result: dict, case: VisionEvalCase) -> tuple[bool, str]:
    """Textual check: every expected term appears in the rendered result.

    Checks ``answers_include`` and ``differences_include`` lists from
    ``case.expected`` against the rendered text of data/synthesis/
    observations, case-insensitively.
    """
    expected = case.expected or {}
    terms = list(expected.get("answers_include", [])) + list(
        expected.get("differences_include", [])
    )
    if not terms:
        return True, "no expected terms to check"
    text = _render_text(result)
    lowered = text.lower()
    missed = [term for term in terms if term.lower() not in lowered]
    if missed:
        return False, f"missing expected terms: {missed}"
    return True, "all expected terms present in output"


def verification_exact(
    result: dict, case: VisionEvalCase
) -> tuple[bool, str]:
    """Compare expected claim statuses against a VerificationResult.

    Claim keys may match by substring in either direction.
    """
    data = result.get("data")
    expected = case.expected or {}
    statuses = expected.get("statuses", {})
    if not isinstance(data, VerificationResult):
        return (
            False,
            f"data is {type(data).__name__}, expected VerificationResult",
        )
    actual = {check.claim: check.status for check in data.checks}
    mismatches: list[str] = []
    for want_claim, want_status in statuses.items():
        matched = next(
            (
                claim
                for claim in actual
                if want_claim.lower() in claim.lower()
                or claim.lower() in want_claim.lower()
            ),
            None,
        )
        if matched is None:
            mismatches.append(f"no check matching {want_claim!r}")
        elif actual[matched] != want_status:
            mismatches.append(
                f"claim {matched!r}: expected {want_status}, got {actual[matched]}"
            )
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "all expected statuses matched"


def structured_valid(result: dict, case: VisionEvalCase) -> bool:
    """True when data parses as the declared schema (or a dict/list)."""
    data = result.get("data")
    if data is None:
        return False
    schema_name = case.output_schema_name
    if schema_name and schema_name in SCHEMAS:
        return isinstance(data, SCHEMAS[schema_name])
    return isinstance(data, (BaseModel, dict, list))


def grounding_score(evidence: list, case: VisionEvalCase) -> float:
    """Fraction of evidence entries with usable provenance.

    For document/video cases an entry needs a source AND a page or
    timestamp; for image cases a source suffices. 1.0 when no evidence is
    required (empty input).
    """
    if not evidence:
        return 1.0
    anchored = "document" in case.tags or "video" in case.tags
    good = 0
    for entry in evidence:
        source = entry.get("source") if isinstance(entry, dict) else getattr(
            entry, "source", None
        )
        if not source:
            continue
        if anchored:
            page = entry.get("page") if isinstance(entry, dict) else getattr(
                entry, "page", None
            )
            timestamp = (
                entry.get("timestamp_seconds")
                if isinstance(entry, dict)
                else getattr(entry, "timestamp_seconds", None)
            )
            if page is None and timestamp is None:
                continue
        good += 1
    return good / len(evidence)


def success_latency_token_metrics(provider_result: Any, case: VisionEvalCase) -> dict:
    """Latency/request/token/cost metrics for a provider response.

    Token counts and cost are only populated when the provider's usage dict
    carries them; otherwise they are null.
    """
    usage = getattr(provider_result, "usage", None) or {}
    has_tokens = (
        usage.get("prompt_tokens") is not None
        or usage.get("completion_tokens") is not None
    )
    if has_tokens:
        tracker = UsageTracker()
        tracker.add_request(usage, 0.0)
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cost = tracker.estimate_cost_usd()
    else:
        input_tokens = output_tokens = cost = None
    return {
        "latency_s": getattr(provider_result, "latency_s", None),
        "requests": getattr(provider_result, "requests", 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }


EVALUATORS: dict[str, Any] = {
    "structured_exact": structured_exact,
    "llm_judge": llm_judge,
    "verification_exact": verification_exact,
}
