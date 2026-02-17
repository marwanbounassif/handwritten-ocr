"""
LLM agent definitions for the agentic OCR pipeline.
Each agent is an LLM call to Qwen3 via a specific system prompt with structured JSON output.
"""

import json
import re
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocr_agent.tools import call_llm_json


# ── Pydantic schemas for agent outputs ───────────────────────────


class CriticIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str = ""
    severity: Literal["critical", "minor", "cosmetic"] = "minor"
    suggestion: str = ""


class CriticSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = ""
    confidence: int = Field(default=50, ge=0, le=100)
    issues: list[CriticIssue] = []


class CriticResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    overall_confidence: int = Field(default=0, ge=0, le=100)
    segments: list[CriticSegment] = []
    verdict: Literal["accept", "needs_editing", "needs_reocr"] = "needs_editing"
    reasoning: str = ""


class EditorChange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    original: str = ""
    corrected: str = ""
    reason: str = ""
    confidence: int = Field(default=50, ge=0, le=100)


class EditorResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    corrected_text: str
    changes: list[EditorChange] = []
    unresolved: list[str] = []


class ArbitratorDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    segment: str = ""
    chosen_version: int = Field(default=1, ge=1)
    reason: str = ""


class ArbitratorResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    final_text: str
    decisions: list[ArbitratorDecision] = []
    confidence: int = Field(default=0, ge=0, le=100)
    uncertain_segments: list[str] = []


# ── Schema → prompt helper ────────────────────────────────────────


_MARKER = "\u00a7"  # § sentinel stripped after json.dumps


def _placeholder(annotation, field_info=None) -> object:
    """Return a representative placeholder value for a type annotation."""
    origin = get_origin(annotation)

    # Literal["a", "b", "c"] → marker-wrapped so we can unescape later
    if origin is Literal:
        return _MARKER + " | ".join(f'"{v}"' for v in get_args(annotation)) + _MARKER

    # list[X] → [placeholder(X)]
    if origin is list:
        (inner,) = get_args(annotation)
        return [_placeholder(inner)]

    # Pydantic sub-model → recurse
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_example(annotation)

    # int with ge/le constraints → "<ge-le>"
    if annotation is int and field_info is not None:
        meta = field_info.metadata
        ge = next((m.ge for m in meta if hasattr(m, "ge") and m.ge is not None), None)
        le = next((m.le for m in meta if hasattr(m, "le") and m.le is not None), None)
        if ge is not None and le is not None:
            return _MARKER + f"<{ge}-{le}>" + _MARKER

    if annotation is int:
        return _MARKER + "<integer>" + _MARKER
    if annotation is str:
        return _MARKER + "<string>" + _MARKER

    return _MARKER + "<value>" + _MARKER


def _model_example(model: type[BaseModel]) -> dict:
    """Build an example dict for a Pydantic model using field metadata."""
    example = {}
    for name, field_info in model.model_fields.items():
        example[name] = _placeholder(field_info.annotation, field_info)
    return example


def schema_example(model: type[BaseModel]) -> str:
    """Generate a human-readable JSON example from a Pydantic model class."""
    raw = json.dumps(_model_example(model), indent=2, ensure_ascii=False)
    # Strip the surrounding quotes and unescape inner quotes around markers
    raw = re.sub(
        r'"' + _MARKER + r'(.*?)' + _MARKER + r'"',
        lambda m: m.group(1).replace('\\"', '"'),
        raw,
    )
    return raw


# ── Critic Agent ──────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """\
You are a specialist in analyzing OCR transcriptions of handwritten documents.
You CANNOT see the original image. You work entirely from the text.

Your job: identify problems in the transcription using linguistic analysis.

Look for:
- Nonsense words or character sequences that are not real words
- Broken or split words (OCR often fragments words: "inter alio ing" should be "internalizing")
- Merged words that should be separate
- OCR artifacts: "rn" that should be "m", "cl" that should be "d", "li" that should be "h"
- Missing or garbled punctuation
- Sentences that don't parse grammatically
- Numbers or dates that don't make sense in context
- Inconsistent formatting or random capitalization
- Stray single characters that are noise (random "a", "y", "H" etc.)

Severity levels:
- "critical": the word/phrase is unintelligible or completely wrong
- "minor": the word is slightly garbled but meaning is recoverable
- "cosmetic": punctuation, spacing, or capitalization issues

Be thorough but precise. Don't flag things that are just unusual writing style."""

CRITIC_USER_TEMPLATE = """\
Analyze the following OCR transcription for errors and quality issues.

## Transcription
{transcription}

{previous_critique_section}

## Output format
Respond with ONLY a JSON object matching this schema:
{schema}

Guidelines for verdict:
- "accept": text is coherent and readable, no critical issues, confidence > 85
- "needs_editing": text has identifiable issues that can be fixed from context
- "needs_reocr": text is so garbled that linguistic correction alone won't recover it"""


def run_critic(transcription: str, previous_critique: "CriticResult | None" = None) -> CriticResult:
    """
    Run the Critic Agent on a transcription.
    Returns validated CriticResult with confidence, segments, verdict, reasoning.
    """
    previous_section = ""
    if previous_critique:
        previous_section = (
            "## Previous Critique (for context — the text was edited since)\n"
            f"Previous confidence: {previous_critique.overall_confidence}\n"
            f"Previous verdict: {previous_critique.verdict}\n"
            f"Previous reasoning: {previous_critique.reasoning}"
        )

    user_msg = CRITIC_USER_TEMPLATE.format(
        transcription=transcription,
        previous_critique_section=previous_section,
        schema=schema_example(CriticResult),
    )

    print("  [critic] Analyzing transcription...")
    raw = call_llm_json(CRITIC_SYSTEM_PROMPT, user_msg,
                        json_schema=CriticResult.model_json_schema())

    try:
        result = CriticResult.model_validate(raw)
    except ValidationError as e:
        print(f"  [critic] WARNING: output validation failed: {e}")
        result = CriticResult(
            overall_confidence=0,
            verdict="needs_editing",
            reasoning="LLM output failed schema validation",
        )

    print(f"  [critic] Verdict: {result.verdict} (confidence {result.overall_confidence})")
    return result


# ── Editor Agent ──────────────────────────────────────────────────

EDITOR_SYSTEM_PROMPT = """\
You are a precise text restoration editor for OCR transcriptions.
You CANNOT see the original image. You work entirely from the text and the critic's feedback.

Your strict mandate:
1. Fix ONLY the issues the critic identified. Do not rephrase or paraphrase.
2. Preserve the original wording exactly where the critic did not flag problems.
3. For each fix, explain what you changed and why.
4. If you cannot confidently fix an issue, list it as unresolved.
5. Never add new content. Never rewrite sentences. Only repair OCR damage.

You are revealing the original text that was obscured by OCR errors, not creating new text."""

EDITOR_USER_TEMPLATE = """\
Fix the issues identified by the critic in this OCR transcription.

## Transcription
{transcription}

## Critic's Findings
Overall confidence: {confidence}
Issues found:
{issues_text}

## Output format
Respond with ONLY a JSON object matching this schema:
{schema}

IMPORTANT: The corrected_text must be the COMPLETE transcription with fixes applied, not just the changed parts."""


def run_editor(transcription: str, critique: CriticResult) -> EditorResult:
    """
    Run the Editor Agent to fix issues identified by the Critic.
    Returns validated EditorResult with corrected_text, changes, unresolved.
    """
    # Format issues for the editor
    issues_lines = []
    for seg in critique.segments:
        for issue in seg.issues:
            issues_lines.append(
                f"- [{issue.severity}] \"{seg.text}\" → "
                f"{issue.description} "
                f"(suggestion: {issue.suggestion or 'none'})"
            )

    if not issues_lines:
        issues_lines = ["No specific issues listed."]

    user_msg = EDITOR_USER_TEMPLATE.format(
        transcription=transcription,
        confidence=critique.overall_confidence,
        issues_text="\n".join(issues_lines),
        schema=schema_example(EditorResult),
    )

    print("  [editor] Fixing flagged issues...")
    raw = call_llm_json(EDITOR_SYSTEM_PROMPT, user_msg,
                        json_schema=EditorResult.model_json_schema())

    try:
        result = EditorResult.model_validate(raw)
    except ValidationError as e:
        print(f"  [editor] WARNING: output validation failed: {e}")
        result = EditorResult(corrected_text=transcription)

    print(f"  [editor] Applied {len(result.changes)} fixes, {len(result.unresolved)} unresolved")
    return result


# ── Arbitrator Agent ──────────────────────────────────────────────

ARBITRATOR_SYSTEM_PROMPT = """\
You are an arbitrator comparing multiple OCR transcription versions of the same document.
You CANNOT see the original image. You work entirely from the text versions provided.

Your job:
1. Compare the versions segment by segment
2. For each disagreement, pick the reading that is most linguistically coherent
3. Consider: grammar, context, common OCR error patterns, word frequency
4. Produce a single final merged transcription that takes the best parts of each version
5. Flag segments where no version is convincing

OCR error patterns to consider:
- "rn" ↔ "m", "cl" ↔ "d", "li" ↔ "h" (character confusion)
- Split words: fragments that should be one word
- Merged words: one blob that should be two words
- Stray characters: noise from the scanning process"""

ARBITRATOR_USER_TEMPLATE = """\
Compare these OCR transcription versions and produce the best merged result.

{versions_text}

## Output format
Respond with ONLY a JSON object matching this schema:
{schema}"""


def run_arbitrator(versions: list[dict]) -> ArbitratorResult:
    """
    Run the Arbitrator Agent to merge multiple transcription versions.
    Each version dict should have: {text, source, score (optional)}
    Returns validated ArbitratorResult with final_text, decisions, confidence, uncertain_segments.
    """
    versions_text_parts = []
    for i, v in enumerate(versions, 1):
        score_info = f" (critic score: {v.get('score', 'N/A')})" if "score" in v else ""
        versions_text_parts.append(
            f"## Version {i} — {v.get('source', 'unknown')}{score_info}\n{v['text']}"
        )

    user_msg = ARBITRATOR_USER_TEMPLATE.format(
        versions_text="\n\n".join(versions_text_parts),
        schema=schema_example(ArbitratorResult),
    )

    print(f"  [arbitrator] Comparing {len(versions)} versions...")
    raw = call_llm_json(ARBITRATOR_SYSTEM_PROMPT, user_msg,
                        json_schema=ArbitratorResult.model_json_schema())

    try:
        result = ArbitratorResult.model_validate(raw)
    except ValidationError as e:
        print(f"  [arbitrator] WARNING: output validation failed: {e}")
        result = ArbitratorResult(
            final_text=versions[0]["text"] if versions else "",
        )

    print(f"  [arbitrator] Merged (confidence {result.confidence})")
    return result
