"""
LLM agent definitions for the agentic OCR pipeline.
Each agent is an LLM call to Qwen3 via a specific system prompt with structured JSON output.
"""

import json
import re
from collections import Counter
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocr_agent import config
from ocr_agent.tools import call_llm_json


# ── Pydantic schemas for agent outputs ───────────────────────────


class CriticIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str = ""
    severity: Literal["critical", "minor", "cosmetic", "ambiguous"] = "minor"
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


# ── Scholar Council models ───────────────────────────────────────


class ScholarReading(BaseModel):
    model_config = ConfigDict(extra="ignore")
    segment_text: str = ""
    proposed_reading: str = ""
    reasoning: str = ""


class ScholarResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    overall_confidence: int = Field(default=0, ge=0, le=100)
    segments: list[CriticSegment] = []
    verdict: Literal["accept", "needs_editing", "needs_reocr"] = "needs_editing"
    reasoning: str = ""
    readings: list[ScholarReading] = []


class ScholarVote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    segment_text: str = ""
    winning_reading: str = ""
    vote_counts: dict[str, int] = {}
    total_votes: int = 0


class CouncilResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    final_text: str
    consensus_verdict: str = "needs_editing"
    consensus_confidence: int = Field(default=0, ge=0, le=100)
    votes: list[ScholarVote] = []
    verdict_tally: dict[str, int] = {}
    merged_critique: CriticResult = Field(default_factory=CriticResult)


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


# ── Scholar Council ──────────────────────────────────────────────

SCHOLAR_PERSPECTIVES = [
    {
        "name": "Grammarian",
        "focus": (
            "You are a grammarian. Focus on grammar, syntax, and subject-verb agreement. "
            "Flag any sentence that doesn't parse correctly. Pay close attention to verb tenses, "
            "articles, prepositions, and word order that OCR may have garbled."
        ),
    },
    {
        "name": "Contextual Analyst",
        "focus": (
            "You are a contextual analyst. Focus on semantic coherence and thematic flow. "
            "Does each sentence make sense in the context of the surrounding text? "
            "Flag segments where the meaning seems implausible or contradicts the document's theme. "
            "Propose readings that restore semantic plausibility."
        ),
    },
    {
        "name": "Paleography Expert",
        "focus": (
            "You are a paleography expert specializing in handwriting recognition errors. "
            "Focus on common character confusion patterns: 'rn' ↔ 'm', 'cl' ↔ 'd', 'li' ↔ 'h', "
            "'vv' ↔ 'w', 'ri' ↔ 'n'. Also look for split words, merged words, and misread "
            "ascenders/descenders (t/l, p/b, etc.)."
        ),
    },
    {
        "name": "Frequency Analyst",
        "focus": (
            "You are a frequency analyst. Focus on word frequency, common collocations, and "
            "idiomatic phrases. If a word or phrase is extremely rare but a common alternative "
            "exists that would fit the context, flag it as ambiguous and propose the more likely "
            "reading. Consider whether word combinations are natural in English."
        ),
    },
    {
        "name": "Document Specialist",
        "focus": (
            "You are a document specialist. Focus on cross-document consistency, formatting "
            "patterns, and structural coherence. Check that names, dates, and references are "
            "internally consistent. Use any provided context passages from other pages to "
            "disambiguate readings."
        ),
    },
]

SCHOLAR_USER_TEMPLATE = """\
Analyze the following OCR transcription for errors and quality issues.

## Transcription
{transcription}

{context_section}

{previous_critique_section}

## Output format
Respond with ONLY a JSON object matching this schema:
{schema}

Guidelines for verdict:
- "accept": text is coherent and readable, no critical issues, confidence > 85
- "needs_editing": text has identifiable issues that can be fixed from context
- "needs_reocr": text is so garbled that linguistic correction alone won't recover it

For ambiguous segments (where two valid readings exist), use severity "ambiguous" and provide
your proposed reading in the readings array."""

ARBITRATOR_COUNCIL_USER_TEMPLATE = """\
Compare these OCR transcription versions and vote on which reading is best for each segment.

{versions_text}

{context_section}

## Output format
Respond with ONLY a JSON object matching this schema:
{schema}

For each segment where the versions disagree, include an entry in readings with your proposed
reading and reasoning. Your verdict should reflect the overall quality of the best merged result."""


def run_scholar(
    transcription: str,
    perspective: dict,
    context_passages: list[dict],
    temperature: float,
    previous_critique: CriticResult | None = None,
) -> ScholarResult:
    """Run one scholar with its specific system prompt and temperature."""
    system_prompt = CRITIC_SYSTEM_PROMPT + "\n\n## Your Specialist Lens\n" + perspective["focus"]

    context_section = ""
    if context_passages:
        passages_text = "\n".join(
            f"- [{p.get('source', 'unknown')}]: {p.get('text', '')}"
            for p in context_passages
        )
        context_section = f"## Context Passages (from other pages/documents)\n{passages_text}"

    previous_section = ""
    if previous_critique:
        previous_section = (
            "## Previous Critique (for context — the text was edited since)\n"
            f"Previous confidence: {previous_critique.overall_confidence}\n"
            f"Previous verdict: {previous_critique.verdict}\n"
            f"Previous reasoning: {previous_critique.reasoning}"
        )

    user_msg = SCHOLAR_USER_TEMPLATE.format(
        transcription=transcription,
        context_section=context_section,
        previous_critique_section=previous_section,
        schema=schema_example(ScholarResult),
    )

    name = perspective["name"]
    print(f"  [scholar:{name}] Analyzing transcription (temp={temperature})...")
    raw = call_llm_json(
        system_prompt, user_msg,
        temperature=temperature,
        json_schema=ScholarResult.model_json_schema(),
    )

    try:
        result = ScholarResult.model_validate(raw)
    except ValidationError as e:
        print(f"  [scholar:{name}] WARNING: output validation failed: {e}")
        result = ScholarResult(
            overall_confidence=0,
            verdict="needs_editing",
            reasoning="LLM output failed schema validation",
        )

    print(f"  [scholar:{name}] Verdict: {result.verdict} (confidence {result.overall_confidence})")
    return result


def _tally_votes(scholar_results: list[ScholarResult], transcription: str) -> CouncilResult:
    """Tally votes from all scholars into a CouncilResult."""
    # Tally verdict votes
    verdict_counts: Counter[str] = Counter()
    for r in scholar_results:
        verdict_counts[r.verdict] += 1
    consensus_verdict = verdict_counts.most_common(1)[0][0]

    # Average confidence
    consensus_confidence = round(
        sum(r.overall_confidence for r in scholar_results) / len(scholar_results)
    )

    # Tally per-segment readings (only for ambiguous segments with proposed readings)
    reading_votes: dict[str, Counter[str]] = {}
    for r in scholar_results:
        for reading in r.readings:
            key = reading.segment_text
            if key not in reading_votes:
                reading_votes[key] = Counter()
            reading_votes[key][reading.proposed_reading] += 1

    votes = []
    final_text = transcription
    for segment_text, counts in reading_votes.items():
        winning_reading, winning_count = counts.most_common(1)[0]
        votes.append(ScholarVote(
            segment_text=segment_text,
            winning_reading=winning_reading,
            vote_counts=dict(counts),
            total_votes=sum(counts.values()),
        ))
        # Apply winning reading to the transcription
        if segment_text in final_text:
            final_text = final_text.replace(segment_text, winning_reading, 1)

    # Merge all scholars' segments into a single critique (union, highest severity wins)
    seen_segments: dict[str, CriticSegment] = {}
    severity_rank = {"critical": 3, "ambiguous": 2, "minor": 1, "cosmetic": 0}
    for r in scholar_results:
        for seg in r.segments:
            key = seg.text
            if key not in seen_segments:
                seen_segments[key] = seg
            else:
                existing = seen_segments[key]
                # Keep the one with highest severity issues
                existing_max = max(
                    (severity_rank.get(i.severity, 0) for i in existing.issues), default=0
                )
                new_max = max(
                    (severity_rank.get(i.severity, 0) for i in seg.issues), default=0
                )
                if new_max > existing_max:
                    seen_segments[key] = seg

    merged_critique = CriticResult(
        overall_confidence=consensus_confidence,
        segments=list(seen_segments.values()),
        verdict=consensus_verdict,
        reasoning=f"Council consensus ({dict(verdict_counts)})",
    )

    return CouncilResult(
        final_text=final_text,
        consensus_verdict=consensus_verdict,
        consensus_confidence=consensus_confidence,
        votes=votes,
        verdict_tally=dict(verdict_counts),
        merged_critique=merged_critique,
    )


def run_council(
    transcription: str,
    context_passages: list[dict],
    previous_critique: CriticResult | None = None,
) -> CouncilResult:
    """
    Run all scholars sequentially, tally votes, and return a CouncilResult.
    """
    temperatures = config.SCHOLAR_TEMPERATURES
    scholar_count = config.SCHOLAR_COUNT
    perspectives = SCHOLAR_PERSPECTIVES[:scholar_count]

    print(f"\n  [council] Running {len(perspectives)} scholars...")
    results = []
    for i, perspective in enumerate(perspectives):
        temp = temperatures[i] if i < len(temperatures) else 0.3
        result = run_scholar(
            transcription=transcription,
            perspective=perspective,
            context_passages=context_passages,
            temperature=temp,
            previous_critique=previous_critique,
        )
        results.append(result)

    council = _tally_votes(results, transcription)
    print(
        f"  [council] Consensus: {council.consensus_verdict} "
        f"(confidence {council.consensus_confidence}, "
        f"tally {council.verdict_tally})"
    )
    return council


def run_arbitrator_council(
    versions: list[dict],
    context_passages: list[dict],
) -> CouncilResult:
    """
    Variant of run_council for the reocr path.
    Each scholar compares two transcription versions and votes on the best segments.
    """
    versions_text_parts = []
    for i, v in enumerate(versions, 1):
        score_info = f" (critic score: {v.get('score', 'N/A')})" if "score" in v else ""
        versions_text_parts.append(
            f"## Version {i} — {v.get('source', 'unknown')}{score_info}\n{v['text']}"
        )
    versions_text = "\n\n".join(versions_text_parts)

    context_section = ""
    if context_passages:
        passages_text = "\n".join(
            f"- [{p.get('source', 'unknown')}]: {p.get('text', '')}"
            for p in context_passages
        )
        context_section = f"## Context Passages (from other pages/documents)\n{passages_text}"

    temperatures = config.SCHOLAR_TEMPERATURES
    scholar_count = config.SCHOLAR_COUNT
    perspectives = SCHOLAR_PERSPECTIVES[:scholar_count]

    print(f"\n  [arbitrator-council] Running {len(perspectives)} scholars on {len(versions)} versions...")
    results = []
    for i, perspective in enumerate(perspectives):
        temp = temperatures[i] if i < len(temperatures) else 0.3
        system_prompt = (
            ARBITRATOR_SYSTEM_PROMPT + "\n\n## Your Specialist Lens\n" + perspective["focus"]
        )

        user_msg = ARBITRATOR_COUNCIL_USER_TEMPLATE.format(
            versions_text=versions_text,
            context_section=context_section,
            schema=schema_example(ScholarResult),
        )

        name = perspective["name"]
        print(f"  [scholar:{name}] Comparing versions (temp={temp})...")
        raw = call_llm_json(
            system_prompt, user_msg,
            temperature=temp,
            json_schema=ScholarResult.model_json_schema(),
        )

        try:
            result = ScholarResult.model_validate(raw)
        except ValidationError as e:
            print(f"  [scholar:{name}] WARNING: output validation failed: {e}")
            result = ScholarResult(
                overall_confidence=0,
                verdict="needs_editing",
                reasoning="LLM output failed schema validation",
            )

        print(f"  [scholar:{name}] Verdict: {result.verdict} (confidence {result.overall_confidence})")
        results.append(result)

    # Use the first version's text as the base for applying reading votes
    base_text = versions[0]["text"] if versions else ""
    council = _tally_votes(results, base_text)
    print(
        f"  [arbitrator-council] Consensus: {council.consensus_verdict} "
        f"(confidence {council.consensus_confidence}, "
        f"tally {council.verdict_tally})"
    )
    return council
