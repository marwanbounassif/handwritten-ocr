"""
LLM agent definitions for the agentic OCR pipeline.
Each agent is an LLM call to Qwen3 via a specific system prompt with structured JSON output.
"""

from typing import Literal

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
    chosen_version: int = 1
    reason: str = ""


class ArbitratorResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    final_text: str
    decisions: list[ArbitratorDecision] = []
    confidence: int = Field(default=0, ge=0, le=100)
    uncertain_segments: list[str] = []

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
Respond with ONLY a JSON object in this exact structure:
{{
  "overall_confidence": <0-100>,
  "segments": [
    {{
      "text": "<the problematic text>",
      "confidence": <0-100>,
      "issues": [
        {{
          "description": "<what's wrong>",
          "severity": "critical" | "minor" | "cosmetic",
          "suggestion": "<what it should probably be>"
        }}
      ]
    }}
  ],
  "verdict": "accept" | "needs_editing" | "needs_reocr",
  "reasoning": "<brief explanation of the verdict>"
}}

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
Respond with ONLY a JSON object in this exact structure:
{{
  "corrected_text": "<the full corrected transcription>",
  "changes": [
    {{
      "original": "<original problematic text>",
      "corrected": "<what you changed it to>",
      "reason": "<why this change>",
      "confidence": <0-100>
    }}
  ],
  "unresolved": [
    "<description of issues you couldn't confidently fix>"
  ]
}}

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
Respond with ONLY a JSON object in this exact structure:
{{
  "final_text": "<the best merged transcription>",
  "decisions": [
    {{
      "segment": "<the text segment in question>",
      "chosen_version": <version number, 1-indexed>,
      "reason": "<why this version's reading is better>"
    }}
  ],
  "confidence": <0-100>,
  "uncertain_segments": [
    "<segments where no version was convincing>"
  ]
}}"""


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
        versions_text="\n\n".join(versions_text_parts)
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
