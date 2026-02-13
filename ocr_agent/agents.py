"""
LLM agent definitions for the agentic OCR pipeline.
Each agent is an LLM call to Qwen3 via a specific system prompt with structured JSON output.
"""

from ocr_agent.tools import call_llm_json

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


def run_critic(transcription: str, previous_critique: dict | None = None) -> dict:
    """
    Run the Critic Agent on a transcription.
    Returns structured dict with confidence, segments, verdict, reasoning.
    """
    previous_section = ""
    if previous_critique:
        previous_section = (
            "## Previous Critique (for context — the text was edited since)\n"
            f"Previous confidence: {previous_critique.get('overall_confidence', 'N/A')}\n"
            f"Previous verdict: {previous_critique.get('verdict', 'N/A')}\n"
            f"Previous reasoning: {previous_critique.get('reasoning', 'N/A')}"
        )

    user_msg = CRITIC_USER_TEMPLATE.format(
        transcription=transcription,
        previous_critique_section=previous_section,
    )

    result = call_llm_json(CRITIC_SYSTEM_PROMPT, user_msg)

    # Ensure required fields exist with safe defaults
    result.setdefault("overall_confidence", 0)
    result.setdefault("segments", [])
    result.setdefault("verdict", "needs_editing")
    result.setdefault("reasoning", "")

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


def run_editor(transcription: str, critique: dict) -> dict:
    """
    Run the Editor Agent to fix issues identified by the Critic.
    Returns structured dict with corrected_text, changes, unresolved.
    """
    # Format issues for the editor
    issues_lines = []
    for seg in critique.get("segments", []):
        for issue in seg.get("issues", []):
            issues_lines.append(
                f"- [{issue.get('severity', '?')}] \"{seg.get('text', '')}\" → "
                f"{issue.get('description', '')} "
                f"(suggestion: {issue.get('suggestion', 'none')})"
            )

    if not issues_lines:
        issues_lines = ["No specific issues listed."]

    user_msg = EDITOR_USER_TEMPLATE.format(
        transcription=transcription,
        confidence=critique.get("overall_confidence", "N/A"),
        issues_text="\n".join(issues_lines),
    )

    result = call_llm_json(EDITOR_SYSTEM_PROMPT, user_msg)

    # Ensure required fields
    result.setdefault("corrected_text", transcription)
    result.setdefault("changes", [])
    result.setdefault("unresolved", [])

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


def run_arbitrator(versions: list[dict]) -> dict:
    """
    Run the Arbitrator Agent to merge multiple transcription versions.
    Each version dict should have: {text, source, score (optional)}
    Returns structured dict with final_text, decisions, confidence, uncertain_segments.
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

    result = call_llm_json(ARBITRATOR_SYSTEM_PROMPT, user_msg)

    # Ensure required fields
    result.setdefault("final_text", versions[0]["text"] if versions else "")
    result.setdefault("decisions", [])
    result.setdefault("confidence", 0)
    result.setdefault("uncertain_segments", [])

    return result
