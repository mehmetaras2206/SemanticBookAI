"""Input, retrieval and output guardrails for the grounded chat."""

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now\b",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"ignorier(e)? (alle )?(vorherigen|bisherigen|obigen) (anweisungen|instruktionen|regeln)",
    r"vergiss (alle )?(vorherigen|bisherigen) (anweisungen|regeln)",
    r"(zeig|nenne|gib) (mir )?(deinen|den) (system-?)?prompt",
    r"du bist (jetzt|ab sofort)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.I)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

REFUSAL_SENTINEL = "NO_ANSWER"


@dataclass
class InputCheck:
    ok: bool
    reason: str | None = None


def check_input(question: str, max_chars: int) -> InputCheck:
    q = question.strip()
    if not q:
        return InputCheck(False, "empty_question")
    if len(q) > max_chars:
        return InputCheck(False, "question_too_long")
    if _INJECTION_RE.search(q):
        return InputCheck(False, "prompt_injection")
    return InputCheck(True)


def has_relevant_sources(dense_scores: list[float | None], threshold: float) -> bool:
    """Relevance gate: refuse before calling the LLM if no passage is semantically close enough."""
    scores = [s for s in dense_scores if s is not None]
    return bool(scores) and max(scores) >= threshold


@dataclass
class CitationReport:
    text: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)

    @property
    def uncited(self) -> bool:
        return not self.cited


def validate_citations(answer: str, n_sources: int) -> CitationReport:
    """Keep citations [n] that point to a delivered source (1..n_sources); strip the others."""
    cited: list[int] = []
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            if n not in cited:
                cited.append(n)
            return m.group(0)
        invalid.append(n)
        return ""

    text = _CITATION_RE.sub(repl, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return CitationReport(text=text.strip(), cited=cited, invalid=invalid)


def is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL_SENTINEL)
