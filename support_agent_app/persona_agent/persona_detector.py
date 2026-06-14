import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config

TECHNICAL_KEYWORDS = [
    "api", "endpoint", "sdk", "token", "webhook", "json", "payload", "schema",
    "stack trace", "error code", "status code", "401", "429", "500", "timeout",
    "integration", "config", "configuration", "log", "logs", "request", "response",
    "header", "auth", "authentication", "rate limit", "sync", "database", "query",
]

FRUSTRATION_KEYWORDS = [
    "terrible", "awful", "worst", "ridiculous", "unacceptable", "furious",
    "angry", "frustrated", "annoyed", "fed up", "sick of", "again", "still not",
    "third time", "again and again", "useless", "waste of time", "disappointed",
    "cancel my", "refund", "this is broken", "doesn't work", "not working",
]

BUSINESS_KEYWORDS = [
    "roi", "contract", "renewal", "enterprise", "account manager", "sla",
    "stakeholder", "budget", "procurement", "invoice", "quarter", "pricing",
    "deal", "vendor", "compliance", "board", "leadership", "partnership",
    "scale", "rollout", "seats", "license",
]

EXCLAMATION_RE = re.compile(r"!{1,}")
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")


@dataclass
class PersonaResult:
    persona: str
    confidence: float
    frustration_score: int
    scores: Dict[str, float] = field(default_factory=dict)
    method: str = "heuristic"


def _count_hits(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _frustration_score(text: str) -> int:
    score = 0
    score += min(_count_hits(text, FRUSTRATION_KEYWORDS), 5)
    score += min(len(EXCLAMATION_RE.findall(text)), 3)
    score += min(len(ALL_CAPS_WORD_RE.findall(text)), 2)
    return min(score, 10)


def heuristic_detect(text: str, history: Optional[List[str]] = None) -> PersonaResult:
    combined = text
    if history:
        combined = " ".join(history[-2:]) + " " + text

    tech_score  = _count_hits(combined, TECHNICAL_KEYWORDS)
    biz_score   = _count_hits(combined, BUSINESS_KEYWORDS)
    frust_kw    = _count_hits(combined, FRUSTRATION_KEYWORDS)
    frust_total = _frustration_score(text)

    scores = {
        "technical_expert":   float(tech_score),
        "business_executive": float(biz_score),
        "frustrated_user":    float(frust_kw) + (frust_total >= config.FRUSTRATION_ESCALATION_THRESHOLD) * 2,
        "general_user":       0.5,
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_persona, top_score = ranked[0]
    total = sum(scores.values()) or 1.0

    return PersonaResult(
        persona=top_persona,
        confidence=round(top_score / total, 2),
        frustration_score=frust_total,
        scores=scores,
        method="heuristic",
    )


def _needs_tiebreak(result: PersonaResult) -> bool:
    vals = sorted(result.scores.values(), reverse=True)
    return len(vals) >= 2 and (vals[0] - vals[1]) <= config.PERSONA_TIE_MARGIN and result.confidence < 0.5


_LLM_SYSTEM = """Classify the user message into exactly one of:
technical_expert, frustrated_user, business_executive, general_user.
Reply with ONLY the label."""


def _llm_classify(text: str, client, model: str) -> Optional[str]:
    try:
        resp = client.messages.create(
            model=model, max_tokens=20,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        label = resp.content[0].text.strip().lower()
        return label if label in config.PERSONAS else None
    except Exception:
        return None


def detect_persona(text: str, conversation_history: Optional[List[str]] = None,
                   anthropic_client=None, model: str = config.MODEL_NAME) -> PersonaResult:
    result = heuristic_detect(text, conversation_history)

    if anthropic_client and _needs_tiebreak(result):
        refined = _llm_classify(text, anthropic_client, model)
        if refined and refined != result.persona:
            result.persona = refined
            result.method = "heuristic+llm"
            result.confidence = max(result.confidence, 0.6)

    return result
