from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .knowledge_base import KnowledgeBase, KBArticle
from .persona_detector import detect_persona, PersonaResult
from . import response_generator as rg

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class TurnResult:
    user_message: str
    persona_result: PersonaResult
    kb_article: Optional[KBArticle]
    kb_score: float
    escalate: bool
    escalation_reason: Optional[str]
    response: str
    handoff_context: Optional[str] = None


@dataclass
class ConversationState:
    history: List[str] = field(default_factory=list)
    unresolved_turns: int = 0
    escalated: bool = False


class PersonaAdaptiveAgent:
    def __init__(self, kb: Optional[KnowledgeBase] = None, use_llm: Optional[bool] = None):
        self.kb = kb or KnowledgeBase()
        self.use_llm = config.USE_LLM if use_llm is None else use_llm
        self.client = None
        if self.use_llm and anthropic:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.state = ConversationState()

    def _check_explicit_escalation(self, text: str) -> Optional[str]:
        lower = text.lower()
        for phrase in config.EXPLICIT_ESCALATION_PHRASES:
            if phrase in lower:
                return f"User requested a human agent (matched: '{phrase}')."
        return None

    def _decide_escalation(self, text: str, persona: PersonaResult, kb_score: float) -> Optional[str]:
        explicit = self._check_explicit_escalation(text)
        if explicit:
            return explicit

        if persona.frustration_score >= config.FRUSTRATION_ESCALATION_THRESHOLD:
            return f"Frustration score {persona.frustration_score}/10 hit the threshold."

        if kb_score < config.KB_RELEVANCE_THRESHOLD:
            self.state.unresolved_turns += 1
        else:
            self.state.unresolved_turns = 0

        if self.state.unresolved_turns >= config.MAX_UNRESOLVED_TURNS:
            return f"No KB match for {self.state.unresolved_turns} turns in a row."

        return None

    def _handoff_summary(self, persona: PersonaResult, article: Optional[KBArticle],
                          kb_score: float, reason: str) -> str:
        history_excerpt = "\n".join(self.state.history[-6:]) or "(no prior turns)"
        kb_note = (
            f"Closest KB article: '{article.title}' (sim={kb_score:.2f}, id={article.id})"
            if article else "No KB article matched."
        )
        return (
            "--- HANDOFF SUMMARY FOR HUMAN AGENT ---\n"
            f"Persona: {persona.persona} (confidence={persona.confidence}, method={persona.method})\n"
            f"Frustration: {persona.frustration_score}/10\n"
            f"Reason: {reason}\n"
            f"{kb_note}\n"
            f"Recent conversation:\n{history_excerpt}\n"
            "----------------------------------------"
        )

    def _escalation_message(self, persona: PersonaResult) -> str:
        if persona.persona == "frustrated_user":
            return (
                "I completely understand, and I don't want you to repeat yourself. "
                "I'm connecting you with a team member right now — they'll have the full context."
            )
        if persona.persona == "business_executive":
            return (
                "I'm escalating this to your account manager with full context so they can follow up directly."
            )
        return "I'm escalating this to a human support agent with the details of our conversation."

    def _generate_response(self, persona: PersonaResult, article: Optional[KBArticle], text: str) -> str:
        if self.client:
            system, messages = rg.build_llm_messages(persona.persona, article, self.state.history, text)
            try:
                resp = self.client.messages.create(
                    model=config.MODEL_NAME, max_tokens=400,
                    system=system, messages=messages,
                )
                return resp.content[0].text.strip()
            except Exception as e:
                return rg.template_response(persona.persona, article) + f"\n\n[LLM error: {e}]"
        return rg.template_response(persona.persona, article)

    def handle_message(self, user_message: str) -> TurnResult:
        persona = detect_persona(
            user_message,
            conversation_history=self.state.history,
            anthropic_client=self.client,
            model=config.MODEL_NAME,
        )

        article, kb_score = self.kb.best_match(user_message)
        if kb_score < config.KB_RELEVANCE_THRESHOLD:
            article = None

        escalation_reason = self._decide_escalation(user_message, persona, kb_score)
        escalate = bool(escalation_reason) and not self.state.escalated

        if escalate:
            self.state.escalated = True
            response = self._escalation_message(persona)
            handoff = self._handoff_summary(persona, article, kb_score, escalation_reason)
        else:
            response = self._generate_response(persona, article, user_message)
            handoff = None

        self.state.history.append(user_message)
        self.state.history.append(response)

        return TurnResult(
            user_message=user_message,
            persona_result=persona,
            kb_article=article,
            kb_score=kb_score,
            escalate=escalate,
            escalation_reason=escalation_reason,
            response=response,
            handoff_context=handoff,
        )
