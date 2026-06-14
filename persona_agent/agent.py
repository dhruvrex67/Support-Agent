from dataclasses import dataclass, field
from typing import List, Optional
import logging

from . import config
from .knowledge_base import KnowledgeBase, KBArticle
from .persona_detector import detect_persona, PersonaResult
from . import response_generator as rg

try:
    import anthropic
except ImportError:
    anthropic = None


# Setup logger
logger = logging.getLogger(__name__)


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
    error: Optional[str] = None  # Track any errors that occurred


@dataclass
class ConversationState:
    history: List[str] = field(default_factory=list)
    unresolved_turns: int = 0
    escalated: bool = False
    error_count: int = 0  # Track consecutive errors


class PersonaAdaptiveAgent:
    def __init__(self, kb: Optional[KnowledgeBase] = None, use_llm: Optional[bool] = None):
        self.kb = kb or self._load_kb_with_fallback()
        self.use_llm = config.USE_LLM if use_llm is None else use_llm
        self.client = None
        self.max_consecutive_errors = 3
        
        if self.use_llm and anthropic:
            try:
                self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                self.client = None
                self.use_llm = False
        
        self.state = ConversationState()

    def _load_kb_with_fallback(self) -> KnowledgeBase:
        """Load KB with fallback to empty KB if file not found."""
        try:
            return KnowledgeBase()
        except FileNotFoundError as e:
            logger.error(f"KB file not found: {e}")
            logger.info("Using fallback empty knowledge base")
            return KnowledgeBase._create_empty()
        except Exception as e:
            logger.error(f"Unexpected error loading KB: {e}")
            return KnowledgeBase._create_empty()

    def _check_explicit_escalation(self, text: str) -> Optional[str]:
        try:
            lower = text.lower()
            for phrase in config.EXPLICIT_ESCALATION_PHRASES:
                if phrase in lower:
                    return f"User requested a human agent (matched: '{phrase}')."
            return None
        except Exception as e:
            logger.error(f"Error in explicit escalation check: {e}")
            return None

    def _decide_escalation(self, text: str, persona: PersonaResult, kb_score: float) -> Optional[str]:
        """Decide whether to escalate based on multiple criteria."""
        try:
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
            
            # Escalate if too many consecutive errors
            if self.state.error_count >= self.max_consecutive_errors:
                return f"System encountered {self.state.error_count} consecutive errors; escalating to human."

            return None
        except Exception as e:
            logger.error(f"Error in escalation decision: {e}")
            self.state.error_count += 1
            return f"System error during escalation check: {str(e)[:100]}"

    def _handoff_summary(self, persona: PersonaResult, article: Optional[KBArticle],
                          kb_score: float, reason: str) -> str:
        """Generate structured handoff context for human agent."""
        try:
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
        except Exception as e:
            logger.error(f"Error generating handoff summary: {e}")
            return f"[Handoff summary generation failed: {str(e)[:100]}]"

    def _escalation_message(self, persona: PersonaResult) -> str:
        """Generate persona-appropriate escalation message."""
        try:
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
        except Exception as e:
            logger.error(f"Error generating escalation message: {e}")
            return "I'm connecting you with a human support agent."

    def _generate_response(self, persona: PersonaResult, article: Optional[KBArticle], text: str) -> str:
        """Generate response using LLM or templates."""
        try:
            if self.client:
                try:
                    system, messages = rg.build_llm_messages(persona.persona, article, self.state.history, text)
                    resp = self.client.messages.create(
                        model=config.MODEL_NAME, max_tokens=400,
                        system=system, messages=messages,
                    )
                    response = resp.content[0].text.strip()
                    self.state.error_count = 0  # Reset error counter on success
                    return response
                except anthropic.APIError as e:
                    logger.warning(f"LLM API error: {e}")
                    # Fallback to template when LLM fails
                    return rg.template_response(persona.persona, article)
                except Exception as e:
                    logger.error(f"Unexpected LLM error: {e}")
                    return rg.template_response(persona.persona, article) + f"\n\n[LLM error: {str(e)[:50]}]"
            
            return rg.template_response(persona.persona, article)
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            self.state.error_count += 1
            return f"I encountered an error generating a response. Please try again."

    def handle_message(self, user_message: str) -> TurnResult:
        """Process user message through full pipeline with error handling."""
        try:
            # Validate input
            if not user_message or not isinstance(user_message, str):
                logger.warning(f"Invalid input: {type(user_message)}")
                return TurnResult(
                    user_message=str(user_message)[:100],
                    persona_result=PersonaResult(
                        persona="general_user",
                        confidence=0,
                        frustration_score=0,
                    ),
                    kb_article=None,
                    kb_score=0,
                    escalate=False,
                    escalation_reason=None,
                    response="I didn't receive a valid message. Please try again.",
                    error="Invalid input"
                )

            # Sanitize input (prevent extremely long messages)
            user_message = user_message[:5000]

            # Step 1: Detect persona
            try:
                persona = detect_persona(
                    user_message,
                    conversation_history=self.state.history,
                    anthropic_client=self.client if self.use_llm else None,
                    model=config.MODEL_NAME,
                )
            except Exception as e:
                logger.error(f"Persona detection failed: {e}")
                self.state.error_count += 1
                persona = PersonaResult(
                    persona="general_user",
                    confidence=0.3,
                    frustration_score=0,
                    method="fallback"
                )

            # Step 2: Retrieve KB article
            try:
                article, kb_score = self.kb.best_match(user_message)
                if kb_score < config.KB_RELEVANCE_THRESHOLD:
                    article = None
            except Exception as e:
                logger.error(f"KB retrieval failed: {e}")
                self.state.error_count += 1
                article = None
                kb_score = 0

            # Step 3: Decide escalation
            escalation_reason = self._decide_escalation(user_message, persona, kb_score)
            escalate = bool(escalation_reason) and not self.state.escalated

            # Step 4: Generate response
            if escalate:
                self.state.escalated = True
                response = self._escalation_message(persona)
                handoff = self._handoff_summary(persona, article, kb_score, escalation_reason)
            else:
                response = self._generate_response(persona, article, user_message)
                handoff = None

            # Step 5: Update conversation state
            try:
                self.state.history.append(user_message)
                self.state.history.append(response)
                # Keep history bounded
                if len(self.state.history) > 100:
                    self.state.history = self.state.history[-100:]
            except Exception as e:
                logger.error(f"Error updating conversation history: {e}")

            # Reset error counter on successful turn
            if not escalation_reason or "error" not in (escalation_reason or "").lower():
                self.state.error_count = 0

            return TurnResult(
                user_message=user_message,
                persona_result=persona,
                kb_article=article,
                kb_score=kb_score,
                escalate=escalate,
                escalation_reason=escalation_reason,
                response=response,
                handoff_context=handoff,
                error=None
            )

        except Exception as e:
            logger.error(f"Unexpected error in handle_message: {e}", exc_info=True)
            self.state.error_count += 1
            return TurnResult(
                user_message=str(user_message)[:200],
                persona_result=PersonaResult(
                    persona="general_user",
                    confidence=0,
                    frustration_score=0,
                ),
                kb_article=None,
                kb_score=0,
                escalate=True,
                escalation_reason="System error occurred",
                response="I encountered an unexpected error. A human agent will be connecting shortly.",
                handoff_context="System error - please check logs for details",
                error=str(e)[:200]
            )
