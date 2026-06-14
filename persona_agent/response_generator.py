from typing import List, Optional
from .knowledge_base import KBArticle

# tone guide per persona - used by both template and LLM paths
PERSONA_STYLE = {
    "technical_expert": {
        "tone": "Direct and precise. No fluff. Use correct terminology and get straight to the fix.",
        "opener": "",
        "closer": "Let me know if you need the relevant log fields or further detail.",
    },
    "frustrated_user": {
        "tone": "Warm and empathetic. Acknowledge frustration first, then explain clearly in simple steps.",
        "opener": "I'm really sorry for the trouble this has caused — let's get this sorted out.",
        "closer": "I'll make sure this gets fully resolved for you.",
    },
    "business_executive": {
        "tone": "Concise. Lead with the outcome, skip step-by-step detail, mention SLAs or account manager where relevant.",
        "opener": "",
        "closer": "Happy to loop in your account manager if a more detailed plan is needed.",
    },
    "general_user": {
        "tone": "Friendly and clear. Plain language, full explanation but not overwhelming.",
        "opener": "Happy to help with that!",
        "closer": "Let me know if anything's unclear.",
    },
}

_BIZ_TERMS = ["sla", "uptime", "enterprise", "contract", "account manager", "pricing", "seats"]


def template_response(persona: str, article: Optional[KBArticle]) -> str:
    style = PERSONA_STYLE.get(persona, PERSONA_STYLE["general_user"])

    if article is None:
        body = "I couldn't find a precise answer to that in our knowledge base."
    elif persona == "technical_expert":
        body = article.content
    elif persona == "business_executive":
        sentences = [s.strip() for s in article.content.split(". ") if s.strip()]
        biz_sentences = [s for s in sentences if any(t in s.lower() for t in _BIZ_TERMS)]
        chosen = (biz_sentences or sentences)[:2]
        body = ". ".join(chosen).rstrip(".") + "."
    else:
        body = article.content

    parts = [p for p in [style["opener"], body, style["closer"]] if p]
    return "\n\n".join(parts)


_SYSTEM_TMPL = """You are a customer support agent.
User persona: {persona}
Tone: {tone}

Answer using only the information below. If it doesn't answer the question, say so.

KB excerpt:
\"\"\"
{kb}
\"\"\""""


def build_llm_messages(persona: str, article: Optional[KBArticle],
                        history: List[str], user_message: str):
    style = PERSONA_STYLE.get(persona, PERSONA_STYLE["general_user"])
    kb = article.content if article else "(no relevant article found)"
    system = _SYSTEM_TMPL.format(persona=persona, tone=style["tone"], kb=kb)

    messages = []
    for i, turn in enumerate(history):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": turn})
    messages.append({"role": "user", "content": user_message})

    return system, messages
