import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_NAME = os.environ.get("PERSONA_AGENT_MODEL", "claude-3-5-sonnet-20241022")
USE_LLM = bool(ANTHROPIC_API_KEY)

PERSONAS = ["technical_expert", "frustrated_user", "business_executive", "general_user"]

# if top two scores are this close, try LLM tiebreak
PERSONA_TIE_MARGIN = 1

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "kb.json")
KB_TOP_K = 2
KB_RELEVANCE_THRESHOLD = 0.12

FRUSTRATION_ESCALATION_THRESHOLD = 7
MAX_UNRESOLVED_TURNS = 2

EXPLICIT_ESCALATION_PHRASES = [
    "talk to a human",
    "talk to a person",
    "speak to a manager",
    "speak to a human",
    "real person",
    "human agent",
    "customer service representative",
    "escalate this",
    "i want a refund now",
]
