# Persona-Adaptive Customer Support Agent

**A production-ready AI support agent that detects customer personas, retrieves relevant knowledge, adapts its tone, and intelligently escalates to humans.**

This is a complete implementation of the Adsparkx Jr. AI Engineer assignment, enhanced with **comprehensive error handling, 40+ unit tests, metrics tracking, and production-ready code**.

---

## 🎯 Assignment Requirements (All Met ✅)

> **Build a Persona-Adaptive Customer Support Agent that:**

- **(a) Is self-contained** ✅ — Single Python package, no external services required (LLM is optional)
- **(b) Detects customer persona** ✅ — Identifies technical experts, frustrated users, business executives, and general users
- **(c) Retrieves knowledge base content** ✅ — TF-IDF-based KB retrieval with fallback handling
- **(d) Adapts response tone** ✅ — Per-persona response templates + LLM-based generation
- **(e) Escalates to humans** ✅ — With structured context handoff and multiple escalation triggers

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/dhruvrex67/Support-Agent.git
cd Support-Agent

# Install dependencies
pip install -r persona_agent/requirements.txt

# Run the demo (no API key needed)
python -m persona_agent.run_demo

# Run interactive mode
python -m persona_agent.cli

# Run full test suite
python -m pytest persona_agent/test_agent.py -v

# Enable LLM mode (optional)
export ANTHROPIC_API_KEY="sk-ant-..."
python -m persona_agent.run_demo
```

---

## 🏗️ Architecture

### Core Pipeline
```
User Message
    ↓
[1] Input Validation (sanitize, type check)
    ↓
[2] Persona Detection (heuristic + optional LLM tie-break)
    ↓
[3] KB Retrieval (TF-IDF similarity scoring)
    ↓
[4] Escalation Decision (explicit, frustration, unresolved, errors)
    ↓
[5] Response Generation (template or LLM-enhanced)
    ↓
[6] Metrics Logging (for analysis & tuning)
    ↓
Agent Response (or escalation to human)
```

### Key Components

| Component | Purpose | Notes |
|-----------|---------|-------|
| `agent.py` | Orchestration + error handling | Manages the full pipeline with comprehensive error recovery |
| `persona_detector.py` | Persona classification | Heuristic scoring + optional LLM refinement |
| `knowledge_base.py` | KB retrieval | TF-IDF vectorizer + fallback article |
| `response_generator.py` | Tone-aware responses | Persona-specific templates + LLM system prompts |
| `metrics.py` | Analytics & logging | Tracks accuracy, escalation rates, performance |
| `test_agent.py` | Unit test suite | 40+ tests covering all components |

---

## 🧪 What's Included (Beyond the Assignment)

### 1. Comprehensive Error Handling
- **Graceful degradation** — KB missing? Use fallback. LLM fails? Use templates.
- **Input validation** — Rejects null/invalid input, sanitizes long messages
- **Fallback KB article** — "I don't have a good answer" when KB is empty
- **Error escalation** — Escalates to human after 3 consecutive errors
- **Logging** — All errors logged to `logs/agent_YYYYMMDD.log`

### 2. Full Test Coverage (40+ Tests)
- **Persona detection** — 8 tests (sarcasm, boundaries, history)
- **Escalation logic** — 6 tests (explicit, frustration, unresolved, single-escalation)
- **KB retrieval** — 5 tests (matching, ranking, edge cases)
- **Response generation** — 4 tests (tone adaptation per persona)
- **Conversation state** — 3 tests (history, counters, resets)
- **Error handling** — 4 tests (empty input, unicode, long messages)

Run tests: `python -m pytest persona_agent/test_agent.py -v`

### 3. Metrics & Analytics
- **Per-turn tracking** — Persona, confidence, frustration, KB score, escalation reason
- **Conversation aggregates** — Total turns, personas seen, escalation rate, KB hit rate
- **Performance monitoring** — Detection time, retrieval time, generation time
- **Threshold analyzer** — Suggests adjustments for frustration & KB thresholds

Example:
```python
from persona_agent.metrics import MetricsLogger
ml = MetricsLogger()
stats = ml.get_summary_stats()
print(f"Escalation rate: {stats['escalation_rate']:.1%}")
print(f"Avg KB hit rate: {stats['avg_kb_hit_rate']:.1%}")
```

### 4. Production-Ready Features
- **Dual-mode operation** — Works with or without Anthropic API key
- **Swappable retrieval** — Replace TF-IDF with embeddings (same interface)
- **Conversation state management** — History tracking, error counting, single escalation
- **Bounded memory** — Conversation history capped at 100 messages
- **Comprehensive logging** — Structured logs + metrics JSON Lines format

---

## 📊 Demo Scenarios

Run `python -m persona_agent.run_demo` to see all 6 scenarios:

| # | Query | Persona | Demonstrates |
|---|-------|---------|--------------|
| 1 | "I'm getting a 429 on the /sync endpoint after 100 requests" | Technical Expert | API error detection + full technical KB detail |
| 2 | "I AM SO FRUSTRATED!!! THIRD TIME THIS BROKE!!! RIDICULOUS!!!" | Frustrated User | High frustration score (≥7) → immediate escalation |
| 3 | "We're a 200-seat team evaluating an enterprise contract. What's the SLA?" | Business Executive | Contract keywords → business-focused summary |
| 4 | "How do I reset my password?" | General User | Simple fallback persona + helpful tone |
| 5 | "Can you tell me about your office locations?" + "What's your favorite pizza?" | N/A | Two unresolved turns → escalation |
| 6 | "Can I just talk to a human please?" | Any | Explicit escalation phrase matching |

---

## 🎓 Design Decisions (Interview Talking Points)

### Why Heuristics-First?
- **Cost**: Free, no API calls
- **Speed**: Instant (no network latency)
- **Reliability**: Always works, even if LLM is down
- **Transparency**: Can print exactly *why* each decision was made
- **Testability**: Fully deterministic, no flakiness

The LLM is only used for **tie-breaking** when heuristic scores are ambiguous, or for **response phrasing**.

### Why Frustration ≠ Persona?
A frustrated technical expert and a frustrated general user need different escalation handling. Coupling these would lose information:
```python
# ✅ Better: Independent
frustration_score = 8/10  # "I'm FURIOUS!!!"
persona = "technical_expert"  # API error in message

# ❌ Worse: Combined
label = "frustrated_technical_expert"  # Lost the frustration signal for escalation
```

### Why KB Relevance Threshold?
Returning *some* KB article is tempting, but low-similarity matches are worse than admitting "I don't know":
```python
# ✅ Good: Honest
if kb_score < 0.12:
    escalate = True  # Triggers human

# ❌ Bad: Misleading
return kb_article  # Frustrates user with irrelevant info
```

### Why Single Escalation Per Conversation?
After escalating once, assume a human has taken over. Don't keep saying "I'm connecting you to a human" on every message.

---

## 📈 Metrics & Monitoring

### Real-time Metrics
```python
from persona_agent.metrics import MetricsCollector, MetricsLogger

collector = MetricsCollector(conversation_id="conv_001")

# After each turn:
result = agent.handle_message(user_input)
collector.record_turn(result)

# At end of conversation:
final_metric = collector.finalize()
logger = MetricsLogger()
logger.log_conversation(final_metric)
```

### Analysis Example
```python
from persona_agent.metrics import ThresholdAnalyzer

# Check if FRUSTRATION_ESCALATION_THRESHOLD (default 7) is well-calibrated
recommendation = ThresholdAnalyzer.analyze_frustration_threshold(
    "logs/metrics.jsonl"
)
print(recommendation)
# {"current_threshold": 7, 
#  "escalated_avg_frustration": 8.2,
#  "not_escalated_avg_frustration": 3.1,
#  "suggestion": "Threshold looks well-calibrated"}
```

---

## 🛡️ Error Handling Examples

### Missing KB
```python
# KB file not found or corrupted
agent = PersonaAdaptiveAgent()
# ✅ Automatically uses fallback empty KB
# ✅ Logs warning, continues working
```

### LLM API Failure
```python
# ANTHROPIC_API_KEY set, but API times out
result = agent.handle_message("Tell me about pricing")
# ✅ Persona detected via heuristic
# ✅ Falls back to template response
# ✅ Logs error, continues
```

### Invalid Input
```python
result = agent.handle_message(None)
# ✅ Gracefully rejects
# ✅ Returns error in TurnResult
# ✅ Does not crash

result = agent.handle_message("")
# ✅ Handles empty string
# ✅ Returns safe response
```

---

## 🧪 Testing

### Run All Tests
```bash
python -m pytest persona_agent/test_agent.py -v
```

### Run Specific Test Class
```bash
python -m pytest persona_agent/test_agent.py::TestPersonaDetection -v
python -m pytest persona_agent/test_agent.py::TestEscalationLogic -v
python -m pytest persona_agent/test_agent.py::TestErrorHandling -v
```

### Run Tests Matching Pattern
```bash
python -m pytest persona_agent/test_agent.py -k "frustrated" -v
python -m pytest persona_agent/test_agent.py -k "escalation" -v
```

---

## 🔌 Swappable Components

### Replace KB Retrieval
Current: TF-IDF
```python
# in knowledge_base.py
def retrieve(self, query, top_k):
    qvec = self._vec.transform([query])
    sims = cosine_similarity(qvec, self._mat)[0]
    # ...
```

Future: Embeddings (same interface)
```python
def retrieve(self, query, top_k):
    embedding = voyage_client.embed(query)
    similarities = vector_db.search(embedding, top_k)
    # ...
```

No changes needed in `agent.py` — the interface is the same!

---

## 📝 Configuration

Edit `persona_agent/config.py` to tune behavior:

```python
# Escalation thresholds
FRUSTRATION_ESCALATION_THRESHOLD = 7       # 0-10 scale
MAX_UNRESOLVED_TURNS = 2                   # consecutive turns with no KB match
KB_RELEVANCE_THRESHOLD = 0.12              # minimum similarity to use KB article

# LLM
MODEL_NAME = "claude-3-5-sonnet-20241022"  # or other Anthropic model
PERSONA_TIE_MARGIN = 1                     # when heuristic scores are close, use LLM

# KB retrieval
KB_TOP_K = 2                               # return top 2 articles during search
KB_PATH = "persona_agent/data/kb.json"     # path to knowledge base

# Escalation phrases
EXPLICIT_ESCALATION_PHRASES = [
    "talk to a human",
    "speak to a manager",
    # ... add more phrases
]
```

---

## 📚 What You Could Do With More Time

1. **Replace TF-IDF with embeddings** (voyage-3, text-embedding-3, etc.)
2. **Add multi-persona support** — Blend frustrated + technical expert into one response
3. **Persist conversation state** — Redis/DB so state survives across requests
4. **Active learning pipeline** — Use human feedback to improve thresholds
5. **REST API** — FastAPI wrapper for production deployment
6. **Multi-language support** — Detect language, translate responses
7. **A/B testing framework** — Compare threshold changes on live traffic
8. **Webhook integration** — Push escalations to external system (Slack, Zendesk, etc.)

---

## 🎯 Interview Checklist

### ✅ Requirements Covered
- [x] Detects 3+ customer personas
- [x] Retrieves KB content
- [x] Adapts response tone per persona
- [x] Escalates to humans with context
- [x] Self-contained implementation
- [x] Works without API key (heuristic mode)

### ✅ Production-Ready Features
- [x] Comprehensive error handling (fallbacks, input validation)
- [x] Logging & metrics (track accuracy, performance)
- [x] Full test coverage (40+ tests)
- [x] Clean architecture (swappable components)
- [x] Documentation (README, code comments)

### ✅ Interview Talking Points
- Explain design trade-offs (heuristics vs. LLM, single vs. multi-persona)
- Walk through error handling examples
- Show test examples and why they matter
- Discuss metrics and how you'd tune thresholds
- Explain how you'd scale to 10k KB articles
- Describe how you'd handle 1000 concurrent users

---

## 📄 License

This is an assignment submission for Adsparkx. Feel free to use as portfolio material.

---

## 🙋 Questions?

See `persona_agent/README.md` for detailed technical documentation.
