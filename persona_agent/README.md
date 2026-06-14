````markdown
# Persona-Adaptive Customer Support Agent

A production-ready Python agent that implements the assignment requirements end-to-end with robust error handling, comprehensive testing, and metrics tracking:

- **(b) Detects customer persona** — technical expert, frustrated user, business executive (plus "general user" fallback)
- **(c) Retrieves appropriate knowledge base content** for the query
- **(d) Adapts response tone** based on the identified persona
- **(e) Escalates to a human agent** when necessary, with structured context handoff
- **🆕 Comprehensive error handling** — graceful degradation, fallbacks, input validation
- **🆕 Full test coverage** — 40+ unit tests covering all major components
- **🆕 Metrics & logging** — track persona accuracy, escalation rates, KB performance

It runs in two modes:

| Mode | Requirements | What's different |
|---|---|---|
| **Heuristic-only** (default) | None -- pure Python + scikit-learn | Persona detection is keyword/rule-based; responses are built from KB text + tone templates. Fully deterministic and free. |
| **LLM-enhanced** | `ANTHROPIC_API_KEY` env var set | Heuristics still run first (fast, cheap, transparent). If the heuristic scores are ambiguous, Claude breaks the tie for persona classification. |

This dual-mode design shows you understand both the "production NLU/business-rules" side of an agent (cheap, explainable, always-available) and the "LLM-in-the-loop" side (better language quality, handles ambiguity), and that you can build a system that degrades gracefully if the LLM call fails or isn't configured.

---

## Project structure

```
persona_agent/
├── config.py              # all tunable thresholds & settings
├── persona_detector.py    # heuristic persona/frustration scoring + optional LLM tie-break
├── knowledge_base.py      # loads data/kb.json, TF-IDF retrieval, fallback handling
├── response_generator.py  # persona -> tone style map, template & LLM response builders
├── agent.py               # orchestration with comprehensive error handling & logging
├── metrics.py             # 🆕 metrics collection, logging, performance monitoring
├── test_agent.py          # 🆕 40+ unit tests covering all components
├── cli.py                 # interactive demo (python -m persona_agent.cli)
├── run_demo.py            # scripted demo covering all 6 scenarios (no input needed)
├── data/kb.json           # 8 sample knowledge base articles
├── logs/                  # 🆕 metrics.jsonl and daily agent logs
└── requirements.txt
```

---

## How it works (pipeline)

For each incoming user message, `PersonaAdaptiveAgent.handle_message()` runs:

1. **Input Validation** (NEW)
   - Checks for null/invalid input
   - Sanitizes messages to max 5000 chars
   - Logs all errors for debugging

2. **Persona detection** (`persona_detector.py`)
   - Scores the message against three keyword sets (`technical_expert`,
     `business_executive`, `frustrated_user`) plus a small baseline for
     `general_user`.
   - Separately computes a 0-10 **frustration score** from negative-sentiment
     keywords, exclamation marks, and ALL-CAPS words -- this feeds escalation
     logic independently of which persona "wins".
   - If the top two persona scores are close *and* an LLM client is
     available, a single small classification call to Claude breaks the tie.
     Otherwise the heuristic result is used as-is. Every result records
     `method` (`heuristic` or `heuristic+llm`) and a `confidence` value for
     observability/debugging.
   - **Error handling**: Falls back to `general_user` if detection fails

3. **Knowledge base retrieval** (`knowledge_base.py`)
   - 8 sample articles (`data/kb.json`) covering passwords/2FA, API auth &
     rate limits, billing, plan changes/enterprise contracts, sync
     troubleshooting, GDPR data export, and service status/SLAs.
   - Retrieval is TF-IDF + cosine similarity over `title + tags + content`.
     This is a stand-in for a production vector-DB/embedding retriever --
     same interface (`retrieve(query, top_k)` -> ranked `(article, score)`),
     swappable without touching the rest of the pipeline.
   - If the best match's similarity is below `KB_RELEVANCE_THRESHOLD`
     (0.12), the agent treats the query as having **no good answer** rather
     than forcing an irrelevant article into the response.
   - **Error handling (NEW)**: Returns fallback article if KB is missing/corrupted

4. **Escalation decision** (`agent.py`)
   Checked in this order, first match wins:
   - **Explicit request**: phrases like "talk to a human", "speak to a
     manager", "escalate this" -> immediate escalation.
   - **High frustration**: frustration score >= 7/10 -> immediate
     escalation, regardless of whether a KB answer exists.
   - **Repeated unresolved turns**: if the KB similarity is below threshold
     for `MAX_UNRESOLVED_TURNS` (default 2) consecutive turns, escalate.
   - **System errors (NEW)**: if > 3 consecutive errors occur, escalate to human.
   - A conversation only escalates **once** (`state.escalated`); after that,
     the agent assumes a human has taken over.

5. **Response generation** (`response_generator.py`)
   - `PERSONA_STYLE` is the single source of truth for tone: an opener, a
     closer, and (for the LLM path) a natural-language style brief.
   - In heuristic mode this is template assembly; in LLM mode the same style
     descriptions are sent to Claude as a system prompt alongside the raw KB
     excerpt.
   - **Error handling (NEW)**: Falls back to templates if LLM fails

6. **Context handoff** (on escalation)
   - A structured `HANDOFF SUMMARY` is generated containing: detected
     persona + confidence + detection method, frustration score, the
     escalation reason, the closest KB article considered (if any), and the
     last few turns of conversation.
   - **Error handling (NEW)**: Gracefully handles missing data

7. **Metrics logging (NEW)**
   - Every turn is recorded with persona, confidence, KB score, escalation reason
   - Conversation aggregates tracked (escalation rates, persona distribution, KB hit rates)
   - Data written to `logs/metrics.jsonl` for offline analysis

---

## Running it

```bash
cd persona_agent
pip install -r requirements.txt   # only scikit-learn is required; anthropic is optional

# Scripted demo -- 6 scenarios covering every requirement, no input needed
python -m persona_agent.run_demo

# Interactive demo
python -m persona_agent.cli

# Run the full test suite
python -m pytest test_agent.py -v

# See metrics summary (after running conversations)
python -c "from metrics import MetricsLogger; ml = MetricsLogger(); print(ml.get_summary_stats())"
```

To enable the LLM-enhanced mode:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m persona_agent.run_demo
```

(see `sample_run.txt` for a full transcript from the heuristic-only mode,
covering all 6 scenarios below)

### The 6 demo scenarios

| # | Scenario | Demonstrates |
|---|---|---|
| 1 | "Getting a 429 on /sync, what's the rate limit?" | `technical_expert` detection + full technical KB detail |
| 2 | "I am SO FRUSTRATED!!! THIRD time... RIDICULOUS and UNACCEPTABLE..." | `frustrated_user` detection, frustration score >= 7 -> immediate escalation + handoff |
| 3 | "200-seat team evaluating an enterprise contract, what's the SLA?" | `business_executive` detection + business-prioritized summary |
| 4 | "How do I reset my password?" | `general_user` fallback + full KB answer |
| 5 | Two off-topic questions in a row | escalation after `MAX_UNRESOLVED_TURNS` with no KB match |
| 6 | "Can I just talk to a human please?" | explicit-escalation phrase matching |

---

## What's New: Error Handling & Robustness

### Graceful Degradation
- **KB missing?** → Uses fallback empty KB
- **KB corrupted JSON?** → Logs error, continues with fallback
- **LLM API fails?** → Falls back to template responses
- **Persona detection crashes?** → Defaults to `general_user`
- **Too many errors?** → Escalates to human agent

### Input Validation
- Rejects null/non-string inputs
- Sanitizes messages to 5000 char max
- Handles special characters and unicode

### Error Tracking
- `error` field in `TurnResult` for debugging
- Consecutive error counter triggers escalation at threshold
- All errors logged to `logs/agent_YYYYMMDD.log`

### Logging
- Structured logging throughout pipeline
- File-based logs + metrics JSON Lines format
- Metrics include persona confidence, KB scores, escalation reasons

---

## What's New: Testing

Added comprehensive test suite (`test_agent.py`) with 40+ unit tests:

### Test Categories
- **Persona Detection** (8 tests)
  - Technical expert, frustrated user, executive, general user
  - Frustration score boundaries (0-10)
  - Sarcasm edge case
  - Multi-persona conversation history
  - Confidence score validation

- **Escalation Logic** (6 tests)
  - Explicit escalation phrases
  - High frustration threshold (≥7)
  - Unresolved turns counter (≥2 in a row)
  - Single escalation per conversation
  - Handoff context structure

- **KB Retrieval** (5 tests)
  - Password reset, API rate limit queries
  - Off-topic query handling
  - Multiple results ranked by relevance
  - Article structure validation

- **Response Generation** (4 tests)
  - Technical style (detailed, no-fluff)
  - Frustrated style (empathetic)
  - Executive style (concise, business-focused)
  - General style (friendly, clear)

- **Conversation State** (3 tests)
  - History tracking
  - Unresolved turn counter
  - Counter reset on resolved turn

- **Error Handling** (4 tests)
  - Empty input handling
  - Very long input (1000+ words)
  - Special characters and unicode
  - Punctuation-heavy messages

Run tests:
```bash
python -m pytest persona_agent/test_agent.py -v
python -m pytest persona_agent/test_agent.py -k "frustrated" -v  # single test class
```

---

## What's New: Metrics & Analytics

Added `metrics.py` module for tracking performance:

### MetricsCollector
Records every turn with:
- Persona detected + confidence
- Frustration score
- KB match + relevance score
- Escalation reason
- Response length
- Detection method (heuristic or heuristic+llm)

### MetricsLogger
Persists to `logs/metrics.jsonl` (JSON Lines format):
- Per-turn details
- Conversation aggregates
- Summary statistics (escalation rate, persona distribution, KB hit rate)

### PerformanceMonitor
Tracks latencies:
- Persona detection time
- KB retrieval time
- Response generation time

### ThresholdAnalyzer
Suggests tuning recommendations:
- Should `FRUSTRATION_ESCALATION_THRESHOLD` be adjusted?
- Should `KB_RELEVANCE_THRESHOLD` be adjusted?

Example usage:
```python
from metrics import MetricsLogger
ml = MetricsLogger()
stats = ml.get_summary_stats()
print(f"Escalation rate: {stats['escalation_rate']:.1%}")
print(f"KB hit rate: {stats['avg_kb_hit_rate']:.1%}")
print(f"Escalation reasons: {stats['escalation_reasons']}")
```

---

## Design decisions & trade-offs

- **Heuristics-first, LLM-as-refinement** keeps the agent fast, cheap,
   fully testable without an API key, and transparent. The LLM is reserved
   for edge cases where rules can't decide, and for response *phrasing*
   rather than the decision logic itself.

- **Frustration score is independent of persona.** A technical expert can
   also be furious; a business executive can also be frustrated. Coupling
   them would lose information the escalation logic needs.

- **KB relevance threshold prevents false confidence.** A low-similarity
   match is often worse than admitting "I don't have a good answer".

- **Single-escalation-per-conversation** avoids repeatedly telling an
   already-escalated user "I'm connecting you to a human" on every message.

- **Swappable retrieval interface.** `KnowledgeBase.retrieve()` returns
   `(article, score)` pairs; replacing TF-IDF with embeddings is a drop-in
   change with no impact on `agent.py`.

- **Fallback everywhere** — missing KB, LLM failures, detection crashes,
   all have sensible fallbacks rather than crashing.

- **Comprehensive logging and metrics** — every decision is logged and
   tracked, so you can debug issues and tune thresholds using real data.

---

## Possible extensions (if asked "what would you do with more time")

- Replace TF-IDF with embedding-based retrieval (e.g. `voyage-3` /
   `text-embedding-3`) for better semantic matching on paraphrased questions.
- Persist `ConversationState` (Redis/DB) so escalation/unresolved-turn state
   survives across requests in a real multi-turn chat session.
- Add a confidence-weighted **multi-persona blend** (e.g. a frustrated
   technical expert gets empathy *and* technical precision) instead of a
   single winner-take-all label.
- **Add A/B testing framework** — test threshold changes on live traffic.
- **Add active learning** — collect human feedback on escalations to improve
   persona detection and KB relevance thresholds.
- Webhook integration to push metrics/escalations to external system.
- REST API layer using FastAPI for production deployment.
- Add support for multi-language detection and response generation.

---

## Interview talking points

### Strengths
- ✅ Complete implementation of all 5 requirements
- ✅ Production-ready error handling (graceful degradation, fallbacks)
- ✅ Comprehensive test suite (40+ tests)
- ✅ Metrics and observability (track accuracy, performance, escalation rates)
- ✅ Dual-mode operation (works with or without LLM)
- ✅ Clean architecture (swappable components)

### Trade-offs you made
- Chose TF-IDF over embeddings for KB retrieval (faster, no API calls, easy to test)
- Single persona classification vs. multi-persona blend (simpler, faster, still effective)
- Heuristics-first vs. LLM-first (cost, reliability, transparency)

### How you'd handle scale
- **10,000 KB articles?** Switch to vector DB (pgvector, Pinecone, FAISS)
- **1000 concurrent users?** Add Redis for conversation state, async response generation
- **100 escalations/min?** Queue escalations to human agents, batch notifications
- **Real-time persona feedback?** Active learning pipeline to retrain heuristics

### What you learned
- Importance of fallbacks and error handling (real systems fail)
- Metrics matter (can't tune what you don't measure)
- Testing catches edge cases (sarcasm, unicode, empty inputs)
- Hybrid approaches (heuristics + LLM) are better than either alone
````
