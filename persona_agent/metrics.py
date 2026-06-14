"""
Metrics and logging module for the Persona-Adaptive Support Agent.

Tracks:
- Persona detection accuracy and confidence distribution
- Escalation rates and reasons
- KB retrieval performance (relevance scores, hit rates)
- Response handling times
- Conversation outcomes

Can export to JSON for analysis and threshold tuning.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class TurnMetric:
    """Metric recorded for each turn in a conversation."""
    timestamp: str
    user_message: str
    persona: str
    confidence: float
    frustration_score: int
    kb_matched: bool
    kb_score: float
    kb_title: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]
    response_length: int
    detection_method: str  # "heuristic" or "heuristic+llm"


@dataclass
class ConversationMetric:
    """Aggregate metrics for a complete conversation."""
    conversation_id: str
    started_at: str
    ended_at: str
    total_turns: int
    personas_seen: List[str]
    escalated: bool
    final_escalation_reason: Optional[str]
    avg_confidence: float
    max_frustration: int
    kb_hit_rate: float  # percentage of turns with KB matches
    turns: List[TurnMetric] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates metrics during conversation."""

    def __init__(self, conversation_id: str = None):
        self.conversation_id = conversation_id or self._generate_id()
        self.started_at = datetime.now().isoformat()
        self.turns: List[TurnMetric] = []
        self.personas_seen = set()
        self.escalated = False
        self.final_escalation_reason = None
        
        # For performance tracking
        self.detection_times: List[float] = []
        self.retrieval_times: List[float] = []

    def record_turn(self, turn_result):
        """Record metrics from a TurnResult."""
        metric = TurnMetric(
            timestamp=datetime.now().isoformat(),
            user_message=turn_result.user_message[:200],  # Truncate for storage
            persona=turn_result.persona_result.persona,
            confidence=turn_result.persona_result.confidence,
            frustration_score=turn_result.persona_result.frustration_score,
            kb_matched=turn_result.kb_article is not None,
            kb_score=turn_result.kb_score,
            kb_title=turn_result.kb_article.title if turn_result.kb_article else None,
            escalated=turn_result.escalate,
            escalation_reason=turn_result.escalation_reason,
            response_length=len(turn_result.response),
            detection_method=turn_result.persona_result.method,
        )
        self.turns.append(metric)
        self.personas_seen.add(turn_result.persona_result.persona)
        
        if turn_result.escalate:
            self.escalated = True
            self.final_escalation_reason = turn_result.escalation_reason

    def finalize(self) -> ConversationMetric:
        """Generate final conversation metric."""
        ended_at = datetime.now().isoformat()
        
        confidences = [t.confidence for t in self.turns]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        max_frustration = max(
            (t.frustration_score for t in self.turns),
            default=0
        )
        
        kb_hits = sum(1 for t in self.turns if t.kb_matched)
        kb_hit_rate = (kb_hits / len(self.turns)) if self.turns else 0
        
        return ConversationMetric(
            conversation_id=self.conversation_id,
            started_at=self.started_at,
            ended_at=ended_at,
            total_turns=len(self.turns),
            personas_seen=list(self.personas_seen),
            escalated=self.escalated,
            final_escalation_reason=self.final_escalation_reason,
            avg_confidence=round(avg_confidence, 3),
            max_frustration=max_frustration,
            kb_hit_rate=round(kb_hit_rate, 3),
            turns=self.turns,
        )

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique conversation ID."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]


class MetricsLogger:
    """Logs and persists metrics to file."""

    def __init__(self, log_dir: str = "persona_agent/logs", log_level=logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("PersonaAgent")
        self.logger.setLevel(log_level)
        
        # File handler for detailed logs
        log_file = self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics JSON file
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_conversation(self, metric: ConversationMetric):
        """Log conversation metrics to JSON Lines file."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metric), default=str) + '\n')
        
        self.logger.info(
            f"Conversation {metric.conversation_id}: "
            f"{metric.total_turns} turns, "
            f"escalated={metric.escalated}, "
            f"kb_hit_rate={metric.kb_hit_rate:.1%}"
        )

    def log_turn(self, metric: TurnMetric):
        """Log individual turn details."""
        self.logger.debug(
            f"Turn: persona={metric.persona} (confidence={metric.confidence}), "
            f"frustration={metric.frustration_score}, "
            f"kb_match={metric.kb_matched} (score={metric.kb_score:.3f})"
        )

    def get_summary_stats(self) -> Dict:
        """Analyze metrics file and return summary statistics."""
        if not self.metrics_file.exists():
            return {"error": "No metrics file found"}
        
        conversations = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not conversations:
            return {"error": "No valid metrics found"}
        
        total_conversations = len(conversations)
        escalation_rate = sum(1 for c in conversations if c['escalated']) / total_conversations
        
        all_personas = []
        for c in conversations:
            all_personas.extend(c['personas_seen'])
        
        avg_confidence = sum(c['avg_confidence'] for c in conversations) / total_conversations
        avg_kb_hit_rate = sum(c['kb_hit_rate'] for c in conversations) / total_conversations
        
        # Escalation reasons breakdown
        escalation_reasons = {}
        for c in conversations:
            if c['escalated'] and c['final_escalation_reason']:
                reason = c['final_escalation_reason'][:50]
                escalation_reasons[reason] = escalation_reasons.get(reason, 0) + 1
        
        return {
            "total_conversations": total_conversations,
            "escalation_rate": round(escalation_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "avg_kb_hit_rate": round(avg_kb_hit_rate, 3),
            "personas_distribution": self._count_personas(all_personas),
            "escalation_reasons": escalation_reasons,
            "sample_conversations": conversations[:3],  # Recent samples
        }

    @staticmethod
    def _count_personas(personas: List[str]) -> Dict[str, int]:
        """Count persona occurrences."""
        counts = {}
        for p in personas:
            counts[p] = counts.get(p, 0) + 1
        return counts


class PerformanceMonitor:
    """Monitors response times and system performance."""

    def __init__(self):
        self.detection_times: List[float] = []
        self.retrieval_times: List[float] = []
        self.generation_times: List[float] = []

    def record_detection_time(self, duration_ms: float):
        """Record persona detection duration."""
        self.detection_times.append(duration_ms)

    def record_retrieval_time(self, duration_ms: float):
        """Record KB retrieval duration."""
        self.retrieval_times.append(duration_ms)

    def record_generation_time(self, duration_ms: float):
        """Record response generation duration."""
        self.generation_times.append(duration_ms)

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        def stats(times):
            if not times:
                return {"count": 0, "avg_ms": 0, "max_ms": 0, "min_ms": 0}
            return {
                "count": len(times),
                "avg_ms": round(sum(times) / len(times), 2),
                "max_ms": round(max(times), 2),
                "min_ms": round(min(times), 2),
            }
        
        return {
            "persona_detection": stats(self.detection_times),
            "kb_retrieval": stats(self.retrieval_times),
            "response_generation": stats(self.generation_times),
        }


class ThresholdAnalyzer:
    """Analyzes metrics to suggest threshold adjustments."""

    @staticmethod
    def analyze_frustration_threshold(metrics_file: str, current_threshold: int = 7):
        """Analyze if FRUSTRATION_ESCALATION_THRESHOLD should be adjusted."""
        conversations = []
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    conversations.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "Cannot read metrics"}
        
        if not conversations:
            return {"error": "No data"}
        
        # Frustration scores for escalated vs not escalated
        escalated_frustrations = []
        not_escalated_frustrations = []
        
        for conv in conversations:
            max_frust = conv.get('max_frustration', 0)
            if conv.get('escalated'):
                escalated_frustrations.append(max_frust)
            else:
                not_escalated_frustrations.append(max_frust)
        
        recommendation = {
            "current_threshold": current_threshold,
            "escalated_avg_frustration": round(sum(escalated_frustrations) / len(escalated_frustrations), 1) if escalated_frustrations else 0,
            "not_escalated_avg_frustration": round(sum(not_escalated_frustrations) / len(not_escalated_frustrations), 1) if not_escalated_frustrations else 0,
        }
        
        # If escalated conversations have much higher frustration, threshold seems good
        if escalated_frustrations and not_escalated_frustrations:
            gap = recommendation["escalated_avg_frustration"] - recommendation["not_escalated_avg_frustration"]
            if gap < 2:
                recommendation["suggestion"] = "Consider lowering threshold (more escalations expected)"
            elif gap > 3:
                recommendation["suggestion"] = "Threshold may be too aggressive (escalating borderline cases)"
            else:
                recommendation["suggestion"] = "Threshold looks well-calibrated"
        
        return recommendation

    @staticmethod
    def analyze_kb_threshold(metrics_file: str, current_threshold: float = 0.12):
        """Analyze if KB_RELEVANCE_THRESHOLD should be adjusted."""
        conversations = []
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    conversations.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "Cannot read metrics"}
        
        hit_rates = [c.get('kb_hit_rate', 0) for c in conversations if c.get('turns')]
        
        if not hit_rates:
            return {"error": "No turn data"}
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        return {
            "current_threshold": current_threshold,
            "avg_kb_hit_rate": round(avg_hit_rate, 3),
            "suggestion": (
                "Threshold too high (lower hit rate)" if avg_hit_rate < 0.5 else
                "Threshold too low (high false positives)" if avg_hit_rate > 0.8 else
                "Threshold looks balanced"
            ),
        }
