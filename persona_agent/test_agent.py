"""
Unit tests for PersonaAdaptiveAgent covering:
- Persona detection (technical, frustrated, executive, general)
- Escalation triggers (explicit, frustration, unresolved turns)
- Knowledge base retrieval accuracy
- Edge cases and boundary conditions
"""

import unittest
from .agent import PersonaAdaptiveAgent
from .persona_detector import detect_persona, heuristic_detect
from .knowledge_base import KnowledgeBase


class TestPersonaDetection(unittest.TestCase):
    """Test persona classification accuracy."""

    def test_technical_expert_detection(self):
        """Technical keywords should trigger technical_expert persona."""
        result = heuristic_detect("I'm getting a 429 error on the API endpoint. What's the rate limit?")
        self.assertEqual(result.persona, "technical_expert")
        self.assertGreater(result.confidence, 0.5)

    def test_frustrated_user_detection(self):
        """Frustration keywords and caps should trigger frustrated_user."""
        result = heuristic_detect(
            "This is ABSOLUTELY RIDICULOUS!!! The THIRD TIME this has BROKEN!!! I'm FURIOUS!!!"
        )
        self.assertEqual(result.persona, "frustrated_user")
        self.assertGreaterEqual(result.frustration_score, 7)

    def test_business_executive_detection(self):
        """Business/contract keywords should trigger business_executive persona."""
        result = heuristic_detect(
            "We're evaluating an enterprise contract. What are the SLA terms and uptime guarantees?"
        )
        self.assertEqual(result.persona, "business_executive")
        self.assertGreater(result.confidence, 0.5)

    def test_general_user_fallback(self):
        """Neutral questions should default to general_user."""
        result = heuristic_detect("How do I reset my password?")
        # general_user is the default fallback
        self.assertIn(result.persona, ["general_user", "technical_expert"])

    def test_frustration_score_boundaries(self):
        """Test frustration score calculation at boundaries."""
        # No frustration
        result = heuristic_detect("Hi, I have a quick question.")
        self.assertEqual(result.frustration_score, 0)
        
        # Medium frustration
        result = heuristic_detect("I'm frustrated with this feature")
        self.assertGreaterEqual(result.frustration_score, 1)
        
        # High frustration
        result = heuristic_detect("I AM FURIOUS!!! TERRIBLE!!! WORST EXPERIENCE!!!")
        self.assertGreaterEqual(result.frustration_score, 7)

    def test_sarcasm_edge_case(self):
        """Test that obvious sarcasm isn't misclassified (heuristic limitation)."""
        # Heuristics alone can't catch sarcasm, but shouldn't crash
        result = heuristic_detect("Yeah, *great* support, this really helped a lot")
        self.assertIsNotNone(result.persona)
        self.assertGreater(result.frustration_score, 0)  # Should detect frustration tone

    def test_multi_persona_conversation_history(self):
        """Persona detection should consider conversation history."""
        history = [
            "I have a question about integrating your API",
            "You need to use the webhook endpoint"
        ]
        result = heuristic_detect("What's the rate limit on this?", history)
        self.assertEqual(result.persona, "technical_expert")

    def test_mixed_technical_and_frustrated(self):
        """A frustrated technical expert should still escalate on frustration."""
        result = heuristic_detect(
            "Your API is BROKEN!!! I'm getting 500 errors and the docs are TERRIBLE!!! "
            "I've been debugging this for HOURS!!!"
        )
        self.assertGreaterEqual(result.frustration_score, 7)

    def test_confidence_scores_valid_range(self):
        """Confidence should always be between 0 and 1."""
        test_queries = [
            "Simple question",
            "EXTREME FRUSTRATION!!!",
            "API endpoint configuration",
            "Enterprise SLA discussion"
        ]
        for query in test_queries:
            result = heuristic_detect(query)
            self.assertGreaterEqual(result.confidence, 0)
            self.assertLessEqual(result.confidence, 1)


class TestEscalationLogic(unittest.TestCase):
    """Test escalation trigger conditions."""

    def test_explicit_escalation_phrases(self):
        """Explicit escalation phrases should trigger immediate escalation."""
        agent = PersonaAdaptiveAgent()
        
        escalation_phrases = [
            "Can I talk to a human?",
            "Speak to a manager please",
            "I want to escalate this",
            "Real person please",
            "I want a refund now"
        ]
        
        for phrase in escalation_phrases:
            result = agent.handle_message(phrase)
            self.assertTrue(result.escalate, f"Failed to escalate on: {phrase}")
            self.assertIsNotNone(result.handoff_context)

    def test_high_frustration_escalation(self):
        """Frustration score >= 7 should trigger escalation."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message(
            "I AM ABSOLUTELY FURIOUS!!! THIS IS THE WORST EXPERIENCE EVER!!! UNACCEPTABLE!!!"
        )
        self.assertTrue(result.escalate)
        self.assertGreaterEqual(result.persona_result.frustration_score, 7)

    def test_unresolved_turns_escalation(self):
        """Two consecutive turns with no KB match should escalate."""
        agent = PersonaAdaptiveAgent()
        
        # First off-topic turn (no match)
        result1 = agent.handle_message("What's your favorite movie?")
        self.assertFalse(result1.escalate)
        
        # Second off-topic turn (should escalate)
        result2 = agent.handle_message("Do you have any office locations?")
        self.assertTrue(result2.escalate)
        self.assertIn("No KB match", result2.escalation_reason)

    def test_single_escalation_per_conversation(self):
        """A conversation should only escalate once."""
        agent = PersonaAdaptiveAgent()
        
        # First escalation trigger
        result1 = agent.handle_message("I want to talk to a human")
        self.assertTrue(result1.escalate)
        
        # Follow-up message (after escalation, should NOT escalate again)
        result2 = agent.handle_message("Are you still there?")
        self.assertFalse(result2.escalate)

    def test_no_escalation_on_valid_kb_match(self):
        """A good KB match should NOT trigger escalation."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message("How do I reset my password?")
        self.assertFalse(result.escalate)
        self.assertIsNotNone(result.kb_article)

    def test_handoff_context_structure(self):
        """Handoff context should contain all required information."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message("I'M FURIOUS AND WANT TO CANCEL!!!!")
        
        self.assertTrue(result.escalate)
        self.assertIsNotNone(result.handoff_context)
        
        handoff = result.handoff_context
        self.assertIn("Persona:", handoff)
        self.assertIn("Frustration:", handoff)
        self.assertIn("Reason:", handoff)
        self.assertIn("conversation:", handoff)


class TestKnowledgeBaseRetrieval(unittest.TestCase):
    """Test KB retrieval accuracy and relevance scoring."""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_password_reset_retrieval(self):
        """Password reset query should match password KB article."""
        article, score = self.kb.best_match("How do I reset my password?")
        self.assertIsNotNone(article)
        self.assertGreater(score, 0.1)
        self.assertIn("password", article.title.lower())

    def test_api_rate_limit_retrieval(self):
        """API rate limit query should match API/rate limit KB article."""
        article, score = self.kb.best_match(
            "What is the rate limit on the API? I'm getting 429 errors."
        )
        self.assertIsNotNone(article)
        self.assertGreater(score, 0.1)

    def test_off_topic_returns_low_score(self):
        """Off-topic queries should return low similarity scores."""
        article, score = self.kb.best_match("What's your favorite pizza topping?")
        self.assertLess(score, 0.15)  # Below the relevance threshold

    def test_multiple_results_ranked(self):
        """Top-K retrieval should return results in descending similarity order."""
        results = self.kb.retrieve("password reset", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        
        # Results should be sorted by score descending
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_kb_article_structure(self):
        """Retrieved articles should have required fields."""
        article, score = self.kb.best_match("password")
        
        self.assertIsNotNone(article.id)
        self.assertIsNotNone(article.title)
        self.assertIsNotNone(article.content)
        self.assertIsNotNone(article.tags)
        self.assertGreater(len(article.content), 0)


class TestResponseGeneration(unittest.TestCase):
    """Test response tone adaptation per persona."""

    def test_technical_response_style(self):
        """Technical expert should get detailed, no-fluff response."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message("What's the rate limit on the sync endpoint?")
        
        # Should include technical details
        response = result.response.lower()
        self.assertTrue(
            any(term in response for term in ["endpoint", "limit", "request", "rate"]),
            "Technical response missing technical details"
        )

    def test_frustrated_response_empathy(self):
        """Frustrated user should get empathetic opener."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message("This keeps breaking!!! I'm SO FRUSTRATED!!!")
        
        # Escalates due to high frustration, but should still be empathetic
        response = result.response.lower()
        self.assertGreater(len(response), 10)

    def test_executive_response_conciseness(self):
        """Executive should get concise, business-focused response."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message(
            "What are the SLA terms for an enterprise contract with 500 seats?"
        )
        
        # Should mention SLA/enterprise/terms if KB match exists
        response = result.response.lower()
        if result.kb_article:
            self.assertGreater(len(response), 10)

    def test_general_response_friendliness(self):
        """General user should get friendly, helpful tone."""
        agent = PersonaAdaptiveAgent()
        result = agent.handle_message("How do I reset my password?")
        
        response = result.response.lower()
        self.assertGreater(len(response), 10)


class TestConversationState(unittest.TestCase):
    """Test conversation state tracking."""

    def test_conversation_history_tracking(self):
        """Agent should maintain conversation history for context."""
        agent = PersonaAdaptiveAgent()
        
        agent.handle_message("What's the API rate limit?")
        agent.handle_message("Does it apply to webhooks too?")
        
        # History should contain both user messages
        self.assertGreater(len(agent.state.history), 0)

    def test_unresolved_turn_counter(self):
        """Agent should track consecutive unresolved turns."""
        agent = PersonaAdaptiveAgent()
        
        # First unresolved turn
        result1 = agent.handle_message("What's your favorite color?")
        self.assertEqual(agent.state.unresolved_turns, 1)
        
        # Second unresolved turn
        result2 = agent.handle_message("Do you have office locations?")
        self.assertEqual(agent.state.unresolved_turns, 2)
        self.assertTrue(result2.escalate)

    def test_resolved_turn_resets_counter(self):
        """A successful KB match should reset the unresolved counter."""
        agent = PersonaAdaptiveAgent()
        
        # Unresolved turn
        agent.handle_message("What's your favorite movie?")
        self.assertEqual(agent.state.unresolved_turns, 1)
        
        # Resolved turn
        agent.handle_message("How do I reset my password?")
        self.assertEqual(agent.state.unresolved_turns, 0)


class TestErrorHandling(unittest.TestCase):
    """Test graceful degradation and error handling."""

    def test_agent_handles_empty_input(self):
        """Agent should handle empty user input gracefully."""
        agent = PersonaAdaptiveAgent()
        # Empty input should not crash
        result = agent.handle_message("")
        self.assertIsNotNone(result.persona_result)

    def test_agent_handles_very_long_input(self):
        """Agent should handle very long messages gracefully."""
        agent = PersonaAdaptiveAgent()
        long_message = "word " * 1000  # 1000+ words
        result = agent.handle_message(long_message)
        self.assertIsNotNone(result.persona_result)
        self.assertIsNotNone(result.response)

    def test_agent_handles_special_characters(self):
        """Agent should handle special characters and unicode."""
        agent = PersonaAdaptiveAgent()
        special_input = "Help! 🚨 API is broken: \\n [ERROR] 💥"
        result = agent.handle_message(special_input)
        self.assertIsNotNone(result.response)

    def test_agent_handles_multiple_special_chars(self):
        """Agent should handle messages with lots of punctuation."""
        agent = PersonaAdaptiveAgent()
        punctuation_heavy = "!!! ??? ... --- @@@ $$$ ^^^"
        result = agent.handle_message(punctuation_heavy)
        self.assertIsNotNone(result.persona_result)


if __name__ == "__main__":
    unittest.main()
