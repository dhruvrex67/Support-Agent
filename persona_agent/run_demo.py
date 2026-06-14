"""
Scripted demo that runs a set of representative conversations through the
agent and prints a transcript. Useful for showing all five required
behaviors (persona detection, KB retrieval, tone adaptation, escalation,
context handoff) without needing to type anything interactively, and
without needing an API key.

Run:
    python -m persona_agent.run_demo
"""

from . import config
from .agent import PersonaAdaptiveAgent


def run_conversation(title: str, turns: list):
    print("=" * 70)
    print(title)
    print("=" * 70)
    agent = PersonaAdaptiveAgent()
    for turn in turns:
        print(f"\nUser: {turn}")
        result = agent.handle_message(turn)
        print(f"  [persona={result.persona_result.persona} "
              f"(confidence={result.persona_result.confidence}, "
              f"frustration={result.persona_result.frustration_score}/10, "
              f"method={result.persona_result.method})]")
        if result.kb_article:
            print(f"  [kb='{result.kb_article.title}' sim={result.kb_score:.2f}]")
        else:
            print(f"  [kb=none, best_sim={result.kb_score:.2f}]")
        if result.escalate:
            print(f"  [ESCALATE: {result.escalation_reason}]")
        print(f"Agent: {result.response}")
        if result.handoff_context:
            print()
            print(result.handoff_context)
    print()


def main():
    print(f"Mode: {'LLM-enhanced (Claude)' if config.USE_LLM else 'Heuristic-only (no API key set)'}\n")

    run_conversation(
        "Scenario 1: Technical expert -- API rate limiting",
        [
            "Hey, I'm integrating with your API and getting a 429 on the /sync "
            "endpoint after about 100 requests. What's the rate limit and is "
            "there a Retry-After header I should respect?",
        ],
    )

    run_conversation(
        "Scenario 2: Frustrated user -- high frustration triggers immediate escalation",
        [
            "I am SO FRUSTRATED right now!!! This is the THIRD time my sync has "
            "failed and it's RIDICULOUS and UNACCEPTABLE. I'm furious and ready "
            "to cancel my subscription.",
        ],
    )

    run_conversation(
        "Scenario 3: Business executive -- enterprise SLA question",
        [
            "We're a 200-seat team evaluating an enterprise contract. Before we "
            "sign, what SLA and uptime guarantees come with the enterprise plan?",
        ],
    )

    run_conversation(
        "Scenario 4: General user -- simple password reset",
        [
            "Hi, how do I reset my password? I don't remember setting one up "
            "with SSO or anything.",
        ],
    )

    run_conversation(
        "Scenario 5: Escalation after repeated unresolved turns",
        [
            "Can you tell me about your office locations?",
            "What's your favorite pizza topping and do you sponsor any sports teams?",
        ],
    )

    run_conversation(
        "Scenario 6: Explicit request for a human agent",
        [
            "I've tried everything in your docs already. Can I just talk to a human please?",
        ],
    )


if __name__ == "__main__":
    main()
