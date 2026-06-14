"""
Interactive CLI demo for the Persona-Adaptive Customer Support Agent.

Usage:
    python -m persona_agent.cli

If ANTHROPIC_API_KEY is set in the environment, the agent will use Claude
for ambiguous persona tie-breaks and for generating final responses.
Otherwise it runs entirely on the built-in heuristics + templates --
no setup required.

Type 'quit' or 'exit' to end the session, or 'new' to start a fresh
conversation (resets persona/escalation state).
"""

import sys

from . import config
from .agent import PersonaAdaptiveAgent


BANNER = """\
==================================================================
 Persona-Adaptive Customer Support Agent -- CLI demo
==================================================================
Mode: {mode}
Type a support question. Try varying your tone/wording to see the
agent detect different personas:

  - technical_expert     e.g. "Getting a 429 on the /sync endpoint, what's the rate limit?"
  - frustrated_user       e.g. "This is the THIRD time my sync has failed!! It's unacceptable."
  - business_executive    e.g. "We're evaluating an enterprise contract -- what's the SLA?"
  - general_user          e.g. "How do I reset my password?"

Commands: 'new' (reset conversation), 'quit' / 'exit'.
==================================================================
"""


def main():
    mode = "LLM-enhanced (Claude)" if config.USE_LLM else "Heuristic-only (no API key found)"
    print(BANNER.format(mode=mode))

    agent = PersonaAdaptiveAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        if user_input.lower() == "new":
            agent = PersonaAdaptiveAgent()
            print("[Started a new conversation]\n")
            continue

        result = agent.handle_message(user_input)

        print(f"\n[persona: {result.persona_result.persona} "
              f"(confidence={result.persona_result.confidence}, "
              f"frustration={result.persona_result.frustration_score}/10, "
              f"method={result.persona_result.method})]")

        if result.kb_article:
            print(f"[kb match: '{result.kb_article.title}' (similarity={result.kb_score:.2f})]")
        else:
            print(f"[kb match: none above threshold (best similarity={result.kb_score:.2f})]")

        if result.escalate:
            print(f"[ESCALATING -- reason: {result.escalation_reason}]")

        print(f"\nAgent: {result.response}\n")

        if result.handoff_context:
            print(result.handoff_context)
            print()


if __name__ == "__main__":
    sys.exit(main())
