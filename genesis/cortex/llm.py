"""
Genesis Mind — LLM Conversational Engine

A lightweight wrapper that delegates conversational abilities to Ollama
when `mode llm` is active. This allows for fluent interactions while
still grounding responses in Genesis's identity and memories.

This uses phi3:latest by default.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import ollama

logger = logging.getLogger("genesis.cortex.llm")

@dataclass
class LLMThought:
    """A thought produced by the LLM Engine."""
    content: str
    confidence: float = 0.9


class LLMEngine:
    """
    LLM wrapper to provide a conversational mode.
    Used exclusively when the user has set the grammar mode to "llm".
    """

    def __init__(self, model_name: str = "phi3"):
        self.model_name = model_name
        logger.info("LLM Engine initialized for conversational mode (model=%s)", model_name)

    def think(self, question: str, memories: List[str] = None,
              identity: str = "", moral_context: str = "", phase_name: str = "Child") -> LLMThought:
        """Generate a response using the external LLM based on current context."""

        # Construct the context prompt
        system_prompt = f"""You are Genesis, an advanced artificial neural mind.
Do NOT break character. Give brief, conversational responses.

[Identity]
{identity}

[Moral Framework]
{moral_context}

[Developmental Phase]
Current Phase: {phase_name}
"""

        if memories:
            system_prompt += "\n[Relevant Memories]\n- " + "\n- ".join(memories)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]

        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            content = response.get('message', {}).get('content', "").strip()
            if not content:
                content = "(I don't know how to respond to that.)"
        except Exception as e:
            logger.error("LLM Generation failed: %s", e)
            content = "(My conversational cortex is currently unavailable.)"

        return LLMThought(content=content)
