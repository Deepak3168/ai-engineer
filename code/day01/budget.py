"""Pre-flight token budget guard.

A request that cannot fit in the context window fails at the provider with a
permanent 4xx error *after* you paid the network round trip (and sometimes after
you were billed for a slow rejection). Checking locally is cheap and gives a
better error. Counts are estimates (`common.tokens`), so we keep a safety margin.
"""

from __future__ import annotations

from collections.abc import Sequence

from common.llm import Message, PermanentLLMError
from common.tokens import count_message_tokens, count_tokens


class ContextBudgetError(PermanentLLMError):
    """The prompt (plus reserved output) does not fit the model's context window."""

    def __init__(self, needed: int, allowed: int):
        super().__init__(f"prompt needs ~{needed} tokens but only {allowed} are available for input")
        self.needed = needed
        self.allowed = allowed


def input_budget(context_window: int, max_output_tokens: int, safety_margin: float = 0.05) -> int:
    """Tokens available for the prompt: window - reserved output - safety margin."""
    if context_window <= 0 or max_output_tokens < 0:
        raise ValueError("context_window must be > 0 and max_output_tokens >= 0")
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be in [0, 1)")
    allowed = int(context_window * (1 - safety_margin)) - max_output_tokens
    if allowed <= 0:
        raise ValueError("max_output_tokens leaves no room for any input")
    return allowed


def guard_input(
    messages: Sequence[Message],
    *,
    context_window: int,
    max_output_tokens: int,
    safety_margin: float = 0.05,
) -> int:
    """Return the estimated input tokens, or raise ContextBudgetError if they do not fit."""
    allowed = input_budget(context_window, max_output_tokens, safety_margin)
    needed = count_message_tokens(messages)
    if needed > allowed:
        raise ContextBudgetError(needed, allowed)
    return needed


def truncate_to_tokens(text: str, max_tokens: int, marker: str = "\n[... truncated ...]\n") -> str:
    """Keep the head and tail of `text` within `max_tokens` (alerts put signal at both ends).

    Binary-searches on characters so it works with whichever counter `common.tokens` uses.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # largest kept-length whose head+marker+tail fits
        mid = (lo + hi + 1) // 2
        head, tail = text[: mid // 2], text[len(text) - (mid - mid // 2) :]
        if count_tokens(head + marker + tail) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    head, tail = text[: lo // 2], text[len(text) - (lo - lo // 2) :] if lo else ""
    return head + marker + tail
