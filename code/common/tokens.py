"""Token counting with a graceful fallback.

`tiktoken` gives exact counts for OpenAI tokenizers but (a) downloads its BPE
files on first use and (b) is wrong for non-OpenAI models anyway. So we treat
counts as *estimates* used for budgeting, and always leave headroom (Day 7).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from common.llm import Message

# Per-message framing overhead (role markers etc.). Rough, provider-dependent.
MESSAGE_OVERHEAD_TOKENS = 4


@lru_cache(maxsize=8)
def _encoder(encoding_name: str):  # pragma: no cover - depends on network/optional dep
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Exact when tiktoken works offline/online, else ~4 chars per token."""
    if not text:
        return 0
    enc = _encoder(encoding_name)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, (len(text) + 3) // 4)


def count_message_tokens(messages: Sequence[Message], encoding_name: str = "cl100k_base") -> int:
    total = 0
    for m in messages:
        total += MESSAGE_OVERHEAD_TOKENS + count_tokens(m.content or "", encoding_name)
        for tc in m.tool_calls or []:
            total += count_tokens(tc.name, encoding_name) + count_tokens(
                tc.raw_arguments or str(tc.arguments), encoding_name
            )
    return total


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, usd_per_mtok_in: float, usd_per_mtok_out: float) -> float:
    """Prices change often; callers pass them in rather than us hardcoding a table."""
    return prompt_tokens / 1e6 * usd_per_mtok_in + completion_tokens / 1e6 * usd_per_mtok_out
