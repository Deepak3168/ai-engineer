"""Executing a *batch* of tool calls from one assistant turn.

Rules enforced here (the registry handles single-call validation):
  * call ids must be unique - they are the only link between a result and its request;
  * a side-effect policy decides which classes of tool may run at all (default: reads only);
  * read-only calls run concurrently, anything with side effects runs sequentially, in model order;
  * results are returned in the order the model requested them, one per call, never dropped.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor

from common.llm import Message, ToolCall, tool_result
from common.tools import SideEffect, ToolRegistry, ToolResult


def _denied(call: ToolCall, side_effect: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error_type="denied",
        content=f"Error: tool '{call.name}' has side effect '{side_effect}' and is not allowed in this run. "
        "Do not retry; tell the user what you wanted to do instead.",
    )


def execute_batch(
    registry: ToolRegistry,
    calls: Sequence[ToolCall],
    *,
    allow: Collection[SideEffect] = ("read",),
    max_workers: int = 4,
) -> list[ToolResult]:
    ids = [c.id for c in calls]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate tool_call ids in one assistant turn: {ids}")

    results: list[ToolResult | None] = [None] * len(calls)
    parallel: list[int] = []
    serial: list[int] = []
    for i, call in enumerate(calls):
        tool = registry.get(call.name)
        if tool is None:  # unknown tool: let the registry produce its standard error
            parallel.append(i)
        elif tool.side_effect not in allow:
            results[i] = _denied(call, tool.side_effect)
        elif tool.side_effect == "read":
            parallel.append(i)
        else:
            serial.append(i)

    if parallel:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(parallel)))) as pool:
            for i, res in zip(parallel, pool.map(lambda idx: registry.execute(calls[idx]), parallel), strict=True):
                results[i] = res
    for i in serial:
        results[i] = registry.execute(calls[i])

    return [r for r in results if r is not None]


def results_to_messages(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> list[Message]:
    """One `role="tool"` message per call, same order, each carrying its call's id."""
    if len(calls) != len(results):
        raise ValueError("every tool call needs exactly one result")
    return [tool_result(c, r.content) for c, r in zip(calls, results, strict=True)]
