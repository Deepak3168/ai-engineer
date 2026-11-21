"""An explicit "final answer" tool: the run ends when the model *submits*, not when it merely stops calling tools.

Implicit finish (no tool calls = done) is the default in `common.loop`. Its weakness: a model that simply
runs out of things to say, or emits a polite refusal, looks identical to a finished investigation. An explicit
submit tool gives the harness a validated, typed end state (and a place to demand evidence).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from common.tools import ToolRegistry

FINAL_TOOL = "submit_answer"


class SubmitAnswerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=20, description="The complete answer for the on-call engineer.")
    evidence: list[str] = Field(
        min_length=1,
        description="Ids that support the answer, e.g. ['INC-1042', 'rb-deploy-rollback']. Required.",
    )
    confidence: Literal["low", "medium", "high"]


def add_final_tool(registry: ToolRegistry) -> ToolRegistry:
    @registry.tool(
        FINAL_TOOL,
        "Call exactly once when the investigation is complete, with the answer and the ids that support it. "
        "Do not write the answer as plain text instead.",
        SubmitAnswerArgs,
    )
    def submit_answer(answer: str, evidence: list[str], confidence: str) -> str:
        return f"Answer recorded with {len(evidence)} evidence id(s), confidence {confidence}."

    return registry
