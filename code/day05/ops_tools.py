"""Ops tools for the Day 5 runs, with an observable side-effect ledger.

`TicketLedger` counts how many times a write *actually executed*. Crash/resume tests assert
on that counter: a correct resume leaves it unchanged for already-completed calls.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from common.fixtures import get_incident_record, metrics_for
from common.tools import ToolRegistry


class TicketLedger:
    def __init__(self) -> None:
        self.created: list[dict[str, str]] = []
        self._by_key: dict[str, str] = {}

    def create(self, title: str, severity: str, key: str | None = None) -> str:
        if key is not None and key in self._by_key:
            return self._by_key[key]  # deduplicated: no second side effect
        ticket_id = f"TCK-{len(self.created) + 1:03d}"
        self.created.append({"id": ticket_id, "title": title, "severity": severity})
        if key is not None:
            self._by_key[key] = ticket_id
        return ticket_id


class IncidentArgs(BaseModel):
    incident_id: str = Field(description="Incident id such as INC-1042")


class MetricArgs(BaseModel):
    service: str
    metric: str = Field(description="error_rate | p95_latency_ms | cpu_pct")
    minutes: int = Field(default=10, ge=1, le=60)


class TicketArgs(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    severity: str = Field(pattern="^SEV[1-3]$")


class KeyedTicketArgs(TicketArgs):
    idempotency_key: str = Field(min_length=8, description="Same key => same ticket, never a duplicate")


def build_registry(ledger: TicketLedger) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("get_incident", "Fetch an incident record by id.", IncidentArgs)
    def get_incident(incident_id: str) -> dict:
        return dict(get_incident_record(incident_id))

    @reg.tool("query_metrics", "Return the last N per-minute values of a metric for a service.", MetricArgs)
    def query_metrics(service: str, metric: str, minutes: int) -> list[dict[str, float]]:
        return metrics_for(service, metric, 20, minutes)

    # Non-idempotent write: running it twice creates two tickets. The runner must never replay it blindly.
    @reg.tool("create_ticket", "Create a follow-up ticket (NOT idempotent).", TicketArgs, side_effect="write")
    def create_ticket(title: str, severity: str) -> str:
        return ledger.create(title, severity)

    # Idempotent write: safe to re-execute after a crash because the key deduplicates.
    @reg.tool("create_ticket_keyed", "Create a follow-up ticket; safe to retry with the same key.",
              KeyedTicketArgs, side_effect="write", idempotent=True)
    def create_ticket_keyed(title: str, severity: str, idempotency_key: str) -> str:
        return ledger.create(title, severity, key=idempotency_key)

    return reg
