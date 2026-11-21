"""The Ops Report Agent's first tool set: three reads and one write.

Everything the model sees about a tool comes from three places, and all three are prompt text:
the tool *name*, its *description*, and the *schema* generated from the Pydantic args model
(field names, `Field(description=...)`, enums, bounds). Error strings are prompt text too.
"""

from __future__ import annotations

import re
import statistics
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from common.fixtures import INCIDENTS, RUNBOOKS, get_incident_record, list_incident_ids, metrics_for
from common.tools import ToolRegistry


class StrictArgs(BaseModel):
    """Reject unknown keys. A model that invents `severity_level` should learn that, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class GetIncidentArgs(StrictArgs):
    incident_id: str = Field(pattern=r"^INC-\d{4}$", description="Incident id, e.g. INC-1042.")


class SearchRunbooksArgs(StrictArgs):
    query: str = Field(min_length=2, max_length=200, description="Keywords describing the symptom, e.g. 'queue backlog'.")
    limit: int = Field(default=3, ge=1, le=5, description="Maximum number of runbooks to return.")


class QueryMetricsArgs(StrictArgs):
    service: str = Field(description="Service name exactly as in an incident record, e.g. checkout-api.")
    metric: Literal["error_rate", "p95_latency_ms", "cpu_pct"]
    start_minute: int = Field(default=0, ge=0, le=1380, description="First minute of the window (minutes from the data epoch).")
    minutes: int = Field(default=60, ge=1, le=120, description="Window length in minutes.")


class CreateTicketArgs(StrictArgs):
    title: str = Field(min_length=5, max_length=120)
    severity: Literal["SEV1", "SEV2", "SEV3"]
    idempotency_key: str = Field(
        min_length=8, max_length=64,
        description="Stable key for this logical ticket. Re-sending the same key returns the existing ticket.",
    )


class TicketStore:
    """In-memory ticket system. `executions` counts real writes, so tests can prove dedup works."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tickets: dict[str, dict[str, str]] = {}
        self._by_key: dict[str, str] = {}
        self.executions = 0

    def create(self, title: str, severity: str, key: str) -> tuple[dict[str, str], bool]:
        with self._lock:
            if key in self._by_key:
                return self.tickets[self._by_key[key]], False
            ticket_id = f"TCK-{len(self.tickets) + 1:03d}"
            ticket = {"id": ticket_id, "title": title, "severity": severity}
            self.tickets[ticket_id] = ticket
            self._by_key[key] = ticket_id
            self.executions += 1
            return ticket, True


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("a an and are as at be by do for from how i in is it its of on or the to we what when with".split())


def keyword_search(query: str, limit: int) -> list[dict[str, Any]]:
    """Naive on purpose: count distinct non-stopword query words appearing in title/tags/text. Day 8-10 replace it."""
    words = set(_WORD.findall(query.lower())) - _STOPWORDS
    scored = []
    for doc in RUNBOOKS:
        haystack = set(_WORD.findall(f"{doc['title']} {' '.join(doc['tags'])} {doc['text']}".lower()))
        score = len(words & haystack)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [
        {"id": d["id"], "title": d["title"], "score": s, "snippet": d["text"][:160].rstrip() + "..."}
        for s, d in scored[:limit]
    ]


def summarize_series(points: list[dict[str, float]], max_points: int = 6) -> dict[str, Any]:
    """Compact a time series: statistics the model can reason about plus a few evenly spaced samples.

    Returning 60 raw points costs tokens and invites the model to 'read' numbers it will misread.
    """
    values = [p["value"] for p in points]
    peak = max(points, key=lambda p: p["value"])
    step = max(1, len(points) // max_points)
    return {
        "points": len(points),
        "first_minute": int(points[0]["minute"]),
        "last_minute": int(points[-1]["minute"]),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 5),
        "peak_minute": int(peak["minute"]),
        "samples": [[int(p["minute"]), p["value"]] for p in points[::step]][:max_points],
    }


def build_registry(store: TicketStore | None = None, *, max_output_chars: int = 8_000) -> ToolRegistry:
    store = store if store is not None else TicketStore()
    reg = ToolRegistry(max_output_chars=max_output_chars)

    @reg.tool(
        "get_incident",
        "Fetch one incident record (title, severity, service, timeline, root cause if known) by id. "
        "Use when the user names an incident id. Does not search; if you lack an id, say so.",
        GetIncidentArgs,
    )
    def get_incident(incident_id: str) -> dict[str, Any]:
        try:
            return dict(get_incident_record(incident_id))
        except KeyError:
            raise ValueError(f"no incident {incident_id}. Known ids: {', '.join(list_incident_ids())}") from None

    @reg.tool(
        "search_runbooks",
        "Keyword search over the on-call runbooks. Returns ids, titles and a short snippet, best match first. "
        "Use to find procedures for a symptom or service. Returns an empty list when nothing matches.",
        SearchRunbooksArgs,
    )
    def search_runbooks(query: str, limit: int) -> list[dict[str, Any]]:
        return keyword_search(query, limit)

    @reg.tool(
        "query_metrics",
        "Summarise one per-minute metric for one service over a window: min, max, mean, peak minute and a few samples. "
        "Use to check whether a metric spiked. Known services: " + ", ".join(sorted({i["service"] for i in INCIDENTS})) + ".",
        QueryMetricsArgs,
    )
    def query_metrics(service: str, metric: str, start_minute: int, minutes: int) -> dict[str, Any]:
        return summarize_series(metrics_for(service, metric, start_minute, minutes))

    @reg.tool(
        "create_ticket",
        "WRITE: create a follow-up ticket in the tracker. Only call when the user asked for a ticket. "
        "Always pass an idempotency_key; retrying with the same key never creates a duplicate.",
        CreateTicketArgs,
        side_effect="write",
        idempotent=True,  # safe to retry *because* of the key, not because writes are safe
    )
    def create_ticket(title: str, severity: str, idempotency_key: str) -> dict[str, Any]:
        ticket, created = store.create(title, severity, idempotency_key)
        return {"ticket_id": ticket["id"], "created": created}

    return reg
