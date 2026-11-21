"""Day 1 demo. Runs offline (ScriptedLLM + a mock HTTP transport) or against a live endpoint.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day01.demo

Sections 1, 2 and 6 never touch a model. Sections 3-5 use `get_client(script)`: with
LLM_API_KEY set (and LESSON_OFFLINE unset) the *same code* calls your real endpoint and the
script is ignored - so outputs will then differ from the ones in the lesson.
"""

from __future__ import annotations

import json

from common.llm import (
    LLMResponse,
    Message,
    PermanentLLMError,
    RateLimitedError,
    ScriptedLLM,
    TransientLLMError,
    Usage,
    assistant,
    get_client,
    system,
    user,
)
from common.structured import StructuredOutputError
from common.tokens import count_message_tokens, estimate_cost_usd
from day01.budget import ContextBudgetError, guard_input
from day01.triage import IncidentTriage, native_response_format, triage_alert, triage_native
from day01.wire import capture_exchange, completion_body

ALERT = (
    "[FIRING:3] checkout-api error_rate=31% (threshold 5%) since 09:12Z. "
    "Release v2026.03.04-1 was deployed at 09:05Z. DB pool active=100/100. Runbook: rb-deploy-rollback"
)

GOOD = IncidentTriage(
    severity="SEV1", service="checkout-api", summary="Error rate 31% after release v2026.03.04-1; DB pool saturated.",
    suspected_cause="Bad deploy: DB connection pool exhaustion", needs_page=True, confidence=0.8,
)


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 66 - len(title)))


def section_wire() -> None:
    banner("1. What actually crosses the wire")
    history = [system("You are an on-call assistant."), user("What is SEV1?")]
    first = capture_exchange(history, completion_body("A customer-facing outage.", prompt_tokens=21, completion_tokens=6),
                             temperature=0.0, max_tokens=100)
    print("POST", first.url + "   (path is appended by the SDK)")
    print("request body :", json.dumps(first.request_body))
    print("response body:", json.dumps(first.response_body["choices"][0]["message"]), "| usage:", first.response_body["usage"])

    # Statelessness: the server remembers nothing, so turn 2 must resend turn 1.
    history += [assistant("A customer-facing outage."), user("And SEV3?")]
    second = capture_exchange(history, completion_body("Minor or internal.", prompt_tokens=39, completion_tokens=5))
    print(f"\nturn 1 sent {len(first.request_body['messages'])} messages, turn 2 sent {len(second.request_body['messages'])}:")
    for m in second.request_body["messages"]:
        print(f"   {m['role']:<9} {m['content']!r}")


def section_tokens() -> None:
    banner("2. Tokens and cost (prices below are PLACEHOLDERS, not a real price list)")
    msgs: list[Message] = [system("You are an on-call assistant."), user(ALERT)]
    n_in = count_message_tokens(msgs)
    print(f"input estimate: {n_in} tokens for {len(ALERT)} chars of alert text")
    cost = estimate_cost_usd(n_in, 120, usd_per_mtok_in=1.0, usd_per_mtok_out=3.0)
    print(f"cost of one call @ $1/$3 per Mtok: ${cost:.6f}")
    # Statelessness makes input cost grow quadratically with turns when history is resent.
    per_turn, total = 300, 0
    for turn in range(1, 11):
        total += per_turn * turn  # turn k resends k * per_turn tokens of history
    print(f"10 turns x {per_turn} tok/turn resent in full: {total} input tokens vs {per_turn * 10} if sent once")


def section_triage() -> None:
    banner("3. Portable structured output: validate and repair")
    bad_syntax = "Sure! Here is the triage:\n```json\n{\"severity\": \"SEV0\", \"service\": \"checkout-api\"}\n```"
    bad_semantic = GOOD.model_copy(update={"needs_page": False}).model_dump_json()
    client = get_client([bad_syntax, bad_semantic, GOOD.model_dump_json()])
    result = triage_alert(client, ALERT)
    print("validated:", result.triage.model_dump_json())
    print("usage:", result.usage.model_dump(), "| strategy:", result.strategy)
    if isinstance(client, ScriptedLLM):
        print(f"model calls: {len(client.calls)} (two repairs)")
        for i, call in enumerate(client.calls[1:], start=1):
            print(f"  repair prompt {i}:", call["messages"][-1].content.replace("\n", " | "))


def section_failures() -> None:
    banner("4. Failure: repairs exhausted")
    client = ScriptedLLM(["not json at all", "{broken", '{"severity": "SEV9"}'])
    try:
        triage_alert(client, ALERT, max_repairs=2)
    except StructuredOutputError as exc:
        print(f"StructuredOutputError after {len(exc.attempts)} attempts: {exc}")
        print("last attempt was:", exc.attempts[-1])
    banner("4b. Failure: truncated output in native mode")
    cut = LLMResponse(message=assistant('{"severity": "SEV1", "service": "checkout-a'), finish_reason="length", usage=Usage())
    try:
        triage_native(ScriptedLLM([cut]), ALERT)
    except PermanentLLMError as exc:
        print("PermanentLLMError:", exc)


def section_native() -> None:
    banner("5. Native JSON-schema mode (what changes in the request)")
    client = get_client([GOOD.model_dump_json()])
    result = triage_native(client, ALERT)
    print("validated:", result.triage.severity, result.triage.service, "| strategy:", result.strategy)
    if isinstance(client, ScriptedLLM):
        rf = client.calls[0]["response_format"]
        print("response_format sent:", rf["type"], "name=", rf["json_schema"]["name"], "strict=", rf["json_schema"]["strict"])
        print("schema required fields:", rf["json_schema"]["schema"]["required"])
        print("note: ScriptedLLM does not enforce the schema; a live provider would constrain decoding.")
    print("same schema via helper:", native_response_format(IncidentTriage)["json_schema"]["schema"]["additionalProperties"] is False)


def section_guard_and_errors() -> None:
    banner("6. Budget guard and error taxonomy")
    huge = ALERT + "\n" + ("stack frame line\n" * 3000)
    try:
        guard_input([system("x"), user(huge)], context_window=8192, max_output_tokens=400)
    except ContextBudgetError as exc:
        print(f"ContextBudgetError (no API call made): needs ~{exc.needed}, allowed {exc.allowed}")
    for exc in (RateLimitedError(retry_after_s=2.0), TransientLLMError("timeout"), PermanentLLMError("400 bad request")):
        action = "retry with backoff" if isinstance(exc, TransientLLMError) else "fail fast"
        print(f"  {type(exc).__name__:<20} -> {action}")


def main() -> None:
    section_wire()
    section_tokens()
    section_triage()
    section_failures()
    section_native()
    section_guard_and_errors()


if __name__ == "__main__":
    main()
