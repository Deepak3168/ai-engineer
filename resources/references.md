# References

Authoritative sources used across the series. Each lesson's own "References" section explains why a source is worth reading for that topic.

**Verification note.** Sources in the first section were fetched and checked while the lessons were written. Sources in the second section are well-known landing pages that could not be re-fetched from the authoring sandbox (its network allowlist blocks them), so confirm they still resolve. Provider documentation changes frequently; where a lesson makes a provider-specific statement it says so and points here instead of asserting current limits or prices.

## Verified while writing Days 1-3

### LLM APIs and structured outputs
- [OpenAI: Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI: Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [NVIDIA NIM API reference: LLM APIs](https://docs.api.nvidia.com/nim/reference/llm-apis)
- [Pydantic: JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [JSON Schema](https://json-schema.org/)
- [tiktoken](https://github.com/openai/tiktoken)

### Tool use and agent loops
- [Anthropic: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Anthropic: Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)
- [Schick et al.: Toolformer](https://arxiv.org/abs/2302.04761)
- [Yao et al.: ReAct](https://arxiv.org/abs/2210.03629)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [OWASP: Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

## Landing pages for later days (re-verify)

- [Model Context Protocol](https://modelcontextprotocol.io): specification and guides (Days 11-12). The Python SDK is 2.x, which renamed `FastMCP` to `MCPServer`; trust the installed package over older blog posts.
- [pgvector](https://github.com/pgvector/pgvector): vector types, HNSW/IVFFlat indexes, iterative scans (Days 8-10).
- [PostgreSQL: Full text search](https://www.postgresql.org/docs/16/textsearch.html) (Day 10).
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) (Day 14).
- [Liu et al.: Lost in the Middle](https://arxiv.org/abs/2307.03172) (Day 7).
- [Lewis et al.: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Day 9).
- [Greshake et al.: Indirect prompt injection](https://arxiv.org/abs/2302.12173) (Day 18).
- [OpenTelemetry: GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): still evolving (Day 21).
- [AWS Builders' Library: Timeouts, retries and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/) (Day 23).
- [Stripe: Idempotent requests](https://docs.stripe.com/api/idempotent_requests) (Day 23).
- [FastAPI](https://fastapi.tiangolo.com/) (Days 12, 24).
