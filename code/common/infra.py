"""Helpers for tests/demos that need Postgres(+pgvector) or Redis.

Start both with `docker compose up -d` and export:
    DATABASE_URL=postgresql://agent:agent@localhost:5432/agents
    REDIS_URL=redis://localhost:6379/0

Rules that keep parallel test runs from interfering with each other:
  * Never FLUSHDB / DROP DATABASE. Create a uniquely named schema or key prefix
    per test (`unique_name`) and remove only that.
  * Tests skip (not fail) when the service is unreachable.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager


def unique_name(prefix: str = "t") -> str:
    """A short, SQL-identifier-safe, collision-free name (letters, digits, underscore)."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def database_url() -> str | None:
    return os.environ.get("DATABASE_URL")


def redis_url() -> str | None:
    return os.environ.get("REDIS_URL")


def postgres_available(require_pgvector: bool = False) -> bool:
    url = database_url()
    if not url:
        return False
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=2) as conn:
            if require_pgvector:
                row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
                return row is not None
            return True
    except Exception:
        return False


def redis_available() -> bool:
    url = redis_url()
    if not url:
        return False
    try:
        import redis

        return bool(redis.Redis.from_url(url, socket_connect_timeout=2).ping())
    except Exception:
        return False


@contextmanager
def temp_schema(require_pgvector: bool = False) -> Iterator[tuple[str, str]]:
    """Yield (database_url, schema_name) with an isolated schema, dropped on exit.

    Callers should run `SET search_path TO <schema>, public` on their connections
    (or schema-qualify table names). `public` stays on the path so the `vector`
    extension's types/operators resolve.
    """
    import psycopg

    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    schema = unique_name("lesson")
    with psycopg.connect(url, autocommit=True) as conn:
        if require_pgvector:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield url, schema
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
