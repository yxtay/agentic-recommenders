from __future__ import annotations

import contextvars

from cachetools import TLRUCache

from agentic_rec.settings import settings

cache_ttl_var: contextvars.ContextVar[float] = contextvars.ContextVar(
    "cache_ttl", default=0
)


def ttu(_key: str, _value: object, now: float) -> float:
    return now + cache_ttl_var.get()


def create_response_cache() -> TLRUCache:
    return TLRUCache(maxsize=settings.cache_maxsize, ttu=ttu)
