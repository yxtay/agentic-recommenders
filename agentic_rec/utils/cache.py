from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TLRUCache


class ResponseCache:
    def __init__(self, maxsize: int = 1000) -> None:
        # TLRU uses a ttu (time-to-use) function to determine expiration.
        # We store (value, expires_at) and return expires_at from ttu.
        # Use time.monotonic for consistency with TLRUCache's default timer.
        self.cache: TLRUCache[str, tuple[Any, float]] = TLRUCache(
            maxsize=maxsize, ttu=lambda _k, v, _now: v[1]
        )

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        item = self.cache.get(key)
        return item[0] if item else None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:  # noqa: ANN401
        import time

        expires_at = time.monotonic() + ttl_seconds
        self.cache[key] = (value, expires_at)


def generate_cache_key(
    namespace: str,
    *args: Any,
    **kwargs: Any,  # noqa: ANN401
) -> str:
    """Generate a unique cache key based on namespace and arguments."""
    key_parts = [namespace]

    for arg in args:
        if hasattr(arg, "model_dump_json"):
            key_parts.append(arg.model_dump_json())
        else:
            key_parts.append(json.dumps(arg, sort_keys=True))

    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(json.dumps(sorted_kwargs, default=str))

    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()
