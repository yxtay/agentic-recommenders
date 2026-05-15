from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TLRUCache


class ResponseCache:
    def __init__(self, maxsize: int = 1000) -> None:
        # ttu computes absolute expiry from cache's own timer.
        self._cache: TLRUCache[str, tuple[Any, int]] = TLRUCache(
            maxsize=maxsize, ttu=lambda _k, v, now: now + v[1]
        )

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        item = self._cache.get(key)
        return item[0] if item else None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:  # noqa: ANN401
        self._cache[key] = (value, ttl_seconds)

    @staticmethod
    def generate_key(
        namespace: str,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
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
