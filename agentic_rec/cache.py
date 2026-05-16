from __future__ import annotations

import contextvars
import functools
import hashlib
import time
from typing import TYPE_CHECKING

from cachetools import TLRUCache
from loguru import logger

from agentic_rec.settings import settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agentic_rec.models import RecommendRequest, RecommendResponse

    type RecommendMethod[T] = Callable[
        [T, str, RecommendRequest], Awaitable[RecommendResponse]
    ]
    type CacheAccessor[T] = Callable[[T], TLRUCache]

cache_ttl_var: contextvars.ContextVar[float] = contextvars.ContextVar(
    "cache_ttl", default=0
)


def ttu(_key: str, _value: object, now: float) -> float:
    return now + cache_ttl_var.get()


def create_response_cache() -> TLRUCache:
    return TLRUCache(maxsize=settings.cache_maxsize, ttu=ttu, timer=time.monotonic)


def async_cachedmethod[T](
    cache: CacheAccessor[T],
) -> Callable[[RecommendMethod[T]], RecommendMethod[T]]:
    """Async-aware memoization decorator for instance methods.

    Uses SHA256 of instructions + request JSON as cache key.
    """

    def decorator(func: RecommendMethod[T]) -> RecommendMethod[T]:
        @functools.wraps(func)
        async def wrapper(
            self: T, instructions: str, request: RecommendRequest
        ) -> RecommendResponse:
            c = cache(self)
            key = hashlib.sha256(
                f"{instructions}:{request.model_dump_json()}".encode()
            ).hexdigest()

            cached = c.get(key)
            if cached is not None:
                logger.info("recommend: cache hit")
                return cached

            result = await func(self, instructions, request)
            c[key] = result
            logger.info(
                "recommend: {} items (cached ttl={}s)",
                len(result.items),
                cache_ttl_var.get(),
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
