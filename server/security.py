"""
Optional API-key authentication and rate limiting
=================================================

Both features are **off by default** so the open-source / hackathon experience
is unchanged. Production operators enable them with environment variables:

``SOC_GYM_API_KEY``
    When set, every API request must carry the key in either an
    ``Authorization: Bearer <key>`` header or an ``X-API-Key: <key>`` header.
    Browser-facing pages (``/``, ``/ui*``, ``/docs``, ``/openapi.json``) and
    the ``/health`` liveness probe stay open so load balancers and humans can
    reach them.

``SOC_GYM_RATE_LIMIT``
    Requests-per-minute cap per client (keyed by API key when present,
    otherwise client IP). Uses a dependency-free token bucket. ``0`` or unset
    disables limiting.

Keys are compared with ``hmac.compare_digest`` to avoid timing side-channels.
"""

from __future__ import annotations

import hmac
import os
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Paths that must stay reachable without a key: liveness probes, the browser
# landing pages, and the interactive API docs.
EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/blog.md", "/favicon.ico"}
EXEMPT_PREFIXES = ("/ui", "/static")


def _configured_api_key() -> str | None:
    key = os.environ.get("SOC_GYM_API_KEY", "").strip()
    return key or None


def _configured_rate_limit() -> int:
    try:
        return max(0, int(os.environ.get("SOC_GYM_RATE_LIMIT", "0")))
    except (TypeError, ValueError):
        return 0


def _extract_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _is_exempt(path: str) -> bool:
    if path in EXEMPT_PATHS:
        return True
    return any(path.startswith(p) for p in EXEMPT_PREFIXES)


class _TokenBucket:
    __slots__ = ("tokens", "updated")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.updated = time.monotonic()


class RateLimiter:
    """Per-client token bucket: ``limit`` requests per minute, burst up to ``limit``."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        rate_per_sec = self.limit / 60.0
        with self._lock:
            bucket = self._buckets.get(client_key)
            if bucket is None:
                bucket = _TokenBucket(capacity=float(self.limit))
                self._buckets[client_key] = bucket
            bucket.tokens = min(float(self.limit), bucket.tokens + (now - bucket.updated) * rate_per_sec)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforces the optional API key and rate limit. Reads config at request
    time so tests (and operators using process managers) can toggle behaviour
    via environment variables without rebuilding the app."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self._limiter: RateLimiter | None = None
        self._limiter_limit = -1
        self._limiter_lock = threading.Lock()

    def _get_limiter(self) -> RateLimiter:
        limit = _configured_rate_limit()
        with self._limiter_lock:
            if self._limiter is None or self._limiter_limit != limit:
                self._limiter = RateLimiter(limit)
                self._limiter_limit = limit
            return self._limiter

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        api_key = _configured_api_key()
        supplied = _extract_key(request)
        if api_key is not None and not _is_exempt(path):
            if supplied is None or not hmac.compare_digest(supplied, api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Missing or invalid API key. Send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'."
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )

        limiter = self._get_limiter()
        if limiter.limit > 0 and not _is_exempt(path):
            client_key = supplied or (request.client.host if request.client else "unknown")
            if not limiter.allow(client_key):
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit exceeded ({limiter.limit} requests/minute). Retry later."},
                    headers={"Retry-After": "60"},
                )

        return await call_next(request)
