"""
Structured logging
==================
One place to configure logging for the server, the benchmark and scripts.

``SOC_GYM_LOG_FORMAT``
    ``json`` (one JSON object per line: what log shippers like Loki, Datadog
    or CloudWatch want) or ``text`` (human-readable, the default for local
    development). The Docker image sets ``json``.

``SOC_GYM_LOG_LEVEL``
    Standard level name, default ``INFO``. ``DEBUG`` adds one line per step.

Every record carries the current request ID (from the ``X-Request-ID`` header,
or generated) so a single episode can be traced across access, environment and
audit lines. Structured fields go in ``extra=`` and appear as top-level JSON keys::

    logger.info("episode.reset", extra={"task_id": "phishing", "seed": 42})
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("soc_gym_request_id", default=None)

# Attributes every LogRecord has; anything else on a record came from extra=.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_HANDLER_MARKER = "_soc_gym_handler"
_MAX_REQUEST_ID = 128


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


def _extra_fields(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in _extra_fields(record).items():
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable line with structured fields appended as key=value."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = " ".join(f"{k}={v}" for k, v in _extra_fields(record).items() if v is not None)
        return f"{base} {fields}" if fields else base


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Install the gym's handler on the root logger. Safe to call repeatedly."""
    level_name = (level or os.environ.get("SOC_GYM_LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.environ.get("SOC_GYM_LOG_FORMAT", "text")).lower()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(JsonFormatter() if fmt_name == "json" else TextFormatter())
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))


def _clean_request_id(raw: str | None) -> str:
    # Echoing a client-supplied ID into logs is a log-injection vector, so
    # accept only short printable tokens and otherwise mint our own.
    if raw and len(raw) <= _MAX_REQUEST_ID and raw.isascii() and raw.isprintable() and " " not in raw:
        return raw
    return uuid.uuid4().hex


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Bind a request ID, emit one structured access line per request."""

    def __init__(self, app, logger_name: str = "soc_gym.access") -> None:
        super().__init__(app)
        self._log = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        request_id = _clean_request_id(request.headers.get("x-request-id"))
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            level = logging.WARNING if status >= 500 else logging.INFO
            self._log.log(
                level,
                "http.request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "session_id": request.headers.get("x-session-id"),
                },
            )
            request_id_var.reset(token)
