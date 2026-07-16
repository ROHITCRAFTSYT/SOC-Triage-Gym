"""
Dependency-free Prometheus metrics
==================================

Exposes operational counters in the Prometheus text exposition format at
``GET /metrics`` — no ``prometheus_client`` dependency, so the base image
stays slim and the endpoint works everywhere the server does.

Tracked:
  * ``socgym_requests_total{method,path,status}``   — HTTP request counts
  * ``socgym_request_seconds_sum/count{path}``      — latency accumulators
  * ``socgym_episodes_started_total{task}``         — episodes begun
  * ``socgym_episodes_completed_total{task}``       — episodes finished
  * ``socgym_steps_total{task}``                    — env steps executed
  * ``socgym_reward_sum/count{task}``               — final-reward accumulators
  * ``socgym_active_sessions``                      — live session gauge

Route labels are the FastAPI route template (``/api/alerts/{alert_id}``),
not the raw path, so cardinality stays bounded.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsRegistry:
    """Thread-safe counter store, rendered as Prometheus text on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.episodes_started: dict[str, int] = defaultdict(int)
        self.episodes_completed: dict[str, int] = defaultdict(int)
        self.steps: dict[str, int] = defaultdict(int)
        self.reward_sum: dict[str, float] = defaultdict(float)
        self.reward_count: dict[str, int] = defaultdict(int)
        self.started_at = time.time()

    # -- recording ------------------------------------------------------------

    def record_request(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(method, path, status)] += 1
            self.latency_sum[path] += seconds
            self.latency_count[path] += 1

    def record_episode_start(self, task: str) -> None:
        with self._lock:
            self.episodes_started[task] += 1

    def record_step(self, task: str) -> None:
        with self._lock:
            self.steps[task] += 1

    def record_episode_complete(self, task: str, final_reward: float) -> None:
        with self._lock:
            self.episodes_completed[task] += 1
            self.reward_sum[task] += final_reward
            self.reward_count[task] += 1

    def reset(self) -> None:
        """Zero all counters (used by tests)."""
        with self._lock:
            self.requests.clear()
            self.latency_sum.clear()
            self.latency_count.clear()
            self.episodes_started.clear()
            self.episodes_completed.clear()
            self.steps.clear()
            self.reward_sum.clear()
            self.reward_count.clear()

    # -- rendering ------------------------------------------------------------

    def render(self, active_sessions: int = 0) -> str:
        with self._lock:
            lines: list[str] = []

            def head(name: str, help_text: str, mtype: str) -> None:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {mtype}")

            head("socgym_uptime_seconds", "Seconds since server start.", "gauge")
            lines.append(f"socgym_uptime_seconds {time.time() - self.started_at:.1f}")

            head("socgym_active_sessions", "Number of live sessions.", "gauge")
            lines.append(f"socgym_active_sessions {active_sessions}")

            head("socgym_requests_total", "HTTP requests served.", "counter")
            for (method, path, status), n in sorted(self.requests.items()):
                lines.append(f'socgym_requests_total{{method="{method}",path="{path}",status="{status}"}} {n}')

            head("socgym_request_seconds_sum", "Total request latency per route.", "counter")
            for path, s in sorted(self.latency_sum.items()):
                lines.append(f'socgym_request_seconds_sum{{path="{path}"}} {s:.6f}')
            head("socgym_request_seconds_count", "Request count per route.", "counter")
            for path, n in sorted(self.latency_count.items()):
                lines.append(f'socgym_request_seconds_count{{path="{path}"}} {n}')

            head("socgym_episodes_started_total", "Episodes started per task.", "counter")
            for task, n in sorted(self.episodes_started.items()):
                lines.append(f'socgym_episodes_started_total{{task="{task}"}} {n}')

            head("socgym_episodes_completed_total", "Episodes completed per task.", "counter")
            for task, n in sorted(self.episodes_completed.items()):
                lines.append(f'socgym_episodes_completed_total{{task="{task}"}} {n}')

            head("socgym_steps_total", "Environment steps executed per task.", "counter")
            for task, n in sorted(self.steps.items()):
                lines.append(f'socgym_steps_total{{task="{task}"}} {n}')

            head("socgym_reward_sum", "Sum of final episode rewards per task.", "counter")
            for task, s in sorted(self.reward_sum.items()):
                lines.append(f'socgym_reward_sum{{task="{task}"}} {s:.6f}')
            head("socgym_reward_count", "Count of graded episodes per task.", "counter")
            for task, n in sorted(self.reward_count.items()):
                lines.append(f'socgym_reward_count{{task="{task}"}} {n}')

            return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Times every request and records it against the route template."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", None) or request.url.path
        METRICS.record_request(request.method, path, response.status_code, elapsed)
        return response
