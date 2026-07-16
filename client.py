"""
SOC-Triage-Gym HTTP Client
============================
Production-grade client for the SOC-Triage-Gym environment server.
Suitable for inference scripts, evals, RL training loops, and testing.

Features:
  * Automatic retries with exponential backoff on transient failures
    (connection errors, 429 rate limits, 5xx responses).
  * Multi-tenant sessions via the ``X-Session-ID`` header, so many clients
    can run isolated episodes against one server.
  * API-key auth (``Authorization: Bearer``) for servers started with
    ``SOC_GYM_API_KEY``.
  * ``run_episode()`` — drive an agent policy to episode completion in
    one call.
  * Accessors for the grader, task catalog, audit trail, and metrics.

Usage:
    with SOCTriageClient("http://localhost:7860", session_id="trainer-1") as client:
        obs = client.reset("phishing", seed=42)
        while not obs.done:
            action = agent.decide(obs)
            obs = client.step(action)
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import httpx

from models import EnvironmentState, SOCAction, SOCObservation

logger = logging.getLogger(__name__)

# Status codes worth retrying: rate limit + transient server-side failures.
_RETRY_STATUS = {429, 502, 503, 504}


class SOCTriageClient:
    """
    HTTP client for the SOC-Triage-Gym environment server.

    Args:
        base_url: Server root, e.g. ``http://localhost:7860``.
        timeout: Per-request timeout in seconds.
        session_id: Optional session name — isolates this client's episodes
            from other clients on the same server. Omit for the default
            (single-tenant) session.
        api_key: Optional API key, required when the server sets
            ``SOC_GYM_API_KEY``.
        max_retries: Retry attempts for transient failures (0 disables).
        backoff_base: First retry delay in seconds; doubles per attempt
            with jitter.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        session_id: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base

        headers = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    # -- low-level request with retry ------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                if response.status_code in _RETRY_STATUS and attempt < self.max_retries:
                    self._sleep(attempt, f"HTTP {response.status_code}")
                    continue
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    self._sleep(attempt, type(exc).__name__)
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("request failed")  # pragma: no cover

    def _sleep(self, attempt: int, reason: str) -> None:
        delay = self.backoff_base * (2**attempt) * (1 + random.random() * 0.25)
        logger.debug("Retrying after %s (attempt %d, sleeping %.2fs)", reason, attempt + 1, delay)
        time.sleep(delay)

    # -- core OpenEnv API --------------------------------------------------------

    def health(self) -> dict:
        """Check server health."""
        return self._request("GET", "/health").json()

    def reset(self, task_id: str = "phishing", seed: int = 42, mode: str = "tier1_solo") -> SOCObservation:
        """
        Start a new episode.

        Args:
            task_id: Any ID from ``tasks()`` — e.g. "phishing", "apt_campaign".
            seed: Deterministic seed for scenario generation.
            mode: "tier1_solo" or "team".

        Returns:
            Initial SOCObservation.
        """
        response = self._request("POST", "/reset", json={"task_id": task_id, "seed": seed, "mode": mode})
        return SOCObservation.model_validate(response.json())

    def step(self, action: SOCAction) -> SOCObservation:
        """
        Execute an action.

        Args:
            action: SOCAction to execute.

        Returns:
            Updated SOCObservation.
        """
        response = self._request(
            "POST",
            "/step",
            content=action.model_dump_json(exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
        return SOCObservation.model_validate(response.json())

    def step_raw(self, action_dict: dict) -> SOCObservation:
        """Execute an action from a plain dict (for LLM output parsing)."""
        action = SOCAction.model_validate(action_dict)
        return self.step(action)

    def state(self) -> EnvironmentState:
        """Get current episode metadata."""
        response = self._request("GET", "/state")
        return EnvironmentState.model_validate(response.json())

    # -- convenience -------------------------------------------------------------

    def run_episode(
        self,
        policy: Callable[[SOCObservation], SOCAction],
        task_id: str = "phishing",
        seed: int = 42,
        mode: str = "tier1_solo",
        max_steps: int | None = None,
    ) -> tuple[SOCObservation, list[SOCObservation]]:
        """
        Drive ``policy`` to episode completion.

        Args:
            policy: Callable mapping the current observation to the next action.
            task_id / seed / mode: Passed to ``reset``.
            max_steps: Optional client-side safety cap on steps.

        Returns:
            (final_observation, trajectory) — trajectory includes every
            observation from reset through the final step.
        """
        obs = self.reset(task_id=task_id, seed=seed, mode=mode)
        trajectory = [obs]
        steps = 0
        while not obs.done:
            if max_steps is not None and steps >= max_steps:
                break
            obs = self.step(policy(obs))
            trajectory.append(obs)
            steps += 1
        return obs, trajectory

    def tasks(self) -> list[dict]:
        """List the task catalog."""
        return self._request("GET", "/tasks").json()["tasks"]

    def grade(self) -> dict:
        """Run the grader on the current episode without terminating it."""
        return self._request("POST", "/grader").json()

    def episodes(self, limit: int = 50) -> list[dict]:
        """List recorded episode audit summaries for this client's session."""
        params: dict = {"limit": limit}
        if self.session_id:
            params["session_id"] = self.session_id
        return self._request("GET", "/episodes", params=params).json()["episodes"]

    def episode_trace(self, episode_id: str) -> dict:
        """Fetch the full audit trace (every action + reward) for an episode."""
        return self._request("GET", f"/episodes/{episode_id}/trace").json()

    def metrics_text(self) -> str:
        """Fetch server metrics in Prometheus text format."""
        return self._request("GET", "/metrics").text

    # -- lifecycle -----------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> SOCTriageClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()
