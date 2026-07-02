"""
SOC-Triage-Gym HTTP Client
============================
Client wrapper for interacting with the SOC-Triage-Gym server via HTTP.
Suitable for use in inference scripts, evals, and testing.
"""


import httpx

from models import EnvironmentState, SOCAction, SOCObservation


class SOCTriageClient:
    """
    HTTP client for the SOC-Triage-Gym environment server.

    Usage:
        client = SOCTriageClient(base_url="http://localhost:8000")
        obs = client.reset("phishing", seed=42)
        obs = client.step(SOCAction(action_type="enrich_indicator", ...))
        while not obs.done:
            action = agent.decide(obs)
            obs = client.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict:
        """Check server health."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def reset(self, task_id: str = "phishing", seed: int = 42) -> SOCObservation:
        """
        Start a new episode.

        Args:
            task_id: "phishing" | "lateral_movement" | "queue_management"
            seed: Deterministic seed for scenario generation.

        Returns:
            Initial SOCObservation.
        """
        response = self._client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        response.raise_for_status()
        return SOCObservation.model_validate(response.json())

    def step(self, action: SOCAction) -> SOCObservation:
        """
        Execute an action.

        Args:
            action: SOCAction to execute.

        Returns:
            Updated SOCObservation.
        """
        response = self._client.post(
            "/step",
            content=action.model_dump_json(exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return SOCObservation.model_validate(response.json())

    def state(self) -> EnvironmentState:
        """Get current episode metadata."""
        response = self._client.get("/state")
        response.raise_for_status()
        return EnvironmentState.model_validate(response.json())

    def step_raw(self, action_dict: dict) -> SOCObservation:
        """Execute an action from a plain dict (for LLM output parsing)."""
        action = SOCAction.model_validate(action_dict)
        return self.step(action)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
