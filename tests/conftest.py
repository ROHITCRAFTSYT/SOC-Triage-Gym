"""
Shared pytest fixtures for SOC-Triage-Gym test suite.
"""

import os
import sys

# Ensure project root is on sys.path so imports like "from models import ..."
# work the same way they do when running the server.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient

from models import ScenarioConfig
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.phishing import PhishingScenario
from scenarios.queue_management import QueueManagementScenario
from scenarios.team_phishing_escalation import TeamPhishingEscalationScenario
from server.app import app
from server.environment import SOCEnvironment

# ---------------------------------------------------------------------------
# Scenario config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def phishing_config() -> ScenarioConfig:
    """Generate a phishing scenario config with seed=42 (true positive)."""
    return PhishingScenario(seed=42).generate()


@pytest.fixture
def lateral_movement_config() -> ScenarioConfig:
    """Generate a lateral movement scenario config with seed=42."""
    return LateralMovementScenario(seed=42).generate()


@pytest.fixture
def queue_management_config() -> ScenarioConfig:
    """Generate a queue management scenario config with seed=42."""
    return QueueManagementScenario(seed=42).generate()


@pytest.fixture
def team_phishing_config() -> ScenarioConfig:
    """Generate a team phishing scenario config with seed=42."""
    return TeamPhishingEscalationScenario(seed=42).generate()


# ---------------------------------------------------------------------------
# Environment fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def environment() -> SOCEnvironment:
    """Return a fresh SOCEnvironment instance."""
    return SOCEnvironment()


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client() -> TestClient:
    """Return a TestClient wired to the FastAPI app with fresh state.

    Episode state lives in per-session containers (server.sessions), so we
    clear the session manager, metrics, and audit trail between tests for
    isolation.
    """
    import server.app as app_module
    with TestClient(app) as client:
        app_module._sessions.clear()
        app_module.METRICS.reset()
        app_module.AUDIT.clear()
        yield client
