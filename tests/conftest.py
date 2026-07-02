"""
Shared pytest fixtures for SOC-Triage-Gym test suite.
"""

import sys
import os

# Ensure project root is on sys.path so imports like "from models import ..."
# work the same way they do when running the server.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.environment import SOCEnvironment
from scenarios.phishing import PhishingScenario
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.queue_management import QueueManagementScenario
from scenarios.team_phishing_escalation import TeamPhishingEscalationScenario
from models import ScenarioConfig


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
    """Return a TestClient wired to the FastAPI app with a fresh environment.

    The lifespan handler creates a fresh SOCEnvironment on startup, so we
    just let the TestClient context manager handle it. After the lifespan
    runs we replace _env to ensure isolation.
    """
    import server.app as app_module
    with TestClient(app) as client:
        # After lifespan creates _env, replace with a fresh one for isolation
        app_module._env = SOCEnvironment()
        yield client
