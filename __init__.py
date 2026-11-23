"""
SOC-Triage-Gym
==============
OpenEnv-compliant reinforcement learning environment for SOC analyst training.

Public API:
    SOCAction        — action model (what the agent sends)
    SOCObservation   — observation model (what the agent receives)
    SOCReward        — reward breakdown model
    SOCEnvironment   — core environment (used by server)

Quick start:
    from server.environment import SOCEnvironment
    env = SOCEnvironment()
    obs = env.reset("phishing", seed=42)
    obs = env.step(SOCAction(action_type="enrich_indicator", indicator="1.2.3.4", indicator_type="ip"))
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from models import SOCAction, SOCObservation, SOCReward
from server.environment import SOCEnvironment

try:
    # Single-source the version from installed package metadata (pyproject), so
    # it can't drift from the hardcoded literal it used to be (was "0.1.0" while
    # pyproject said "0.3.0").
    __version__ = _pkg_version("soc-triage-gym")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.3.0"
__all__ = ["SOCAction", "SOCObservation", "SOCReward", "SOCEnvironment"]
