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

from models import SOCAction, SOCObservation, SOCReward
from server.environment import SOCEnvironment

__version__ = "0.3.0"
__all__ = ["SOCAction", "SOCObservation", "SOCReward", "SOCEnvironment"]
