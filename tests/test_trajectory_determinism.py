"""Trajectory-level determinism.

``test_determinism.py`` covers scenario generation. This covers the whole loop:
same (task, seed, action sequence) must give byte-identical observations and
rewards, in solo and team mode. Regression guard for uuid4 ticket IDs, which
made team-mode observations and the manager-judge trajectory hash differ
between two otherwise identical episodes.
"""

import pytest

from baselines import SOLO_TASKS, TEAM_TASKS, make_agent
from models import EpisodeMode, SOCAction
from server.environment import SOCEnvironment


def _rollout(task: str, seed: int, agent_name: str, return_env: bool = False):
    mode = EpisodeMode.TEAM.value if task in TEAM_TASKS else EpisodeMode.TIER1_SOLO.value
    env = SOCEnvironment()
    agent = make_agent(agent_name)
    obs = env.reset(task, seed=seed, mode=mode)
    agent.reset(task, seed, mode, env._config)
    frames = []
    for _ in range(400):
        if obs.done:
            break
        obs = env.step(SOCAction.model_validate(agent.act(obs.model_dump(mode="json"))))
        # episode_id is a per-episode unique identifier (audit-trail key), not state.
        frames.append(obs.model_dump_json(exclude={"episode_id"}))
    if return_env:
        return env
    return frames, env._trajectory_hash()


@pytest.mark.parametrize("task", SOLO_TASKS + TEAM_TASKS)
@pytest.mark.parametrize("agent", ["heuristic", "random"])
def test_same_seed_same_actions_same_trajectory(task, agent):
    a_frames, a_hash = _rollout(task, 7, agent)
    b_frames, b_hash = _rollout(task, 7, agent)
    assert len(a_frames) == len(b_frames)
    for i, (a, b) in enumerate(zip(a_frames, b_frames, strict=True)):
        assert a == b, f"{task}/{agent}: observation diverged at step {i + 1}"
    assert a_hash == b_hash


def test_team_ticket_ids_are_sequential():
    env = _rollout("team_phishing_escalation", 42, "heuristic", return_env=True)
    ids = [t.ticket_id for t in env._tickets]
    assert ids, "scripted team policy should open at least one ticket"
    assert ids == [f"TKT-EP{i:04d}" for i in range(1, len(ids) + 1)]


def test_reset_clears_ticket_sequence():
    env = SOCEnvironment()
    for _ in range(2):
        env.reset("team_phishing_escalation", seed=42, mode="team")
        assert env._ticket_seq == 0
        assert env._next_ticket_id() == "TKT-EP0001"
