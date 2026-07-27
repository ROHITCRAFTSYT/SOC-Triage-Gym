"""Behavioural tests for the token-scaled quality reward.

token_scaled_reward is anti-gaming reward shaping: long-and-empty must earn 0,
short-but-perfect must earn 0, and only long-AND-substantive may saturate. The
existing suite only imports the module for theme coverage; these pin the actual
math at the floor, the cap, the quality gate, and the clamps.
"""
from __future__ import annotations

import pytest

from graders.token_scaled_reward import count_tokens, explain, token_scaled_bonus
from models import RewardBlendConfig

CFG = RewardBlendConfig()  # floor=20, cap=400, max_bonus=0.10


def _text(n: int) -> str:
    return " ".join(["word"] * n)


# --------------------------------------------------------------- count_tokens
def test_count_tokens_whitespace_and_empty():
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    assert count_tokens("one two   three") == 3


# --------------------------------------------------------------- quality gate
def test_zero_quality_earns_nothing_even_when_long():
    assert token_scaled_bonus(_text(1000), content_quality=0.0, config=CFG) == 0.0


def test_quality_is_clamped_into_unit_range():
    # quality > 1 must not exceed max_bonus at saturation
    hi = token_scaled_bonus(_text(CFG.token_scale_cap), 5.0, CFG)
    assert hi == pytest.approx(CFG.token_scale_max_bonus)
    # negative quality clamps to 0
    assert token_scaled_bonus(_text(CFG.token_scale_cap), -3.0, CFG) == 0.0


# --------------------------------------------------------------- length gate
def test_below_floor_earns_nothing_even_when_perfect():
    assert token_scaled_bonus(_text(CFG.token_scale_floor - 1), 1.0, CFG) == 0.0


def test_at_or_above_cap_saturates():
    at_cap = token_scaled_bonus(_text(CFG.token_scale_cap), 1.0, CFG)
    beyond = token_scaled_bonus(_text(CFG.token_scale_cap + 500), 1.0, CFG)
    assert at_cap == pytest.approx(CFG.token_scale_max_bonus)
    assert beyond == pytest.approx(CFG.token_scale_max_bonus)


def test_midpoint_scales_linearly():
    mid_tokens = (CFG.token_scale_floor + CFG.token_scale_cap) // 2
    bonus = token_scaled_bonus(_text(mid_tokens), 1.0, CFG)
    # roughly half of max_bonus at the midpoint
    assert bonus == pytest.approx(CFG.token_scale_max_bonus * 0.5, abs=CFG.token_scale_max_bonus * 0.02)


def test_bonus_is_monotonic_in_length():
    prev = -1.0
    for n in (CFG.token_scale_floor, 100, 200, 300, CFG.token_scale_cap):
        cur = token_scaled_bonus(_text(n), 1.0, CFG)
        assert cur >= prev
        prev = cur


# --------------------------------------------------------------- disabled
def test_disabled_scaler_returns_zero():
    cfg = RewardBlendConfig(token_scale_enabled=False)
    assert token_scaled_bonus(_text(1000), 1.0, cfg) == 0.0


# --------------------------------------------------------------- explain
def test_explain_reports_consistent_fields():
    info = explain(_text(CFG.token_scale_cap), 1.0, CFG)
    assert info["tokens"] == CFG.token_scale_cap
    assert info["floor"] == CFG.token_scale_floor
    assert info["cap"] == CFG.token_scale_cap
    assert info["bonus"] == pytest.approx(CFG.token_scale_max_bonus)
