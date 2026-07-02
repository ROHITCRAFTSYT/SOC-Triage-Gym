"""
Token-scaled quality reward (Mercor sub-theme, Theme #2).

Free-text actions (Manager explanations, APT-campaign narratives) earn a
length-scaled bonus:

    raw_len = count_tokens(text)
    if raw_len < floor:  bonus = 0.0   # too terse to judge
    elif raw_len >= cap: frac  = 1.0   # saturated — can't farm by rambling
    else:                frac  = (raw_len - floor) / (cap - floor)

    quality = content_quality_score     # 0.0..1.0, from ManagerJudge / ExpertPanel
    bonus   = max_bonus * frac * quality

The quality gate means long-and-empty gets 0; short-but-perfect also gets 0.
Only long-AND-substantive saturates. This matches Mercor's brief of "frontier
model rewards scale with token output" without allowing trivial farming.
"""
from __future__ import annotations

from models import RewardBlendConfig


def count_tokens(text: str) -> int:
    """Cheap whitespace tokenizer. Good enough for reward scaling."""
    if not text:
        return 0
    return len(text.strip().split())


def token_scaled_bonus(
    text: str,
    content_quality: float,
    config: RewardBlendConfig | None = None,
) -> float:
    """
    Compute the token-length-scaled reward bonus.

    Args:
        text: agent free-text output (explanation, narrative, etc.)
        content_quality: 0.0..1.0 quality gate from an upstream judge.
        config: blend config; defaults to RewardBlendConfig() if None.

    Returns:
        Non-negative float in [0, config.token_scale_max_bonus].
    """
    cfg = config or RewardBlendConfig()
    if not cfg.token_scale_enabled:
        return 0.0
    quality = max(0.0, min(1.0, content_quality))
    if quality == 0.0:
        return 0.0

    n = count_tokens(text)
    if n < cfg.token_scale_floor:
        return 0.0
    if n >= cfg.token_scale_cap:
        frac = 1.0
    else:
        span = max(1, cfg.token_scale_cap - cfg.token_scale_floor)
        frac = (n - cfg.token_scale_floor) / span
    return cfg.token_scale_max_bonus * frac * quality


def explain(text: str, content_quality: float, config: RewardBlendConfig | None = None) -> dict:
    """Return a dict suitable for debugging / /reward_config reporting."""
    cfg = config or RewardBlendConfig()
    bonus = token_scaled_bonus(text, content_quality, cfg)
    return {
        "tokens": count_tokens(text),
        "floor": cfg.token_scale_floor,
        "cap": cfg.token_scale_cap,
        "max_bonus": cfg.token_scale_max_bonus,
        "content_quality": content_quality,
        "bonus": bonus,
    }
