"""
Grader for the APT Campaign task (Theme #2, sparse + delayed reward).

Score blend:
    classification_f1        0.40
    kill_chain_recall        0.25
    narrative_token_quality  0.20   (Mercor token-scaled bonus)
    policy_compliance        0.15   (Patronus schema/policy honour rate)

The narrative component rewards long, substantive campaign summaries,
implementing the Mercor sub-theme directly inside the grader.
"""
from __future__ import annotations

from graders.base import BaseGrader
from graders.token_scaled_reward import token_scaled_bonus
from models import AlertClassification, InvestigationState, RewardBlendConfig, ScenarioConfig


class APTCampaignGrader(BaseGrader):
    """Sparse-reward grader for the apt_campaign task."""

    # ExternalState — set before calling grade() via set_context().
    def __init__(self) -> None:
        self._narrative_text: str = ""
        self._policy_compliance: float = 1.0
        self._blend: RewardBlendConfig | None = None

    def set_context(
        self,
        narrative_text: str = "",
        policy_compliance_rate: float = 1.0,
        blend: RewardBlendConfig | None = None,
    ) -> None:
        self._narrative_text = narrative_text or ""
        self._policy_compliance = max(0.0, min(1.0, policy_compliance_rate))
        self._blend = blend

    def _components(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
    ):
        gt = config.ground_truth
        tp_set = set(gt.true_positive_ids)
        pred_tp = {
            aid for aid, inv in investigations.items()
            if inv.classification == AlertClassification.TRUE_POSITIVE
        }
        if not tp_set:
            f1 = 1.0 if not pred_tp else 0.0
        else:
            tp_hit = len(tp_set & pred_tp)
            precision = tp_hit / len(pred_tp) if pred_tp else 0.0
            recall = tp_hit / len(tp_set)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        chain_hits = 0
        chain_total = 0
        for chain in gt.attack_chain_ids:
            chain_total += 1
            if chain and all(cid in pred_tp for cid in chain):
                chain_hits += 1
        chain_recall = chain_hits / chain_total if chain_total else 1.0

        blend = self._blend or RewardBlendConfig()
        narrative_bonus = token_scaled_bonus(
            text=self._narrative_text,
            content_quality=f1,
            config=blend,
        )
        narrative_component = (
            narrative_bonus / blend.token_scale_max_bonus if blend.token_scale_max_bonus else 0.0
        )
        return f1, chain_recall, narrative_component, self._policy_compliance

    def grade(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        f1, chain_recall, narrative, compliance = self._components(config, investigations)
        total = 0.40 * f1 + 0.25 * chain_recall + 0.20 * narrative + 0.15 * compliance
        return self._clamp(total)

    def grade_with_breakdown(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> tuple:
        f1, chain_recall, narrative, compliance = self._components(config, investigations)
        total = 0.40 * f1 + 0.25 * chain_recall + 0.20 * narrative + 0.15 * compliance
        total = self._clamp(total)
        breakdown = {
            "classification_f1": round(f1, 4),
            "kill_chain_recall": round(chain_recall, 4),
            "narrative_token_quality": round(narrative, 4),
            "policy_compliance": round(compliance, 4),
            "total": round(total, 4),
        }
        feedback = (
            f"APT campaign: F1={f1:.2f}, kill-chain={chain_recall:.2f}, "
            f"narrative={narrative:.2f}, policy={compliance:.2f}."
        )
        return total, breakdown, feedback
