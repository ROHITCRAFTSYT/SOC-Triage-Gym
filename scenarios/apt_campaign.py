"""
APT Campaign Scenario — (Super) Long-Horizon Planning (Theme #2).

Composes Insider-Threat + Lateral-Movement + Phishing into a single
5-phase APT simulation:

    1. initial_access       (phishing seed)
    2. persistence          (credential stuff)
    3. lateral_movement     (kill chain)
    4. exfiltration         (insider data theft)
    5. cleanup              (noise + false positives)

Exposes 60+ alerts and a 250-step budget. Reward is sparse — the full
grader fires only after the agent emits a campaign narrative via
`explain_team_behavior`, so early steps produce only tiny shaping rewards.
The context-window pressure forces summarization and note-taking, the
capability the long-horizon theme targets.
"""
from __future__ import annotations

from models import GroundTruth, ScenarioConfig
from scenarios.base import BaseScenario
from scenarios.insider_threat import InsiderThreatScenario
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.phishing import PhishingScenario


class APTCampaignScenario(BaseScenario):
    """250-step super-long-horizon composite scenario."""

    MAX_STEPS = 250

    def generate(self) -> ScenarioConfig:
        # Reuse existing scenario generators with distinct sub-seeds.
        phish = PhishingScenario(seed=self.seed + 101).generate()
        lateral = LateralMovementScenario(seed=self.seed + 202).generate()
        insider = InsiderThreatScenario(seed=self.seed + 303).generate()

        all_alerts = list(phish.alerts) + list(lateral.alerts) + list(insider.alerts)
        self.rng.shuffle(all_alerts)
        alert_ids = [a.alert_id for a in all_alerts]

        # Merge enrichment / log DBs (later wins on collision; collisions unlikely).
        enrichment_db = {}
        enrichment_db.update(phish.enrichment_db)
        enrichment_db.update(lateral.enrichment_db)
        enrichment_db.update(insider.enrichment_db)

        asset_db = {**phish.asset_db, **lateral.asset_db, **insider.asset_db}
        user_db = {**phish.user_db, **lateral.user_db, **insider.user_db}

        log_db = self._empty_log_db(alert_ids)
        for src_db in (phish.log_db, lateral.log_db, insider.log_db):
            for source_key, alert_map in src_db.items():
                if source_key not in log_db:
                    log_db[source_key] = {aid: [] for aid in alert_ids}
                for aid, entries in alert_map.items():
                    if aid in log_db[source_key]:
                        log_db[source_key][aid] = list(entries)

        # Compose ground truth.
        classifications = {}
        classifications.update(phish.ground_truth.alert_classifications)
        classifications.update(lateral.ground_truth.alert_classifications)
        classifications.update(insider.ground_truth.alert_classifications)

        tp_ids = (
            list(phish.ground_truth.true_positive_ids)
            + list(lateral.ground_truth.true_positive_ids)
            + list(insider.ground_truth.true_positive_ids)
        )
        btp_ids = (
            list(phish.ground_truth.benign_tp_ids)
            + list(lateral.ground_truth.benign_tp_ids)
            + list(insider.ground_truth.benign_tp_ids)
        )
        fp_ids = (
            list(phish.ground_truth.false_positive_ids)
            + list(lateral.ground_truth.false_positive_ids)
            + list(insider.ground_truth.false_positive_ids)
        )

        expected_tech = {}
        expected_tech.update(phish.ground_truth.expected_techniques)
        expected_tech.update(lateral.ground_truth.expected_techniques)
        expected_tech.update(insider.ground_truth.expected_techniques)

        attack_chains = (
            list(phish.ground_truth.attack_chain_ids)
            + list(lateral.ground_truth.attack_chain_ids)
            + list(insider.ground_truth.attack_chain_ids)
        )

        gt = GroundTruth(
            alert_classifications=classifications,
            true_positive_ids=tp_ids,
            false_positive_ids=fp_ids,
            benign_tp_ids=btp_ids,
            expected_techniques=expected_tech,
            attack_chain_ids=attack_chains,
            required_escalations=tp_ids,
        )

        return ScenarioConfig(
            scenario_id=f"apt-campaign-{self.seed}",
            task_id="apt_campaign",
            seed=self.seed,
            description=(
                "APT campaign: a single threat actor progresses through initial-access, "
                "persistence, lateral movement, exfiltration, and cleanup across 3 "
                "simulated days. Agent must track state across 60+ alerts and produce "
                "a coherent end-to-end narrative."
            ),
            max_steps=self.MAX_STEPS,
            alerts=all_alerts,
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=gt,
            difficulty_floor=0.75,
            noise_density=0.5,
            ioc_freshness=0.6,
            correlation_obfuscation=0.5,
        )
