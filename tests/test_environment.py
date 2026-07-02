"""
Tests for SOCEnvironment: reset, step, state, budget.
"""


from models import (
    ActionType,
    AgentRole,
    AlertClassification,
    EpisodeMode,
    IndicatorType,
    LogSource,
    SOCAction,
    SOCObservation,
)


class TestSOCEnvironment:
    """Tests for the SOCEnvironment state machine."""

    def test_reset_creates_episode(self, environment):
        """reset() should create a new episode with alerts and initial observation."""
        obs = environment.reset(task_id="phishing", seed=42)

        assert isinstance(obs, SOCObservation)
        assert obs.step == 0
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.cumulative_reward == 0.0
        assert obs.task_id == "phishing"
        assert obs.episode_id is not None
        assert len(obs.alert_queue) == 1  # phishing has 1 alert
        assert obs.investigation_budget == 15  # phishing max_steps

    def test_step_enrich_indicator(self, environment):
        """Enriching a known indicator should return enrichment results and positive reward."""
        obs = environment.reset(task_id="phishing", seed=42)
        alert = obs.alert_queue[0]

        # Get an IP indicator from the alert
        ip_indicators = alert.indicators.get("ip", [])
        assert len(ip_indicators) > 0, "Phishing alert should have IP indicators"

        action = SOCAction(
            action_type=ActionType.ENRICH_INDICATOR,
            indicator=ip_indicators[0],
            indicator_type=IndicatorType.IP,
        )
        obs = environment.step(action)

        assert obs.step == 1
        assert obs.done is False
        assert len(obs.enrichment_results) == 1
        assert obs.enrichment_results[0].indicator == ip_indicators[0]

    def test_step_query_logs(self, environment):
        """Querying a relevant log source should return log entries."""
        obs = environment.reset(task_id="phishing", seed=42)
        alert_id = obs.alert_queue[0].alert_id

        action = SOCAction(
            action_type=ActionType.QUERY_LOGS,
            log_source=LogSource.EMAIL_GATEWAY,
            query_alert_id=alert_id,
        )
        obs = environment.step(action)

        assert obs.step == 1
        assert len(obs.log_results) > 0, "Email gateway should have logs for phishing alert"

    def test_step_classify_alert(self, environment):
        """Classifying an alert after gathering evidence should work."""
        obs = environment.reset(task_id="phishing", seed=42)
        alert = obs.alert_queue[0]
        alert_id = alert.alert_id

        # First gather some evidence (enrich an indicator)
        ip_indicators = alert.indicators.get("ip", [])
        enrich_action = SOCAction(
            action_type=ActionType.ENRICH_INDICATOR,
            indicator=ip_indicators[0],
            indicator_type=IndicatorType.IP,
        )
        environment.step(enrich_action)

        # Now classify
        classify_action = SOCAction(
            action_type=ActionType.CLASSIFY_ALERT,
            alert_id=alert_id,
            classification=AlertClassification.TRUE_POSITIVE,
            confidence=0.9,
        )
        obs = environment.step(classify_action)

        assert obs.step == 2
        # Check that the classification was recorded
        assert obs.investigations[alert_id].classification == AlertClassification.TRUE_POSITIVE

    def test_classify_after_user_lookup_counts_as_evidence(self, environment):
        """User lookups should count as evidence and avoid the no-evidence penalty."""
        obs = environment.reset(task_id="phishing", seed=42)
        alert_id = obs.alert_queue[0].alert_id
        username = next(iter(environment._config.user_db.keys()))

        lookup_obs = environment.step(SOCAction(
            action_type=ActionType.CHECK_USER,
            username=username,
        ))
        assert lookup_obs.user_info is not None

        obs = environment.step(SOCAction(
            action_type=ActionType.CLASSIFY_ALERT,
            alert_id=alert_id,
            classification=AlertClassification.TRUE_POSITIVE,
            confidence=0.9,
        ))

        assert "without gathering any evidence" not in obs.message
        assert obs.reward == 0.30

    def test_step_submit_investigation(self, environment):
        """Submitting investigation should finalize the episode (done=True)."""
        obs = environment.reset(task_id="phishing", seed=42)
        alert_id = obs.alert_queue[0].alert_id

        # Classify the alert first (with evidence)
        ip_indicators = obs.alert_queue[0].indicators.get("ip", [])
        environment.step(SOCAction(
            action_type=ActionType.ENRICH_INDICATOR,
            indicator=ip_indicators[0],
            indicator_type=IndicatorType.IP,
        ))
        environment.step(SOCAction(
            action_type=ActionType.CLASSIFY_ALERT,
            alert_id=alert_id,
            classification=AlertClassification.TRUE_POSITIVE,
            confidence=0.9,
        ))

        # Submit
        obs = environment.step(SOCAction(action_type=ActionType.SUBMIT_INVESTIGATION))

        assert obs.done is True
        assert "submitted" in obs.message.lower() or "grader" in obs.message.lower()

    def test_budget_exhaustion(self, environment):
        """Exhausting the step budget should auto-terminate the episode."""
        obs = environment.reset(task_id="phishing", seed=42)

        # Phishing has max_steps=15, use NOOP to burn through budget
        for _ in range(15):
            obs = environment.step(SOCAction(action_type=ActionType.NOOP))

        assert obs.done is True
        assert "budget" in obs.message.lower() or "exhausted" in obs.message.lower()

    def test_deterministic_seed(self, environment):
        """Same seed should produce the same scenario."""
        obs1 = environment.reset(task_id="phishing", seed=123)
        alert1 = obs1.alert_queue[0]

        # Reset with same seed
        obs2 = environment.reset(task_id="phishing", seed=123)
        alert2 = obs2.alert_queue[0]

        assert alert1.alert_id == alert2.alert_id
        assert alert1.title == alert2.title
        assert alert1.indicators == alert2.indicators

        # Different seed should produce different scenario
        obs3 = environment.reset(task_id="phishing", seed=999)
        alert3 = obs3.alert_queue[0]
        assert alert3.alert_id != alert1.alert_id

    def test_team_reset_starts_in_tier1_phase(self, environment):
        """Team reset should expose team metadata and start with Tier-1."""
        obs = environment.reset(task_id="team_phishing_escalation", seed=42, mode="team")

        assert obs.episode_mode == EpisodeMode.TEAM
        assert obs.current_role == AgentRole.TIER1
        assert obs.current_phase is not None
        assert obs.phase_steps_remaining == 40
        assert obs.investigation_budget == 68
        assert obs.tickets == []

    def test_team_escalation_creates_ticket_and_filters_tier2_view(self, environment):
        """Tier-1 escalation should create a ticket and reveal it to Tier-2 after phase advance."""
        obs = environment.reset(task_id="team_phishing_escalation", seed=42, mode="team")
        alert_id = obs.alert_queue[0].alert_id
        ip_indicator = obs.alert_queue[0].indicators["ip"][0]

        environment.step(SOCAction(
            action_type=ActionType.ENRICH_INDICATOR,
            role=AgentRole.TIER1,
            indicator=ip_indicator,
            indicator_type=IndicatorType.IP,
        ))
        environment.step(SOCAction(
            action_type=ActionType.CLASSIFY_ALERT,
            role=AgentRole.TIER1,
            alert_id=alert_id,
            classification=AlertClassification.TRUE_POSITIVE,
            confidence=0.9,
        ))
        environment.step(SOCAction(
            action_type=ActionType.ESCALATE_TO_TIER2,
            role=AgentRole.TIER1,
            alert_id=alert_id,
            justification="Malicious phishing indicators and execution evidence.",
        ))

        obs = environment.step(SOCAction(
            action_type=ActionType.PHASE_COMPLETE,
            role=AgentRole.TIER1,
        ))

        assert obs.current_role == AgentRole.TIER2
        assert len(obs.alert_queue) == 1
        assert len(obs.tickets) == 1
        assert obs.tickets[0].alert_id == alert_id

    def test_manager_observation_includes_consistency_stats(self, environment):
        """Manager phase should expose historical consistency stats."""
        obs = environment.reset(task_id="team_phishing_escalation", seed=42, mode="team")
        alert_id = obs.alert_queue[0].alert_id
        ip_indicator = obs.alert_queue[0].indicators["ip"][0]

        environment.step(SOCAction(
            action_type=ActionType.ENRICH_INDICATOR,
            role=AgentRole.TIER1,
            indicator=ip_indicator,
            indicator_type=IndicatorType.IP,
        ))
        environment.step(SOCAction(
            action_type=ActionType.CLASSIFY_ALERT,
            role=AgentRole.TIER1,
            alert_id=alert_id,
            classification=AlertClassification.TRUE_POSITIVE,
            confidence=0.9,
        ))
        environment.step(SOCAction(
            action_type=ActionType.ESCALATE_TO_TIER2,
            role=AgentRole.TIER1,
            alert_id=alert_id,
            justification="Malicious indicators and execution evidence.",
        ))
        environment.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
        environment.step(SOCAction(
            action_type=ActionType.CLOSE_CASE,
            role=AgentRole.TIER2,
            alert_id=alert_id,
            justification="Contained and documented.",
        ))
        obs = environment.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER2))

        assert obs.current_role == AgentRole.MANAGER
        assert obs.consistency_stats is not None
        assert obs.consistency_stats.tickets_total >= 2
