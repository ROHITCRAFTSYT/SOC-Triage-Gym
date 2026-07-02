"""
SOC-Triage-Gym Core Environment v2
====================================
Implements SOCEnvironment with reset(), step(), and state() methods.
Supports both tier1_solo mode (backward-compatible with Round 1) and
team mode (multi-agent: Tier-1 → Tier-2 → Manager, with phase budgets).

All data access is from in-memory ScenarioConfig; no disk I/O during episodes.
Tool functions are pure and imported from tools/.
"""

import hashlib
import json
import uuid

from data.mitre_attack import is_valid_technique
from graders import GRADER_REGISTRY
from graders.team_grader import compute_team_metrics
from models import (
    ActionType,
    AgentRole,
    AlertClassification,
    ConsistencyStats,
    ContainmentResult,
    EnvironmentState,
    EpisodeMode,
    EpisodePhase,
    InvestigationState,
    ManagerReviewResult,
    ResponseActionType,
    ScenarioConfig,
    SOCAction,
    SOCObservation,
    TeamRewardBreakdown,
    TicketKind,
    TicketMessage,
)
from scenarios import SCENARIO_REGISTRY
from tools.asset_lookup import lookup_asset
from tools.containment import (
    block_ioc,
    close_case,
    disable_user_account,
    forensic_timeline,
    isolate_host,
    memory_analysis,
    sandbox_detonate,
)
from tools.correlation import correlate_alerts
from tools.enrichment import enrich_indicator
from tools.log_query import query_logs
from tools.oversight import (
    explain_team_behavior,
    flag_inconsistency,
    override_classification,
    review_decision,
)
from tools.user_lookup import lookup_user

# Per-phase step budgets in team mode
_PHASE_BUDGETS: dict[EpisodePhase, int] = {
    EpisodePhase.TRIAGE: 40,
    EpisodePhase.RESPONSE: 20,
    EpisodePhase.OVERSIGHT: 8,
}
_TEAM_MAX_STEPS = sum(_PHASE_BUDGETS.values())

# Actions allowed per role
_TIER1_ACTIONS = {
    ActionType.ENRICH_INDICATOR, ActionType.QUERY_LOGS, ActionType.CORRELATE_ALERTS,
    ActionType.CHECK_ASSET, ActionType.CHECK_USER, ActionType.CLASSIFY_ALERT,
    ActionType.MAP_TECHNIQUE, ActionType.RECOMMEND_ACTION, ActionType.ESCALATE,
    ActionType.ESCALATE_TO_TIER2, ActionType.SUBMIT_INVESTIGATION,
    ActionType.PHASE_COMPLETE, ActionType.NOOP,
}
_TIER2_ACTIONS = {
    ActionType.FORENSIC_TIMELINE, ActionType.SANDBOX_DETONATE,
    ActionType.MEMORY_ANALYSIS, ActionType.ISOLATE_HOST, ActionType.DISABLE_USER,
    ActionType.BLOCK_IOC, ActionType.CLOSE_CASE, ActionType.PHASE_COMPLETE, ActionType.NOOP,
}
_MANAGER_ACTIONS = {
    ActionType.REVIEW_DECISION, ActionType.OVERRIDE_CLASSIFICATION,
    ActionType.FLAG_INCONSISTENCY, ActionType.EXPLAIN_TEAM_BEHAVIOR,
    ActionType.SUBMIT_INVESTIGATION, ActionType.PHASE_COMPLETE, ActionType.NOOP,
}


class SOCEnvironment:
    """
    Stateful SOC triage environment supporting both solo and multi-agent team modes.

    Thread safety: callers hold a lock when calling reset(), step(), state().
    """

    def __init__(self) -> None:
        self._config: ScenarioConfig | None = None
        self._investigations: dict[str, InvestigationState] = {}
        self._cumulative_reward: float = 0.0
        self._step: int = 0
        self._done: bool = False
        self._task_id: str | None = None
        self._episode_id: str | None = None
        self._action_history: list[str] = []
        # Multi-agent state
        self._mode: EpisodeMode = EpisodeMode.TIER1_SOLO
        self._phase: EpisodePhase | None = None
        self._phase_step: int = 0
        self._tickets: list[TicketMessage] = []
        self._containment_results: list[ContainmentResult] = []
        self._manager_review: ManagerReviewResult | None = None
        self._escalated_alert_ids: list[str] = []
        self._generated_scenario: ScenarioConfig | None = None
        # Per-role cumulative rewards (team mode)
        self._tier1_reward: float = 0.0
        self._tier2_reward: float = 0.0
        self._manager_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = 42, mode: str = "tier1_solo") -> SOCObservation:
        """
        Start a new episode.

        Args:
            task_id: Scenario ID. Legacy: "phishing"|"lateral_movement"|"queue_management"|
                     "insider_threat". Team: "team_phishing_escalation"|"team_lateral_team".
            seed: RNG seed for deterministic scenario generation.
            mode: "tier1_solo" (backward compat) or "team".
        """
        self._mode = EpisodeMode(mode)
        if task_id == "red_team_generated":
            if self._generated_scenario is None:
                raise ValueError("No generated scenario loaded. Call /generate_scenario first.")
            self._config = self._generated_scenario.model_copy(deep=True)
        else:
            if task_id not in SCENARIO_REGISTRY:
                valid = list(SCENARIO_REGISTRY.keys()) + ["red_team_generated"]
                raise ValueError(f"Unknown task_id '{task_id}'. Valid: {valid}")
            generator_cls = SCENARIO_REGISTRY[task_id]
            generator = generator_cls(seed=seed)
            self._config = generator.generate()

        self._investigations = {
            alert.alert_id: InvestigationState(alert_id=alert.alert_id)
            for alert in self._config.alerts
        }

        self._cumulative_reward = 0.0
        self._step = 0
        self._done = False
        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())[:8]
        self._action_history = []
        self._tickets = []
        self._containment_results = []
        self._manager_review = None
        self._escalated_alert_ids = []
        self._prev_team_shared = 0.0
        self._tier1_reward = 0.0
        self._tier2_reward = 0.0
        self._manager_reward = 0.0

        if self._mode == EpisodeMode.TEAM:
            self._phase = EpisodePhase.TRIAGE
            self._phase_step = 0
        else:
            self._phase = None
            self._phase_step = 0

        active_role = AgentRole.TIER1 if self._mode == EpisodeMode.TEAM else None
        msg = (
            f"Episode started. Task: {task_id}. Mode: {mode}. "
            f"{len(self._config.alerts)} alert(s). Budget: {(_TEAM_MAX_STEPS if self._mode == EpisodeMode.TEAM else self._config.max_steps)} steps."
        )
        if self._mode == EpisodeMode.TEAM:
            budget = _PHASE_BUDGETS[EpisodePhase.TRIAGE]
            msg += f" Team mode: Tier-1 phase begins ({budget} steps)."

        return self._build_observation(
            role=active_role,
            reward=0.0,
            message=msg,
        )

    def set_generated_scenario(self, scenario: ScenarioConfig) -> None:
        """Load a generated scenario for reset(task_id='red_team_generated')."""
        self._generated_scenario = scenario.model_copy(deep=True)

    def step(self, action: SOCAction) -> SOCObservation:
        """Execute an action and return the updated observation."""
        if self._config is None:
            return self._build_observation(role=None, reward=-0.05, message="Error: call /reset before /step.")

        if self._done:
            return self._build_observation(role=None, reward=0.0, message="Episode done. Call /reset.")

        self._step += 1

        # Determine acting role
        acting_role = self._current_role()

        # In team mode validate role from action
        if self._mode == EpisodeMode.TEAM and action.role is not None:
            if action.role != acting_role:
                return self._build_observation(
                    role=acting_role,
                    reward=-0.03,
                    message=f"Wrong role. Expected {acting_role.value}, got {action.role.value}. Action ignored.",
                )

        # Phase complete / submit trigger phase advance
        if action.action_type == ActionType.PHASE_COMPLETE:
            return self._advance_phase(acting_role)
        if self._mode == EpisodeMode.TEAM and action.action_type == ActionType.SUBMIT_INVESTIGATION:
            return self._finalize_team_episode()

        # Loop detection
        action_sig = self._action_signature(action)
        loop_penalty = 0.0
        repeat_count = self._action_history.count(action_sig)
        if repeat_count >= 2:
            loop_penalty = -0.05 * min(repeat_count - 1, 4)
        self._action_history.append(action_sig)

        # Dispatch
        result = self._dispatch(action, acting_role)
        role_specific_reward = result["reward"]
        team_shared_reward = 0.0
        if self._mode == EpisodeMode.TEAM and acting_role is not None:
            team_shared_reward, _, _ = compute_team_metrics(self._config, self._investigations)
            team_shared_delta = team_shared_reward - self._prev_team_shared
            self._prev_team_shared = team_shared_reward
            step_reward = (0.6 * role_specific_reward) + (0.4 * team_shared_delta) + loop_penalty
        else:
            step_reward = role_specific_reward + loop_penalty
        if loop_penalty < 0:
            result["message"] += f" [Loop penalty: {loop_penalty:.2f}]"
        if self._mode == EpisodeMode.TEAM and acting_role is not None:
            result["message"] += (
                f" [Blended reward: role={role_specific_reward:.3f}, "
                f"team={team_shared_reward:.3f}]"
            )

        # Track per-role reward (team mode)
        team_rb = self._accumulate_team_reward(step_reward, acting_role, team_shared_reward)
        self._cumulative_reward += step_reward

        # Phase step budget exhaustion (team mode)
        if self._mode == EpisodeMode.TEAM and self._phase is not None:
            self._phase_step += 1
            budget = _PHASE_BUDGETS.get(self._phase, 99999)
            if self._phase_step >= budget and self._phase != EpisodePhase.COMPLETE:
                phase_obs = self._advance_phase(acting_role)
                phase_obs.reward += step_reward
                phase_obs.cumulative_reward = self._cumulative_reward
                phase_obs.message = result["message"] + " | " + phase_obs.message
                phase_obs.team_reward_breakdown = team_rb
                return phase_obs

        # Solo mode: budget exhaustion
        if self._mode == EpisodeMode.TIER1_SOLO:
            if self._step >= self._config.max_steps and not self._done:
                self._done = True
                auto = self._auto_grade_on_timeout()
                self._cumulative_reward += auto
                result["message"] += f" | Budget exhausted. Auto-graded: {auto:.3f}"

        return self._build_observation(
            role=acting_role,
            reward=step_reward,
            enrichment_results=result.get("enrichment_results", []),
            log_results=result.get("log_results", []),
            asset_info=result.get("asset_info"),
            user_info=result.get("user_info"),
            correlated_events=result.get("correlated_events", []),
            containment_results=result.get("containment_results", []),
            manager_review_result=result.get("manager_review_result"),
            message=result["message"],
            task_score=result.get("task_score"),
            team_reward_breakdown=team_rb,
        )

    def grade(self) -> float:
        """Run grader on current state without terminating."""
        if self._config is None or self._task_id is None:
            return 0.0
        grader_cls = GRADER_REGISTRY.get(self._task_id)
        if grader_cls is None:
            return 0.0
        return grader_cls().grade(
            config=self._config,
            investigations=self._investigations,
            steps_used=self._step,
            max_steps=self._config.max_steps,
        )

    def grade_with_breakdown(self) -> tuple:
        """Run grader and return (score, breakdown_dict, feedback_str)."""
        if self._config is None or self._task_id is None:
            return 0.0, {}, "No active episode."
        grader_cls = GRADER_REGISTRY.get(self._task_id)
        if grader_cls is None:
            return 0.0, {}, "No grader registered."
        return grader_cls().grade_with_breakdown(
            config=self._config,
            investigations=self._investigations,
            steps_used=self._step,
            max_steps=self._config.max_steps,
        )

    def state(self) -> EnvironmentState:
        """Return current episode state metadata."""
        classified = sum(
            1 for inv in self._investigations.values() if inv.classification is not None
        )
        return EnvironmentState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step,
            max_steps=_TEAM_MAX_STEPS if self._mode == EpisodeMode.TEAM else (self._config.max_steps if self._config else 0),
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            alert_count=len(self._config.alerts) if self._config else 0,
            classified_count=classified,
            seed=self._config.seed if self._config else None,
            metadata={
                "scenario_id": self._config.scenario_id if self._config else None,
                "description": self._config.description if self._config else None,
                "mode": self._mode.value,
            },
            episode_mode=self._mode,
            current_phase=self._phase,
            current_role=self._current_role(),
        )

    # ------------------------------------------------------------------
    # Phase Management (team mode)
    # ------------------------------------------------------------------

    def _current_role(self) -> AgentRole | None:
        """Return which role should act next."""
        if self._mode == EpisodeMode.TIER1_SOLO:
            return None
        phase_to_role = {
            EpisodePhase.TRIAGE: AgentRole.TIER1,
            EpisodePhase.RESPONSE: AgentRole.TIER2,
            EpisodePhase.OVERSIGHT: AgentRole.MANAGER,
            EpisodePhase.COMPLETE: None,
        }
        return phase_to_role.get(self._phase)

    def _advance_phase(self, current_role: AgentRole | None) -> SOCObservation:
        """Advance to the next episode phase and return updated observation."""
        if self._phase == EpisodePhase.TRIAGE:
            if len(self._escalated_alert_ids) == 0:
                self._phase = EpisodePhase.COMPLETE
                self._cumulative_reward += -0.10
                msg = (
                    "Triage phase complete with zero escalations — no work for Tier-2/Manager. "
                    "Episode short-circuited with -0.10 penalty."
                )
                finalize = self._finalize_team_episode()
                finalize.message = msg + " | " + finalize.message
                return finalize
            self._phase = EpisodePhase.RESPONSE
            self._phase_step = 0
            budget = _PHASE_BUDGETS[EpisodePhase.RESPONSE]
            msg = f"Triage phase complete. Tier-2 response phase begins ({budget} steps). {len(self._escalated_alert_ids)} alert(s) escalated."
            return self._build_observation(role=AgentRole.TIER2, reward=0.0, message=msg)

        elif self._phase == EpisodePhase.RESPONSE:
            self._phase = EpisodePhase.OVERSIGHT
            self._phase_step = 0
            budget = _PHASE_BUDGETS[EpisodePhase.OVERSIGHT]
            msg = f"Response phase complete. Manager oversight phase begins ({budget} steps)."
            return self._build_observation(role=AgentRole.MANAGER, reward=0.0, message=msg)

        elif self._phase == EpisodePhase.OVERSIGHT:
            # Final grading
            return self._finalize_team_episode()

        return self._build_observation(role=None, reward=0.0, message="Episode already complete.")

    def _finalize_team_episode(self) -> SOCObservation:
        """Grade team episode and mark done."""
        self._phase = EpisodePhase.COMPLETE
        self._done = True

        # Individual grader (task-specific)
        grader_cls = GRADER_REGISTRY.get(self._task_id)
        individual_score = 0.5
        if grader_cls:
            individual_score = grader_cls().grade(
                config=self._config,
                investigations=self._investigations,
                steps_used=self._step,
                max_steps=_TEAM_MAX_STEPS,
            )

        # Team shaped score
        team_score, _, _ = compute_team_metrics(self._config, self._investigations)

        # Final blended score: 60% individual, 40% team
        final_score = max(0.001, min(0.999, 0.6 * individual_score + 0.4 * team_score))
        self._cumulative_reward += final_score

        breakdown = TeamRewardBreakdown(
            tier1_individual=round(self._tier1_reward, 4),
            tier2_individual=round(self._tier2_reward, 4),
            manager_individual=round(self._manager_reward, 4),
            team_shared=round(team_score, 4),
            total=round(final_score, 4),
        )

        msg = (
            f"Team episode complete. Individual: {individual_score:.3f} | "
            f"Team score: {team_score:.3f} | Final: {final_score:.3f} | "
            f"Cumulative: {self._cumulative_reward:.3f}"
        )
        return self._build_observation(
            role=None,
            reward=final_score,
            message=msg,
            task_score=final_score,
            team_reward_breakdown=breakdown,
        )

    def _accumulate_team_reward(
        self,
        reward: float,
        role: AgentRole | None,
        team_shared_reward: float,
    ) -> TeamRewardBreakdown | None:
        """Track per-role rewards and return current snapshot."""
        if self._mode != EpisodeMode.TEAM or role is None:
            return None
        if role == AgentRole.TIER1:
            self._tier1_reward += reward
        elif role == AgentRole.TIER2:
            self._tier2_reward += reward
        elif role == AgentRole.MANAGER:
            self._manager_reward += reward
        return TeamRewardBreakdown(
            tier1_individual=round(self._tier1_reward, 4),
            tier2_individual=round(self._tier2_reward, 4),
            manager_individual=round(self._manager_reward, 4),
            team_shared=round(team_shared_reward, 4),
            total=round(self._cumulative_reward, 4),
        )

    # ------------------------------------------------------------------
    # Action Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: SOCAction, acting_role: AgentRole | None) -> dict:
        """Route action to appropriate handler based on action_type."""
        # Validate role/action compatibility in team mode
        if self._mode == EpisodeMode.TEAM and acting_role is not None:
            allowed = {
                AgentRole.TIER1: _TIER1_ACTIONS,
                AgentRole.TIER2: _TIER2_ACTIONS,
                AgentRole.MANAGER: _MANAGER_ACTIONS,
            }.get(acting_role, set())
            if action.action_type not in allowed:
                return {
                    "reward": -0.03,
                    "message": f"Action {action.action_type.value} not allowed for role {acting_role.value}.",
                }

        match action.action_type:
            case ActionType.ENRICH_INDICATOR:
                return self._handle_enrich(action)
            case ActionType.QUERY_LOGS:
                return self._handle_query_logs(action)
            case ActionType.CORRELATE_ALERTS:
                return self._handle_correlate(action)
            case ActionType.CHECK_ASSET:
                return self._handle_check_asset(action)
            case ActionType.CHECK_USER:
                return self._handle_check_user(action)
            case ActionType.CLASSIFY_ALERT:
                return self._handle_classify(action)
            case ActionType.MAP_TECHNIQUE:
                return self._handle_map_technique(action)
            case ActionType.RECOMMEND_ACTION:
                return self._handle_recommend_action(action)
            case ActionType.ESCALATE:
                return self._handle_escalate(action)
            case ActionType.ESCALATE_TO_TIER2:
                return self._handle_escalate_to_tier2(action)
            case ActionType.SUBMIT_INVESTIGATION:
                return self._handle_submit()
            # --- Tier-2 actions ---
            case ActionType.FORENSIC_TIMELINE:
                return self._handle_forensic_timeline(action)
            case ActionType.SANDBOX_DETONATE:
                return self._handle_sandbox_detonate(action)
            case ActionType.MEMORY_ANALYSIS:
                return self._handle_memory_analysis(action)
            case ActionType.ISOLATE_HOST:
                return self._handle_isolate_host(action)
            case ActionType.DISABLE_USER:
                return self._handle_disable_user(action)
            case ActionType.BLOCK_IOC:
                return self._handle_block_ioc(action)
            case ActionType.CLOSE_CASE:
                return self._handle_close_case(action)
            # --- Manager actions ---
            case ActionType.REVIEW_DECISION:
                return self._handle_review_decision(action)
            case ActionType.OVERRIDE_CLASSIFICATION:
                return self._handle_override_classification(action)
            case ActionType.FLAG_INCONSISTENCY:
                return self._handle_flag_inconsistency(action)
            case ActionType.EXPLAIN_TEAM_BEHAVIOR:
                return self._handle_explain_team_behavior(action)
            case ActionType.NOOP:
                return {"reward": -0.01, "message": "No operation performed."}
            case _:
                return {"reward": -0.03, "message": f"Unknown action type: {action.action_type}"}

    # ------------------------------------------------------------------
    # Tier-1 Action Handlers (original)
    # ------------------------------------------------------------------

    def _handle_enrich(self, action: SOCAction) -> dict:
        if not action.indicator or not action.indicator_type:
            return {"reward": -0.03, "message": "enrich_indicator requires 'indicator' and 'indicator_type'."}
        alert_id = action.query_alert_id or self._infer_alert_id(action.indicator)
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.03, "message": "No active investigation found."}
        result, reward, message = enrich_indicator(self._config, inv, action.indicator, action.indicator_type)
        inv.enriched_indicators[action.indicator] = result
        inv.reward_breakdown[f"enrich_{action.indicator[:20]}"] = reward
        inv.evidence_timeline.append(f"Step {self._step}: Enriched {action.indicator_type.value} '{action.indicator}'")
        return {"reward": reward, "enrichment_results": [result], "message": message}

    def _handle_query_logs(self, action: SOCAction) -> dict:
        if not action.log_source:
            return {"reward": -0.03, "message": "query_logs requires 'log_source'."}
        alert_id = action.query_alert_id
        if not alert_id:
            alert_id = self._config.alerts[0].alert_id if self._config.alerts else None
        if not alert_id or alert_id not in self._investigations:
            return {"reward": -0.03, "message": f"Invalid alert_id '{alert_id}' for log query."}
        inv = self._investigations[alert_id]
        entries, reward, message = query_logs(self._config, inv, action.log_source, alert_id, action.time_window_hours or 24)
        inv.queried_sources[action.log_source.value] = entries
        inv.reward_breakdown[f"query_{action.log_source.value}_{alert_id[:8]}"] = reward
        inv.evidence_timeline.append(f"Step {self._step}: Queried {action.log_source.value} for {alert_id} — {len(entries)} entries")
        return {"reward": reward, "log_results": entries, "message": message}

    def _handle_correlate(self, action: SOCAction) -> dict:
        if not action.alert_id_a or not action.alert_id_b:
            return {"reward": -0.03, "message": "correlate_alerts requires 'alert_id_a' and 'alert_id_b'."}
        event, reward, message = correlate_alerts(self._config, self._investigations, action.alert_id_a, action.alert_id_b)
        if event:
            for aid in [action.alert_id_a, action.alert_id_b]:
                inv = self._investigations.get(aid)
                if inv:
                    existing = {tuple(sorted(e.alert_ids)) for e in inv.correlations_found}
                    if tuple(sorted(event.alert_ids)) not in existing:
                        inv.correlations_found.append(event)
                        inv.evidence_timeline.append(f"Step {self._step}: Correlated with {action.alert_id_b if aid == action.alert_id_a else action.alert_id_a}")
        return {"reward": reward, "correlated_events": [event] if event else [], "message": message}

    def _handle_check_asset(self, action: SOCAction) -> dict:
        if not action.hostname:
            return {"reward": -0.03, "message": "check_asset requires 'hostname'."}
        inv = self._get_most_relevant_investigation(hostname=action.hostname) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.03, "message": "No active investigation."}
        asset, reward, message = lookup_asset(self._config, inv, action.hostname)
        if asset:
            inv.assets_looked_up[action.hostname] = asset
            inv.evidence_timeline.append(f"Step {self._step}: Looked up asset '{action.hostname}'")
        return {"reward": reward, "asset_info": asset, "message": message}

    def _handle_check_user(self, action: SOCAction) -> dict:
        if not action.username:
            return {"reward": -0.03, "message": "check_user requires 'username'."}
        inv = self._get_most_relevant_investigation(username=action.username) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.03, "message": "No active investigation."}
        user, reward, message = lookup_user(self._config, inv, action.username)
        if user:
            inv.users_looked_up[action.username] = user
            inv.evidence_timeline.append(f"Step {self._step}: Looked up user '{action.username}'")
        return {"reward": reward, "user_info": user, "message": message}

    def _handle_classify(self, action: SOCAction) -> dict:
        if not action.alert_id or not action.classification:
            return {"reward": -0.03, "message": "classify_alert requires 'alert_id' and 'classification'."}
        if action.alert_id not in self._investigations:
            return {"reward": -0.03, "message": f"Alert '{action.alert_id}' not found."}
        inv = self._investigations[action.alert_id]
        evidence_count = (
            len(inv.enriched_indicators) + len(inv.queried_sources)
            + len(inv.correlations_found) + len(inv.assets_looked_up) + len(inv.users_looked_up)
        )
        if evidence_count < 1:
            reward = -0.10
            message = f"Classified {action.alert_id} as {action.classification.value} without evidence. (-0.10 penalty)"
        else:
            gt_class = self._config.ground_truth.alert_classifications.get(action.alert_id)
            if gt_class == action.classification:
                reward = 0.30
                message = f"Classified {action.alert_id} as {action.classification.value}. [Correct]"
            else:
                reward = -0.20
                message = f"Classified {action.alert_id} as {action.classification.value}. [Check your evidence]"
        inv.classification = action.classification
        inv.classification_confidence = action.confidence or 0.8
        inv.reward_breakdown[f"classify_{action.alert_id[:8]}"] = reward
        inv.evidence_timeline.append(f"Step {self._step}: Classified '{action.alert_id}' as {action.classification.value}")
        for alert in self._config.alerts:
            if alert.alert_id == action.alert_id:
                alert.classification = action.classification
                break
        if self._all_classified():
            message += " | All alerts classified. Call submit_investigation to finalize."
        return {"reward": reward, "message": message}

    def _handle_map_technique(self, action: SOCAction) -> dict:
        if not action.alert_id:
            action = action.model_copy(update={"alert_id": self._config.alerts[0].alert_id if self._config.alerts else None})
        if not action.alert_id or action.alert_id not in self._investigations:
            return {"reward": -0.02, "message": "map_technique requires a valid 'alert_id'."}
        if not action.technique_id:
            return {"reward": -0.02, "message": "map_technique requires 'technique_id' (e.g. T1566.001)."}
        inv = self._investigations[action.alert_id]
        if not is_valid_technique(action.technique_id):
            return {"reward": -0.02, "message": f"Unknown MITRE ATT&CK technique: '{action.technique_id}'."}
        if action.technique_id not in inv.mapped_techniques:
            inv.mapped_techniques.append(action.technique_id)
            inv.evidence_timeline.append(f"Step {self._step}: Mapped technique {action.technique_id} to {action.alert_id}")
        expected = self._config.ground_truth.expected_techniques.get(action.alert_id, [])
        if action.technique_id in expected:
            return {"reward": 0.05, "message": f"Mapped technique {action.technique_id} to {action.alert_id}. [Relevant]"}
        elif action.technique_id.split(".")[0] in [t.split(".")[0] for t in expected]:
            return {"reward": 0.02, "message": f"Mapped {action.technique_id}. [Related — consider sub-technique]"}
        return {"reward": -0.01, "message": f"Mapped technique {action.technique_id} to {action.alert_id}."}

    def _handle_recommend_action(self, action: SOCAction) -> dict:
        if not action.alert_id:
            action = action.model_copy(update={"alert_id": self._config.alerts[0].alert_id if self._config.alerts else None})
        if not action.alert_id or action.alert_id not in self._investigations:
            return {"reward": -0.02, "message": "recommend_action requires a valid 'alert_id'."}
        if not action.response_action:
            return {"reward": -0.02, "message": "recommend_action requires 'response_action'."}
        inv = self._investigations[action.alert_id]
        gt_class = self._config.ground_truth.alert_classifications.get(action.alert_id)
        expected_actions = set(self._config.ground_truth.expected_response_actions.get(action.alert_id, []))
        if action.response_action not in inv.recommended_actions:
            inv.recommended_actions.append(action.response_action)
            inv.evidence_timeline.append(f"Step {self._step}: Recommended {action.response_action.value} for {action.alert_id}")
        if action.response_action == ResponseActionType.NO_ACTION:
            if gt_class == AlertClassification.FALSE_POSITIVE:
                return {"reward": 0.05, "message": f"Recommended no_action for {action.alert_id}. [Correct for FP]"}
            return {"reward": -0.10, "message": "Recommended no_action for a true positive. [Insufficient]"}
        elif action.response_action in expected_actions:
            return {"reward": 0.08, "message": f"Recommended {action.response_action.value} for {action.alert_id}. [Appropriate]"}
        return {"reward": 0.02, "message": f"Recommended {action.response_action.value} for {action.alert_id}."}

    def _handle_escalate(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._config.alerts[0].alert_id if self._config.alerts else None)
        if not alert_id or alert_id not in self._investigations:
            return {"reward": -0.02, "message": "escalate requires a valid 'alert_id'."}
        inv = self._investigations[alert_id]
        gt_class = self._config.ground_truth.alert_classifications.get(alert_id)
        inv.escalated = True
        inv.escalation_severity = action.escalation_severity or "high"
        inv.escalation_justification = action.justification or ""
        if gt_class == AlertClassification.TRUE_POSITIVE:
            return {"reward": 0.05, "message": f"Escalated {alert_id}. [Appropriate for TP]"}
        elif gt_class == AlertClassification.FALSE_POSITIVE:
            return {"reward": -0.10, "message": f"Escalated {alert_id} — appears to be FP. [Incorrect escalation]"}
        return {"reward": 0.02, "message": f"Escalated {alert_id} for review."}

    def _handle_escalate_to_tier2(self, action: SOCAction) -> dict:
        """Team-mode Tier-1 escalation to Tier-2 via ticket."""
        alert_id = action.alert_id or (self._config.alerts[0].alert_id if self._config.alerts else None)
        if not alert_id or alert_id not in self._investigations:
            return {"reward": -0.02, "message": "escalate_to_tier2 requires a valid 'alert_id'."}

        # Enforce over-escalation penalty: if >30% of alerts escalated by T1, penalise
        escalated_count = len(self._escalated_alert_ids)
        total_alerts = len(self._config.alerts)
        prospective_count = escalated_count + (1 if alert_id not in self._escalated_alert_ids else 0)
        if prospective_count / max(total_alerts, 1) > 0.25 and alert_id not in self._escalated_alert_ids:
            penalty = -0.08
        else:
            penalty = 0.0

        inv = self._investigations[alert_id]
        inv.escalated = True
        inv.escalation_severity = action.escalation_severity or "high"
        inv.escalation_justification = action.justification or ""

        if alert_id not in self._escalated_alert_ids:
            self._escalated_alert_ids.append(alert_id)

        # Create escalation ticket
        ticket = TicketMessage(
            ticket_id=f"TKT-{str(uuid.uuid4())[:8]}",
            alert_id=alert_id,
            from_role=AgentRole.TIER1,
            to_role=AgentRole.TIER2,
            kind=TicketKind.ESCALATION,
            payload={
                "classification": inv.classification.value if inv.classification else "unclassified",
                "justification": action.justification or "",
                "evidence_count": len(inv.enriched_indicators) + len(inv.queried_sources),
                "severity": action.escalation_severity or "high",
            },
            step_created=self._step,
        )
        self._tickets.append(ticket)

        gt_class = self._config.ground_truth.alert_classifications.get(alert_id)
        required = self._config.ground_truth.required_escalations
        if alert_id in required:
            reward = 0.12 + penalty
            msg = f"Escalated {alert_id} to Tier-2 (ticket created). [Required escalation — correct]"
        elif gt_class == AlertClassification.TRUE_POSITIVE:
            reward = 0.06 + penalty
            msg = f"Escalated {alert_id} to Tier-2. [TP — good judgment]"
        elif gt_class == AlertClassification.FALSE_POSITIVE:
            reward = -0.10 + penalty
            msg = f"Escalated {alert_id} to Tier-2. [FP alert — unnecessary escalation]"
        else:
            reward = 0.03 + penalty
            msg = f"Escalated {alert_id} to Tier-2."
        return {"reward": reward, "message": msg}

    def _handle_submit(self) -> dict:
        """Run the grader and finalize the episode (solo mode or manager submit in team mode)."""
        if self._mode == EpisodeMode.TEAM:
            final_obs = self._finalize_team_episode()
            return {
                "reward": final_obs.reward,
                "task_score": final_obs.task_score,
                "message": final_obs.message,
                "team_reward_breakdown": final_obs.team_reward_breakdown,
            }
        grader_cls = GRADER_REGISTRY.get(self._task_id)
        if grader_cls is None:
            return {"reward": 0.0, "message": "No grader registered for this task."}
        raw_score = grader_cls().grade(
            config=self._config,
            investigations=self._investigations,
            steps_used=self._step,
            max_steps=self._config.max_steps,
        )
        final_reward = max(0.001, min(0.999, raw_score * self._efficiency_multiplier()))
        self._done = True
        self._cumulative_reward += final_reward
        return {
            "reward": final_reward,
            "task_score": final_reward,
            "message": f"Investigation submitted. Score: {raw_score:.3f} → {final_reward:.3f}. Total: {self._cumulative_reward:.3f}",
        }

    # ------------------------------------------------------------------
    # Tier-2 Action Handlers
    # ------------------------------------------------------------------

    def _handle_forensic_timeline(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_host = action.target_host or action.hostname
        if not target_host:
            return {"reward": -0.02, "message": "forensic_timeline requires 'target_host'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = forensic_timeline(self._config, inv, alert_id or inv.alert_id, target_host)
        inv.evidence_timeline.append(f"Step {self._step}: Forensic timeline for '{target_host}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_sandbox_detonate(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_ioc = action.target_ioc or action.indicator
        if not target_ioc:
            return {"reward": -0.02, "message": "sandbox_detonate requires 'target_ioc'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = sandbox_detonate(self._config, inv, alert_id or inv.alert_id, target_ioc)
        inv.evidence_timeline.append(f"Step {self._step}: Sandbox detonated '{target_ioc}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_memory_analysis(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_host = action.target_host or action.hostname
        if not target_host:
            return {"reward": -0.02, "message": "memory_analysis requires 'target_host'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = memory_analysis(self._config, inv, alert_id or inv.alert_id, target_host)
        inv.evidence_timeline.append(f"Step {self._step}: Memory analysis on '{target_host}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_isolate_host(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_host = action.target_host or action.hostname
        if not target_host:
            return {"reward": -0.02, "message": "isolate_host requires 'target_host'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = isolate_host(self._config, inv, alert_id or inv.alert_id, target_host)
        inv.evidence_timeline.append(f"Step {self._step}: Isolated host '{target_host}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_disable_user(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_user = action.target_user or action.username
        if not target_user:
            return {"reward": -0.02, "message": "disable_user requires 'target_user'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = disable_user_account(self._config, inv, alert_id or inv.alert_id, target_user)
        inv.evidence_timeline.append(f"Step {self._step}: Disabled user '{target_user}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_block_ioc(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        target_ioc = action.target_ioc or action.indicator
        ioc_type = action.ioc_type or (action.indicator_type.value if action.indicator_type else "ip")
        if not target_ioc:
            return {"reward": -0.02, "message": "block_ioc requires 'target_ioc'."}
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = block_ioc(self._config, inv, alert_id or inv.alert_id, target_ioc, ioc_type)
        inv.evidence_timeline.append(f"Step {self._step}: Blocked IOC '{target_ioc}'")
        self._containment_results.append(result)
        return {"reward": reward, "containment_results": [result], "message": message}

    def _handle_close_case(self, action: SOCAction) -> dict:
        alert_id = action.alert_id or (self._escalated_alert_ids[0] if self._escalated_alert_ids else None)
        resolution = action.justification or "Closed by Tier-2 responder."
        inv = self._investigations.get(alert_id) or self._get_any_investigation()
        if inv is None:
            return {"reward": -0.02, "message": "No investigation found."}
        result, reward, message = close_case(self._config, inv, alert_id or inv.alert_id, resolution)
        self._containment_results.append(result)
        # Create closure ticket for Manager
        ticket = TicketMessage(
            ticket_id=f"TKT-{str(uuid.uuid4())[:8]}",
            alert_id=alert_id or inv.alert_id,
            from_role=AgentRole.TIER2,
            to_role=AgentRole.MANAGER,
            kind=TicketKind.CLOSURE,
            payload={"resolution": resolution, "classification": inv.classification.value if inv.classification else "unclassified"},
            step_created=self._step,
        )
        self._tickets.append(ticket)
        return {"reward": reward, "containment_results": [result], "message": message}

    # ------------------------------------------------------------------
    # Manager Action Handlers
    # ------------------------------------------------------------------

    def _handle_review_decision(self, action: SOCAction) -> dict:
        ticket_id = action.ticket_id
        ticket = next((t for t in self._tickets if t.ticket_id == ticket_id), None)
        if ticket is None:
            # Allow review by alert_id directly
            alert_id = action.alert_id
            if not alert_id:
                return {"reward": -0.02, "message": "review_decision requires 'ticket_id' or 'alert_id'."}
            ticket = next((t for t in self._tickets if t.alert_id == alert_id), None)
        if ticket is None:
            return {"reward": -0.02, "message": f"No ticket found for ticket_id={ticket_id} or alert_id={action.alert_id}."}
        result, reward, message = review_decision(self._config, self._investigations, ticket)
        ticket.resolved = True
        return {"reward": reward, "manager_review_result": result, "message": message}

    def _handle_override_classification(self, action: SOCAction) -> dict:
        if not action.alert_id or not action.new_classification:
            return {"reward": -0.02, "message": "override_classification requires 'alert_id' and 'new_classification'."}
        if action.alert_id not in self._investigations:
            return {"reward": -0.02, "message": f"Alert '{action.alert_id}' not found."}
        result, reward, message = override_classification(
            self._config, self._investigations, action.alert_id, action.new_classification
        )
        # Create override ticket
        ticket = TicketMessage(
            ticket_id=f"TKT-{str(uuid.uuid4())[:8]}",
            alert_id=action.alert_id,
            from_role=AgentRole.MANAGER,
            to_role=AgentRole.TIER1,
            kind=TicketKind.OVERRIDE,
            payload={"new_classification": action.new_classification.value, "finding": result.finding},
            step_created=self._step,
        )
        self._tickets.append(ticket)
        return {"reward": reward, "manager_review_result": result, "message": message}

    def _handle_flag_inconsistency(self, action: SOCAction) -> dict:
        if not action.alert_id:
            return {"reward": -0.02, "message": "flag_inconsistency requires 'alert_id'."}
        result, reward, message = flag_inconsistency(
            self._config, self._investigations, action.alert_id, action.flag_reason or ""
        )
        ticket = TicketMessage(
            ticket_id=f"TKT-{str(uuid.uuid4())[:8]}",
            alert_id=action.alert_id,
            from_role=AgentRole.MANAGER,
            to_role=None,
            kind=TicketKind.FLAG,
            payload={"reason": action.flag_reason or "", "inconsistency_found": result.inconsistency_found},
            step_created=self._step,
        )
        self._tickets.append(ticket)
        return {"reward": reward, "manager_review_result": result, "message": message}

    def _handle_explain_team_behavior(self, action: SOCAction) -> dict:
        if not action.explanation_text:
            return {"reward": -0.02, "message": "explain_team_behavior requires 'explanation_text'."}
        result, reward, message = explain_team_behavior(
            self._config,
            self._investigations,
            self._tickets,
            action.explanation_text,
            self._episode_id or "unknown",
            self._config.seed if self._config else 42,
            self._trajectory_hash(),
        )
        return {"reward": reward, "manager_review_result": result, "message": message}

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        role: AgentRole | None,
        reward: float,
        enrichment_results=None,
        log_results=None,
        asset_info=None,
        user_info=None,
        correlated_events=None,
        containment_results=None,
        manager_review_result=None,
        message: str = "",
        task_score=None,
        team_reward_breakdown=None,
    ) -> SOCObservation:
        """Construct role-filtered SOCObservation from current state."""
        visible_investigations = self._investigations

        # Collect all correlations
        all_correlations = []
        seen_pairs = set()
        for inv in visible_investigations.values():
            for corr in inv.correlations_found:
                pair = tuple(sorted(corr.alert_ids))
                if pair not in seen_pairs:
                    all_correlations.append(corr)
                    seen_pairs.add(pair)

        # Role-filter alert queue (Tier-2 sees only escalated alerts in team mode)
        if self._mode == EpisodeMode.TEAM and role == AgentRole.TIER2:
            visible_alerts = [a for a in self._config.alerts if a.alert_id in self._escalated_alert_ids] if self._config else []
            visible_investigations = {
                alert_id: inv
                for alert_id, inv in self._investigations.items()
                if alert_id in self._escalated_alert_ids
            }
        else:
            visible_alerts = self._config.alerts if self._config else []
            visible_investigations = self._investigations

        # Role-filter tickets
        if self._mode == EpisodeMode.TEAM and role is not None:
            if role == AgentRole.MANAGER:
                role_tickets = list(self._tickets)
            elif role == AgentRole.TIER2:
                role_tickets = [
                    t for t in self._tickets
                    if t.kind == TicketKind.ESCALATION and (t.to_role == AgentRole.TIER2 or t.to_role is None)
                ]
            else:
                role_tickets = [
                    t for t in self._tickets
                    if (
                        t.to_role == AgentRole.TIER1
                        or t.to_role is None
                        or (t.kind == TicketKind.CLOSURE and t.from_role == AgentRole.TIER2)
                    )
                ]
        else:
            role_tickets = []

        # Phase info
        current_phase = self._phase if self._mode == EpisodeMode.TEAM else None
        phase_budget = _PHASE_BUDGETS.get(current_phase, 0) if current_phase else 0
        phase_steps_remaining = max(0, phase_budget - self._phase_step) if self._mode == EpisodeMode.TEAM else None
        consistency_stats = self._build_consistency_stats() if role == AgentRole.MANAGER else None
        total_budget_remaining = (
            max(0, _TEAM_MAX_STEPS - self._step)
            if self._mode == EpisodeMode.TEAM
            else max(0, (self._config.max_steps if self._config else 0) - self._step)
        )

        return SOCObservation(
            alert_queue=visible_alerts,
            investigations=visible_investigations,
            enrichment_results=enrichment_results or [],
            log_results=log_results or [],
            correlated_events=correlated_events if correlated_events is not None else all_correlations,
            asset_info=asset_info,
            user_info=user_info,
            investigation_budget=total_budget_remaining,
            step=self._step,
            done=self._done,
            reward=reward,
            cumulative_reward=self._cumulative_reward,
            message=message,
            task_id=self._task_id,
            episode_id=self._episode_id,
            task_score=task_score,
            episode_mode=self._mode,
            current_role=role,
            current_phase=current_phase,
            phase_steps_remaining=phase_steps_remaining,
            tickets=role_tickets,
            containment_results=containment_results or self._containment_results,
            manager_review_result=manager_review_result,
            team_reward_breakdown=team_reward_breakdown,
            consistency_stats=consistency_stats,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_alert_id(self, indicator: str) -> str | None:
        if not self._config:
            return None
        for alert in self._config.alerts:
            for vals in alert.indicators.values():
                if indicator in vals:
                    return alert.alert_id
        return self._config.alerts[0].alert_id if self._config.alerts else None

    def _get_any_investigation(self) -> InvestigationState | None:
        if self._investigations:
            return next(iter(self._investigations.values()))
        return None

    def _get_most_relevant_investigation(self, hostname=None, username=None) -> InvestigationState | None:
        if not self._config:
            return None
        for alert in self._config.alerts:
            if hostname and hostname in alert.indicators.get("hostname", []):
                return self._investigations.get(alert.alert_id)
            if username and username in alert.indicators.get("user", []):
                return self._investigations.get(alert.alert_id)
        return None

    def _all_classified(self) -> bool:
        return all(inv.classification is not None for inv in self._investigations.values())

    def _efficiency_multiplier(self) -> float:
        if not self._config:
            return 1.0
        ratio = self._step / self._config.max_steps
        if ratio <= 0.50:
            return 1.0
        if ratio <= 0.75:
            return 1.0
        if ratio <= 0.90:
            return 0.85
        return 0.70

    def _auto_grade_on_timeout(self) -> float:
        if not self._config:
            return 0.0
        gt = self._config.ground_truth
        missed_tps = sum(
            1 for aid in gt.true_positive_ids
            if self._investigations.get(aid, InvestigationState(alert_id=aid)).classification
            not in {AlertClassification.TRUE_POSITIVE, AlertClassification.BENIGN_TRUE_POSITIVE}
        )
        return min(0.0, -0.5 * missed_tps)

    def _action_signature(self, action: SOCAction) -> str:
        return (
            f"{action.action_type.value}|{action.indicator or ''}|"
            f"{action.log_source.value if action.log_source else ''}|"
            f"{action.alert_id or ''}|{action.query_alert_id or ''}|"
            f"{action.alert_id_a or ''}|{action.alert_id_b or ''}|"
            f"{action.hostname or ''}|{action.username or ''}|"
            f"{action.target_host or ''}|{action.target_user or ''}|{action.target_ioc or ''}"
        )

    def _trajectory_hash(self) -> str:
        """Stable hash of the current decision trail for judge caching."""
        payload = {
            "tickets": [ticket.model_dump(mode="json") for ticket in self._tickets],
            "investigations": {
                alert_id: investigation.model_dump(mode="json")
                for alert_id, investigation in sorted(self._investigations.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _build_consistency_stats(self) -> ConsistencyStats:
        """Build manager-visible historical consistency stats from the audit trail."""
        tickets_total = len(self._tickets)
        tickets_resolved = sum(1 for ticket in self._tickets if ticket.resolved)
        escalation_reviews = sum(
            1 for ticket in self._tickets
            if ticket.kind == TicketKind.ESCALATION and ticket.resolved
        )
        closure_reviews = sum(
            1 for ticket in self._tickets
            if ticket.kind == TicketKind.CLOSURE and ticket.resolved
        )
        overrides_total = sum(1 for ticket in self._tickets if ticket.kind == TicketKind.OVERRIDE)
        valid_flags = sum(
            1
            for ticket in self._tickets
            if ticket.kind == TicketKind.FLAG and ticket.payload.get("inconsistency_found")
        )
        invalid_flags = sum(
            1
            for ticket in self._tickets
            if ticket.kind == TicketKind.FLAG and not ticket.payload.get("inconsistency_found")
        )
        denominator = escalation_reviews + closure_reviews + valid_flags + invalid_flags
        consistency_rate = (
            (escalation_reviews + closure_reviews + valid_flags) / denominator
            if denominator
            else 0.0
        )
        return ConsistencyStats(
            tickets_total=tickets_total,
            tickets_resolved=tickets_resolved,
            escalation_reviews=escalation_reviews,
            closure_reviews=closure_reviews,
            overrides_total=overrides_total,
            valid_flags=valid_flags,
            invalid_flags=invalid_flags,
            consistency_rate=round(consistency_rate, 4),
        )
