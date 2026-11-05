"""
Prompts for training and evaluating LLM policies in team mode
=============================================================
Single source of truth for the system prompt and observation prompt used by
GRPO training (``train_grpo.py``), held-out eval
(``scripts/train_and_evaluate.py``) and the in-training eval policy.

What the original prompt got wrong (it produced a 0.001-scoring adapter):

* The observation showed alert IDs, titles, severity and IPs only. The policy
  could not see what it had already done (classifications, evidence,
  escalations) or the environment's feedback, so it was effectively memoryless.
* Only IP indicators were shown, so hostnames / users / domains needed by
  ``check_asset``, ``check_user`` and containment actions were invisible.
* The system prompt listed action names but no parameter names or allowed
  values, so the model had to guess field names and enum values.
* Training and eval passed different step counters (0- vs 1-based).

Budget: GRPO trains with ``max_prompt_length=512`` and TRL truncates longer
prompts from the left, i.e. the system prompt goes first. Prompts must stay
under that with margin; ``tests/test_prompts.py`` enforces a character budget
calibrated against the Qwen2.5 tokenizer. Allowed values are generated from the
``models`` enums so the prompt cannot drift from the API.
"""

from __future__ import annotations

import re

from models import AlertClassification, IndicatorType, LogSource, ResponseActionType

# Token budget is enforced via characters in tests (no tokenizer dependency).
MAX_ALERTS_SHOWN = 5
MAX_TITLE_CHARS = 48
MAX_MESSAGE_CHARS = 120
MAX_TICKETS_SHOWN = 4


def _values(enum_cls, exclude: tuple[str, ...] = ()) -> str:
    return "|".join(m.value for m in enum_cls if m.value not in exclude)


_CLASSES = _values(AlertClassification, exclude=("unclassified",))

_TIER1 = f"""You are the Tier-1 SOC analyst (Tier-2 contains, the Manager audits).
Per alert: gather evidence (enrich_indicator/query_logs), classify_alert, then escalate_to_tier2 if true_positive. When all alerts are handled: phase_complete.
Classifying without evidence, or repeating an identical action, is penalised.
Reply with ONE JSON object only. Actions (fields):
enrich_indicator (query_alert_id, indicator, indicator_type: {_values(IndicatorType)})
query_logs (query_alert_id, log_source: {_values(LogSource)})
classify_alert (alert_id, classification: {_CLASSES}, confidence 0-1)
escalate_to_tier2 (alert_id, justification)
map_technique (alert_id, technique_id) | correlate_alerts (alert_id_a, alert_id_b)
check_asset (hostname) | check_user (username) | phase_complete
Example: {{"action_type": "classify_alert", "alert_id": "ALT-1", "classification": "true_positive", "confidence": 0.9}}"""

_TIER2 = f"""You are the Tier-2 SOC responder. Work only escalated alerts (see Tickets): investigate, contain, then close_case. When all escalations are closed, phase_complete.
Reply with ONE JSON object and nothing else. Actions and fields:
forensic_timeline / memory_analysis / isolate_host: alert_id, target_host
sandbox_detonate: alert_id, target_ioc
disable_user: alert_id, target_user
block_ioc: alert_id, target_ioc, ioc_type ({_values(IndicatorType)})
close_case: alert_id, justification
phase_complete
Example: {{"action_type": "isolate_host", "alert_id": "ALT-1", "target_host": "WS-042"}}"""

_MANAGER = f"""You are the SOC Manager. Audit the team: review each ticket, override wrong classifications, flag real inconsistencies, explain the team's behaviour, then submit_investigation.
Reply with ONE JSON object and nothing else. Actions and fields:
review_decision: ticket_id, alert_id
override_classification: alert_id, new_classification ({_CLASSES})
flag_inconsistency: alert_id, flag_reason
explain_team_behavior: explanation_text
submit_investigation
Example: {{"action_type": "review_decision", "ticket_id": "TKT-EP0001", "alert_id": "ALT-1"}}"""

ROLE_SYSTEM_PROMPTS: dict[str, str] = {"tier1": _TIER1, "tier2": _TIER2, "manager": _MANAGER}

# Response actions are offered to tier-1 in solo mode only; keep the enum
# imported so a rename breaks loudly here.
_ = ResponseActionType

_BLENDED = re.compile(r"\s*\[Blended reward:[^\]]*\]")


def _last_result(obs: dict) -> str:
    msg = _BLENDED.sub("", obs.get("message") or "").strip()
    if len(msg) > MAX_MESSAGE_CHARS:
        msg = msg[: MAX_MESSAGE_CHARS - 1] + "…"
    return msg or "none"


def _iocs(alert: dict, role: str, limit: int = 3) -> str:
    # Hashes are long (~40 tokens each) and only Tier-2 can act on them.
    order = ["ip", "hostname", "user", "domain", "email", "url"]
    if role == "tier2":
        order.insert(1, "file_hash")
    parts = []
    indicators = alert.get("indicators") or {}
    for kind in order:
        for value in indicators.get(kind) or []:
            if len(parts) == limit:
                break
            parts.append(f"{kind}={value}")
    return " ".join(parts) or "none"


def _alert_done(role: str, inv: dict, tickets_by_alert: dict[str, list[dict]]) -> bool:
    cls = inv.get("classification")
    if role == "tier1":
        return cls is not None and (cls != "true_positive" or bool(inv.get("escalated")))
    if role == "tier2":
        if not inv.get("escalated"):
            return True  # not Tier-2's work
        return any(t.get("kind") == "closure" for t in tickets_by_alert.get(inv.get("alert_id"), []))
    return False  # the manager audits everything


def _status(inv: dict) -> str:
    """Only what has happened, so an untouched alert costs one token, not 25."""
    parts = []
    if inv.get("classification"):
        parts.append(f"class={inv['classification']}")
    if inv.get("escalated"):
        parts.append("escalated")
    if inv.get("enriched_indicators"):
        parts.append(f"enriched={len(inv['enriched_indicators'])}")
    if inv.get("queried_sources"):
        parts.append(f"logs={','.join(sorted(inv['queried_sources']))}")
    if inv.get("mapped_techniques"):
        parts.append(f"techniques={','.join(inv['mapped_techniques'])}")
    return " ".join(parts) or "new"


def format_obs_prompt(obs: dict, role: str, step: int | None = None) -> str:
    """Render an observation for ``role``.

    ``step`` is accepted for backward compatibility but the environment's own
    step counter is used, so training and eval prompts agree.
    """
    alerts = obs.get("alert_queue") or []
    invs = obs.get("investigations") or {}
    tickets = obs.get("tickets") or []
    tickets_by_alert: dict[str, list[dict]] = {}
    for t in tickets:
        tickets_by_alert.setdefault(t.get("alert_id"), []).append(t)

    env_step = obs.get("step", step if step is not None else 0)
    lines = [
        f"role={role} phase={obs.get('current_phase') or 'triage'} step={env_step} "
        f"phase_steps_left={obs.get('phase_steps_remaining', '?')}",
        f"Last result: {_last_result(obs)}",
    ]

    pending, done = [], []
    for a in alerts:
        inv = invs.get(a["alert_id"]) or {"alert_id": a["alert_id"]}
        (done if _alert_done(role, inv, tickets_by_alert) else pending).append((a, inv))

    lines.append(f"Alerts: {len(alerts)} total, {len(pending)} need action")
    for a, inv in pending[:MAX_ALERTS_SHOWN]:
        title = (a.get("title") or "")[:MAX_TITLE_CHARS]
        line = f'- {a["alert_id"]} {a.get("severity", "?")} "{title}" | {_status(inv)}'
        if role != "manager":  # the manager never acts on raw indicators
            line += f" | iocs: {_iocs(a, role)}"
        lines.append(line)
    if len(pending) > MAX_ALERTS_SHOWN:
        lines.append(
            f"(+{len(pending) - MAX_ALERTS_SHOWN} more pending: "
            + ",".join(a["alert_id"] for a, _ in pending[MAX_ALERTS_SHOWN:])
            + ")"
        )
    if done:
        lines.append(
            "Done: "
            + ", ".join(
                f"{a['alert_id']}={inv.get('classification') or 'n/a'}{'+escalated' if inv.get('escalated') else ''}"
                for a, inv in done
            )
        )

    if tickets:
        lines.append("Tickets:")
        for t in tickets[-MAX_TICKETS_SHOWN:]:
            lines.append(
                f"- {t.get('ticket_id')} {t.get('kind')} alert={t.get('alert_id')} "
                f"{'resolved' if t.get('resolved') else 'open'}"
            )
    lines.append("Reply with one JSON action.")
    return "\n".join(lines)


class PromptBudgetError(ValueError):
    """A training prompt would be truncated by the trainer."""


def _token_count(tokenizer, messages: list[dict]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):  # BatchEncoding on newer transformers
        ids = ids["input_ids"]
    return len(ids)


def check_prompt_budget(rows: list[dict], tokenizer, max_prompt_length: int) -> int:
    """Tokenise every dataset prompt with the real tokenizer; fail if any is too long.

    TRL truncates over-long prompts from the *left*, so the first thing lost is
    the system prompt (action schemas included), silently. This turns that
    into a loud error at dataset-build time. Returns the longest prompt length.
    """
    lengths = [_token_count(tokenizer, row["prompt"]) for row in rows]
    longest = max(lengths, default=0)
    over = sum(1 for n in lengths if n > max_prompt_length)
    if over:
        raise PromptBudgetError(
            f"{over}/{len(lengths)} prompts exceed max_prompt_length={max_prompt_length} "
            f"(longest {longest} tokens). Raise max_prompt_length or tighten training/prompts.py."
        )
    return longest
