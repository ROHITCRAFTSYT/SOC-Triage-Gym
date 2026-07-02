"""
Manager Judge
=============
LLM-based judge for evaluating the quality of a SOC Manager agent's
written explanation of their team's security investigation.

Scoring rubric (weighted average):
  accuracy       0.4  — correct summary of alert IDs and decisions
  reasoning      0.3  — evidence-based logical chain
  actionability  0.3  — concrete gaps / next steps identified

Falls back to a keyword heuristic when the OpenAI API is unavailable
(network error, missing dependency, missing API key, parse failure).

Settings are read from environment variables:
  API_BASE_URL   — optional custom base URL (e.g. for proxies / HF endpoints)
  HF_TOKEN       — tried first as the API key
  OPENAI_API_KEY — fallback API key
  MODEL_NAME     — model to use (default: gpt-4o-mini)
"""

import hashlib
import json
import os

try:
    import openai as _openai_lib  # noqa: F401 — presence check only
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ScenarioConfig is used for type hints; import lazily to avoid circulars
try:
    from models import ScenarioConfig
except ImportError:
    ScenarioConfig = None  # type: ignore[assignment, misc]


class ManagerJudge:
    """
    LLM judge for Manager explanation quality.
    Temperature 0, seeded, with an in-process result cache keyed on
    (episode_id, first 200 chars of explanation).
    """

    JUDGE_PROMPT = """You are grading a SOC Manager AI agent's explanation of their team's security investigation.

# Investigation context
{investigation_summary}

# Manager's explanation
{explanation}

# Grading rubric (score each 0.0-1.0)
1. Accuracy (weight 0.4): Does the explanation correctly describe what the team found and decided?
   - 1.0 = accurate summary with specific alert IDs and decisions
   - 0.5 = mostly correct but missing details
   - 0.0 = incorrect or contradicts the actual decisions
2. Reasoning Quality (weight 0.3): Is the reasoning sound and evidence-based?
   - 1.0 = specific evidence cited, logical chain clear
   - 0.5 = reasonable but generic
   - 0.0 = no reasoning or hallucinated facts
3. Actionability (weight 0.3): Does the explanation identify gaps or next steps?
   - 1.0 = concrete gaps identified with specific recommendations
   - 0.5 = vague but reasonable suggestions
   - 0.0 = no actionable content

Respond with ONLY valid JSON: {{"accuracy": <float>, "reasoning": <float>, "actionability": <float>, "explanation": "<one sentence why>"}}
Default to 0.5 when uncertain. A score of 1.0 should be rare."""

    def __init__(self) -> None:
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(
        self,
        explanation: str,
        investigations: dict,
        config: "ScenarioConfig",
        episode_id: str,
        seed: int,
        trajectory_hash: str,
    ) -> float:
        """
        Score a Manager's explanation string.

        Args:
            explanation: The Manager agent's free-text explanation.
            investigations: Dict of alert_id → InvestigationState (or plain dicts).
            config: ScenarioConfig with ground_truth.
            episode_id: Unique episode identifier used for cache keying.

        Returns:
            Score in (0.001, 0.999).
        """
        cache_key = hashlib.sha256(
            f"{seed}:{episode_id}:{trajectory_hash}".encode()
        ).hexdigest()[:16]

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            score = self._call_llm(explanation, investigations, config, seed)
        except json.JSONDecodeError:
            score = 0.0
        except Exception:
            score = self.heuristic_score(explanation, investigations)

        self._cache[cache_key] = score
        return score

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        explanation: str,
        investigations: dict,
        config: "ScenarioConfig",
        seed: int,
    ) -> float:
        """Call the OpenAI-compatible API and parse the JSON response."""
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")

        import openai  # local import — guarded by availability check above

        base_url = os.environ.get("API_BASE_URL")
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        # HF_TOKEN is NOT a valid OpenAI key — only use it when API_BASE_URL
        # is explicitly set (indicating an HF-compatible endpoint).
        if base_url:
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("No API key for manager judge; using heuristic fallback.")

        client_kwargs: dict = {"api_key": api_key, "timeout": 5.0}
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            client = openai.OpenAI(**client_kwargs)
        except Exception as exc:
            raise RuntimeError(f"OpenAI client init failed: {exc}")

        investigation_summary = self._build_investigation_summary(
            investigations, config
        )

        prompt = self.JUDGE_PROMPT.format(
            investigation_summary=investigation_summary,
            explanation=explanation,
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            seed=seed,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.choices[0].message.content or ""
        parsed = json.loads(raw_text.strip())

        accuracy = float(parsed.get("accuracy", 0.5))
        reasoning = float(parsed.get("reasoning", 0.5))
        actionability = float(parsed.get("actionability", 0.5))

        weighted = 0.4 * accuracy + 0.3 * reasoning + 0.3 * actionability
        return self._clamp(weighted)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def heuristic_score(self, explanation: str, investigations: dict) -> float:
        """
        Keyword-based fallback scorer used when the LLM API is unavailable.

        Scoring:
          +0.05 per alert_id mentioned in explanation (max +0.20)
          +0.05 if "escalat" appears in explanation
          +0.05 if "contain" or "isolat" appears in explanation
          +0.05 if len(explanation) > 150

        Returns:
            Score in (0.001, 0.999).
        """
        score = 0.0
        exp_lower = explanation.lower()

        # Alert ID mentions (capped at 4 = +0.20)
        mention_count = 0
        for alert_id in investigations:
            if str(alert_id).lower() in exp_lower:
                mention_count += 1
                if mention_count >= 4:
                    break
        score += mention_count * 0.05

        # Escalation signal
        if "escalat" in exp_lower:
            score += 0.05

        # Containment / isolation signal
        if "contain" in exp_lower or "isolat" in exp_lower:
            score += 0.05

        # Length signal
        if len(explanation) > 150:
            score += 0.05

        return self._clamp(score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_investigation_summary(
        self, investigations: dict, config: "ScenarioConfig"
    ) -> str:
        """
        Build a compact summary of the first 5 alerts for the judge prompt.
        Each line: <alert_id>: <classification>
        """
        lines = []
        alert_ids = list(investigations.keys())[:5]
        for alert_id in alert_ids:
            inv = investigations[alert_id]
            # Support both InvestigationState objects and plain dicts
            if hasattr(inv, "classification"):
                cls = str(inv.classification.value if inv.classification else "unclassified")
            else:
                cls = str(inv.get("classification", "unclassified"))
            lines.append(f"  {alert_id}: {cls}")

        if not lines:
            return "  (no alerts investigated)"
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp to strictly (0.001, 0.999) as required by the OpenEnv validator."""
        return max(0.001, min(0.999, value))
