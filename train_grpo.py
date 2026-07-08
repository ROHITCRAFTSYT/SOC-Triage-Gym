"""
SOC-Triage-Gym v2 — GRPO Training Script
==========================================
Trains a Tier-1 SOC analyst LLM on team-mode episodes using GRPO.

Hackathon minimum: run this script (or the Colab notebook) to show
reward improvement curves. Uses HF TRL's GRPOTrainer.

Usage:
    # Tier-1 (primary) training on team tasks
    python train_grpo.py --role tier1 --model Qwen/Qwen2.5-1.5B-Instruct

    # Tier-2 training (with frozen trained Tier-1 as oracle)
    python train_grpo.py --role tier2 --model Qwen/Qwen2.5-1.5B-Instruct

    # Manager training
    python train_grpo.py --role manager --model Qwen/Qwen2.5-1.5B-Instruct

Environment variables:
    SERVER_URL      SOC-Triage-Gym server URL (default: http://localhost:7860)
    HF_TOKEN        Hugging Face token for gated models
    WANDB_API_KEY   Optional W&B key for logging reward curves

Training plan (onsite):
    Day 1 AM: Tier-1, frozen scripted T2+Manager, group=8, ~4h H100
    Day 1 PM: Tier-2, frozen trained T1
    Day 2 AM: Manager, cheapest (small action space)
    Day 2 PM: Joint fine-tune pass; Red-Team GRPO in parallel
"""

import argparse
import json
import os
import random
import sys

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
HF_TOKEN = os.getenv("HF_TOKEN")

# Windows consoles default to cp1252 and crash on the Unicode glyphs printed
# below; force UTF-8 so output is identical everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Team tasks ordered by complexity — train easier first.
# Overridable via SOC_TRAIN_TASKS env var (comma-separated) for quick-mode training
# on T4 where the full 2-task curriculum would take >10 hours per role.
_default_tier1 = ["team_phishing_escalation", "team_lateral_team"]
_env_tasks = os.getenv("SOC_TRAIN_TASKS")
if _env_tasks:
    TIER1_TASKS = [t.strip() for t in _env_tasks.split(",") if t.strip()]
else:
    TIER1_TASKS = _default_tier1
TIER2_TASKS = _default_tier1
MANAGER_TASKS = _default_tier1
# Seeds overridable via SOC_TRAIN_N_SEEDS (e.g. =15 for quick mode).
_n_seeds = int(os.getenv("SOC_TRAIN_N_SEEDS", "50"))
SEEDS = list(range(42, 42 + _n_seeds))

# Per-role system prompts (must match inference.py)
ROLE_SYSTEM_PROMPTS = {
    "tier1": (
        "You are a Tier-1 SOC Analyst. Triage incoming security alerts quickly and accurately.\n"
        "Respond with a JSON action. Available actions: enrich_indicator, query_logs, correlate_alerts,\n"
        "check_asset, check_user, classify_alert, map_technique, recommend_action,\n"
        "escalate_to_tier2, phase_complete, noop.\n"
        "Always include: {\"action_type\": \"...\", \"role\": \"tier1\", ...}\n"
        "Escalate to Tier-2 only confirmed true positives needing containment."
    ),
    "tier2": (
        "You are a Tier-2 SOC Responder. Investigate escalated tickets and contain confirmed threats.\n"
        "Respond with a JSON action. Available actions: forensic_timeline, sandbox_detonate,\n"
        "memory_analysis, isolate_host, disable_user, block_ioc, close_case, phase_complete, noop.\n"
        "Always include: {\"action_type\": \"...\", \"role\": \"tier2\", ...}"
    ),
    "manager": (
        "You are a SOC Manager. Review team decisions, catch inconsistencies, and explain behavior.\n"
        "Respond with a JSON action. Available actions: review_decision, override_classification,\n"
        "flag_inconsistency, explain_team_behavior, phase_complete, noop.\n"
        "Always include: {\"action_type\": \"...\", \"role\": \"manager\", ...}"
    ),
}


# ---------------------------------------------------------------------------
# Heuristic oracle actions (frozen baseline for non-trained roles)
# ---------------------------------------------------------------------------

def oracle_action(obs: dict) -> dict:
    """Scripted oracle for frozen roles during staged training."""
    role = obs.get("current_role") or "tier1"
    phase = obs.get("current_phase") or "triage"
    alerts = obs.get("alert_queue", [])
    invs = obs.get("investigations") or {}

    if phase == "triage" or role == "tier1":
        # Pass 1: enrich then classify each unclassified alert
        for alert in alerts:
            aid = alert["alert_id"]
            inv = invs.get(aid, {})
            if inv.get("classification"):
                continue
            ips = list(alert.get("indicators", {}).get("ip", []))
            # enriched_indicators is a dict keyed by indicator value
            already_enriched = inv.get("enriched_indicators") or {}
            if ips and ips[0] not in already_enriched:
                return {"action_type": "enrich_indicator", "indicator": ips[0],
                        "indicator_type": "ip", "query_alert_id": aid, "role": "tier1"}
            cls = "true_positive" if alert.get("severity") in ("high", "critical") else "false_positive"
            return {"action_type": "classify_alert", "alert_id": aid,
                    "classification": cls, "confidence": 0.85, "role": "tier1"}
        # Pass 2: escalate all confirmed TPs (env handles over-escalation penalty)
        for alert in alerts:
            aid = alert["alert_id"]
            inv = invs.get(aid, {})
            if inv.get("classification") == "true_positive" and not inv.get("escalated"):
                return {"action_type": "escalate_to_tier2", "alert_id": aid,
                        "justification": "TP confirmed via enrichment", "role": "tier1"}
        return {"action_type": "phase_complete", "role": "tier1"}

    elif phase == "response" or role == "tier2":
        tickets = obs.get("tickets") or []
        handled = set()
        for ticket in tickets:
            if ticket.get("kind") != "escalation":
                continue
            aid = ticket.get("alert_id")
            if not aid or aid in handled:
                continue
            inv = invs.get(aid, {})
            # Try to find a real TP host from the alert queue
            host = None
            for a in alerts:
                if a["alert_id"] == aid:
                    hosts = list(a.get("indicators", {}).get("hostname", []))
                    if hosts:
                        host = hosts[0]
                    break
            host = host or "WORKSTATION-ALPHA"
            already_closed = any(
                e.startswith("Case closed:") for e in (inv.get("evidence_timeline") or [])
            )
            if not already_closed:
                handled.add(aid)
                return {"action_type": "close_case", "alert_id": aid,
                        "justification": f"Contained host {host}", "role": "tier2"}
        return {"action_type": "phase_complete", "role": "tier2"}

    else:  # oversight / manager
        tickets = obs.get("tickets") or []
        reviewed = set()
        for ticket in tickets:
            aid = ticket.get("alert_id")
            if aid and aid not in reviewed:
                reviewed.add(aid)
                return {"action_type": "review_decision", "alert_id": aid,
                        "ticket_id": ticket.get("ticket_id", ""), "role": "manager"}
        alerts_summary = ", ".join(
            f"{a['alert_id']}({(invs.get(a['alert_id']) or {}).get('classification','?')})"
            for a in alerts[:3]
        )
        return {"action_type": "explain_team_behavior",
                "explanation_text": (
                    f"Team triaged {len(alerts)} alerts. Escalated confirmed true positives. "
                    f"Tier-2 contained threats and closed cases. "
                    f"Alert decisions: {alerts_summary}. No inconsistencies observed."
                ),
                "role": "manager"}


# ---------------------------------------------------------------------------
# Environment interaction
# ---------------------------------------------------------------------------

def run_episode(
    client: httpx.Client,
    task_id: str,
    seed: int,
    model_actions: list[dict] | None = None,
    role_to_train: str = "tier1",
    max_steps: int = 80,
) -> tuple[float, list[dict]]:
    """
    Run one full team episode.

    If model_actions is provided, replay them for the trained role.
    Otherwise use oracle for all roles.

    Returns (final_score, trajectory) where trajectory is list of
    {step, role, obs_summary, action, reward, done}.
    """
    reset_resp = client.post("/reset", json={"task_id": task_id, "seed": seed, "mode": "team"})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    trajectory = []
    action_cursor = 0
    step = 0

    while not obs.get("done", False) and step < max_steps:
        step += 1
        role = obs.get("current_role") or "tier1"

        if role == role_to_train and model_actions and action_cursor < len(model_actions):
            action = model_actions[action_cursor]
            action_cursor += 1
        else:
            action = oracle_action(obs)

        step_resp = client.post(
            "/step",
            content=json.dumps(action),
            headers={"Content-Type": "application/json"},
        )
        if step_resp.status_code != 200:
            break
        obs = step_resp.json()

        trajectory.append({
            "step": step,
            "role": role,
            "action": action.get("action_type"),
            "reward": obs.get("reward", 0.0),
            "done": obs.get("done", False),
        })

    final_score = obs.get("task_score") or obs.get("cumulative_reward", 0.0)
    return float(final_score), trajectory


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_prompt_dataset(
    client: httpx.Client,
    tasks: list[str],
    seeds: list[int],
    role: str,
) -> list[dict]:
    """
    Legacy initial-observation dataset builder (one prompt per episode).

    Kept for backward-compatibility with callers that expect a single prompt
    per (task, seed). For real per-step GRPO training use `build_step_dataset`.
    """
    dataset = []
    for task_id in tasks:
        for seed in seeds:
            try:
                reset_resp = client.post("/reset", json={"task_id": task_id, "seed": seed, "mode": "team"})
                reset_resp.raise_for_status()
                obs = reset_resp.json()
                prompt = format_obs_prompt(obs, role, step=0)
                dataset.append({
                    "prompt": [{"role": "system", "content": ROLE_SYSTEM_PROMPTS[role]},
                               {"role": "user",   "content": prompt}],
                    "task_id": task_id,
                    "seed": seed,
                    "step_index": 0,
                })
            except Exception as e:
                print(f"  [WARN] Failed to generate prompt for {task_id} seed={seed}: {e}")
    return dataset


def build_step_dataset(
    client: httpx.Client,
    tasks: list[str],
    seeds: list[int],
    role: str,
    max_steps_per_episode: int = 80,
) -> list[dict]:
    """
    Per-step dataset builder for real per-step GRPO.

    Runs oracle rollouts and records every observation where acting role ==
    `role`. Each dataset row is a single (prompt, task_id, seed, step_index)
    tuple; the reward function replays oracle actions up to `step_index`
    to reproduce the exact state before applying the model's action.

    Returns list of {"prompt": chat-messages, "task_id": str, "seed": int,
                     "step_index": int}.
    """
    dataset = []
    for task_id in tasks:
        for seed in seeds:
            try:
                reset_resp = client.post(
                    "/reset", json={"task_id": task_id, "seed": seed, "mode": "team"}
                )
                reset_resp.raise_for_status()
                obs = reset_resp.json()

                step = 0
                while not obs.get("done", False) and step < max_steps_per_episode:
                    acting_role = obs.get("current_role") or "tier1"
                    if acting_role == role:
                        prompt = format_obs_prompt(obs, role, step=step)
                        dataset.append({
                            "prompt": [
                                {"role": "system", "content": ROLE_SYSTEM_PROMPTS[role]},
                                {"role": "user",   "content": prompt},
                            ],
                            "task_id": task_id,
                            "seed": seed,
                            "step_index": step,
                        })
                    step += 1
                    action = oracle_action(obs)
                    step_resp = client.post(
                        "/step",
                        content=json.dumps(action),
                        headers={"Content-Type": "application/json"},
                    )
                    if step_resp.status_code != 200:
                        break
                    obs = step_resp.json()
            except Exception as e:
                print(f"  [WARN] Failed step-dataset build for {task_id} seed={seed}: {e}")
    return dataset


def replay_to_step(
    client: httpx.Client,
    task_id: str,
    seed: int,
    step_index: int,
    max_steps_per_episode: int = 80,
) -> dict:
    """Reset env and run oracle until reaching `step_index`. Returns that obs."""
    reset_resp = client.post(
        "/reset", json={"task_id": task_id, "seed": seed, "mode": "team"}
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()
    step = 0
    while step < step_index and not obs.get("done", False) and step < max_steps_per_episode:
        action = oracle_action(obs)
        step_resp = client.post(
            "/step",
            content=json.dumps(action),
            headers={"Content-Type": "application/json"},
        )
        if step_resp.status_code != 200:
            break
        obs = step_resp.json()
        step += 1
    return obs


def format_obs_prompt(obs: dict, role: str, step: int) -> str:
    alerts = obs.get("alert_queue", [])
    tickets = obs.get("tickets") or []
    phase = obs.get("current_phase", "triage")
    budget = obs.get("phase_steps_remaining", 40)

    lines = [
        f"=== SOC OBSERVATION | role={role} phase={phase} step={step} budget={budget} ===",
        f"Alerts ({len(alerts)}):",
    ]
    for a in alerts[:5]:  # cap at 5 to fit context
        lines.append(
            f"  [{a['alert_id']}] {a.get('title', '')} | severity={a.get('severity','')} "
            f"| IPs={list(a.get('indicators', {}).get('ip', []))}"
        )
    if tickets:
        lines.append(f"Tickets ({len(tickets)}):")
        for t in tickets[:3]:
            lines.append(f"  [{t.get('ticket_id','')}] kind={t.get('kind','')} alert={t.get('alert_id','')}")
    lines.append("Respond with a single JSON action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reward function (called by GRPOTrainer)
# ---------------------------------------------------------------------------

def _classify_parse_quality(text: str) -> str:
    """Tag the model's output by how cleanly it parses as a JSON action.

    Returns one of:
      "strict"   — entire completion is one valid JSON object with action_type
      "loose"    — JSON object with action_type recovered via fenced/regex extraction
      "fallback" — no parseable JSON; we can only infer intent from keywords (or noop)

    Used by the reward function to shape the model toward emitting clean JSON,
    which is the single most common failure mode for sub-3B models on
    structured action environments.
    """
    import re
    text = (text or "").strip()
    # Strict: bare JSON object with action_type
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return "strict"
    except (json.JSONDecodeError, TypeError):
        pass
    # Loose: extract via fenced or balanced-brace regex
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```", r"(\{[\s\S]*\})"):
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "action_type" in obj:
                return "loose"
        except json.JSONDecodeError:
            continue
    return "fallback"


# Shaping bonuses applied on top of the env's per-step reward. Tuned so that
# strict JSON over a noop-fallback is worth ~0.15 — enough to dominate small
# noisy env rewards early in training but small enough that a strict-JSON
# malicious action still loses to a loose-JSON correct action.
_PARSE_BONUS = {"strict": 0.05, "loose": 0.01, "fallback": -0.10}


def make_reward_fn(client: httpx.Client, role: str):
    """
    Real per-step GRPO reward function with JSON-validity shaping.

    Each training example is one (observation, step_index) pair produced by
    `build_step_dataset`. For each group-sampled completion we:
      1. Classify the completion's parse quality (strict / loose / fallback).
      2. Parse the action; if unparseable, fall back to noop.
      3. Reset the env to (task_id, seed) and replay oracle actions up to
         step_index — reaching the same state the prompt was drawn from.
      4. Apply the action and use the env's immediate step reward.
      5. Add a shaping bonus based on parse quality.

    The shaping term bootstraps sub-3B models out of the malformed-output
    trap: without it, a model that emits free-form text gets the env's noop
    reward (~0) and never receives a gradient signal toward valid JSON.
    """
    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        quality_counts = {"strict": 0, "loose": 0, "fallback": 0}
        task_ids = kwargs.get("task_id", ["team_phishing_escalation"] * len(completions))
        seeds = kwargs.get("seed", [42] * len(completions))
        step_indices = kwargs.get("step_index", [0] * len(completions))

        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            task_id = task_ids[i] if i < len(task_ids) else "team_phishing_escalation"
            seed = seeds[i] if i < len(seeds) else 42
            step_index = step_indices[i] if i < len(step_indices) else 0

            quality = _classify_parse_quality(text)
            quality_counts[quality] += 1
            shaping = _PARSE_BONUS[quality]

            try:
                action = parse_action_from_text(text, role)
                if action is None:
                    rewards.append(-0.05 + shaping)
                    continue

                obs = replay_to_step(client, task_id, seed, step_index)
                if obs.get("done", False):
                    rewards.append(0.0 + shaping)
                    continue

                # Guard against role mismatch: replay may have advanced past
                # the target role if the oracle completed the phase.
                acting_role = obs.get("current_role") or "tier1"
                if acting_role != role:
                    rewards.append(-0.02 + shaping)
                    continue

                step_resp = client.post(
                    "/step",
                    content=json.dumps(action),
                    headers={"Content-Type": "application/json"},
                )
                if step_resp.status_code != 200:
                    rewards.append(-0.05 + shaping)
                    continue
                stepped = step_resp.json()
                env_reward = float(stepped.get("reward", 0.0))
                rewards.append(env_reward + shaping)
            except Exception:
                rewards.append(0.0 + shaping)

        # Periodic visibility into JSON parse quality — most informative signal
        # for diagnosing why a small model is or isn't learning. Prints once per
        # batch (group of completions for the same prompt).
        total = sum(quality_counts.values()) or 1
        if total >= 4:  # batch of meaningful size
            def pct(k):
                return 100 * quality_counts[k] / total
            print(
                f"[reward_fn] parse: strict {pct('strict'):.0f}% "
                f"loose {pct('loose'):.0f}% fallback {pct('fallback'):.0f}% "
                f"| mean reward {sum(rewards)/len(rewards):+.3f}",
                flush=True,
            )
        return rewards

    return reward_fn


def parse_action_from_text(text: str, role: str) -> dict | None:
    """Extract JSON action from model output text."""
    import re
    # Try to find a JSON block
    patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
        r"(\{[\s\S]*?\})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                action = json.loads(m.group(1))
                if "action_type" in action:
                    action["role"] = role
                    return action
            except json.JSONDecodeError:
                continue
    # Keyword fallback
    text_lower = text.lower()
    for keyword, action_type in [
        ("escalate_to_tier2", "escalate_to_tier2"),
        ("classify", "classify_alert"),
        ("enrich", "enrich_indicator"),
        ("isolate", "isolate_host"),
        ("block_ioc", "block_ioc"),
        ("review", "review_decision"),
        ("explain", "explain_team_behavior"),
        ("phase_complete", "phase_complete"),
    ]:
        if keyword in text_lower:
            return {"action_type": action_type, "role": role}
    return {"action_type": "noop", "role": role}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    role: str = "tier1",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 8,  # GRPO group size — must match plan (group=8)
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    learning_rate: float = 5e-6,
    output_dir: str = None,
    use_unsloth: bool = False,
):
    """
    Full GRPO training for a SOC analyst role.
    Call this from the Colab notebook or CLI.
    """
    output_dir = output_dir or f"./soc_grpo_{role}"
    tasks = {"tier1": TIER1_TASKS, "tier2": TIER2_TASKS, "manager": MANAGER_TASKS}[role]

    print(f"\n{'='*60}")
    print("SOC-Triage-Gym v2 — GRPO Training")
    print(f"  role={role}  model={model_name}  epochs={num_train_epochs}")
    print(f"  tasks={tasks}  group_size={num_generations}")
    print(f"{'='*60}\n")

    # ---- Connect to environment server ----
    # 180s so long team_lateral_team rollouts (68 steps) don't trip during dataset build.
    client = httpx.Client(base_url=SERVER_URL, timeout=180.0)
    try:
        health = client.get("/health")
        health.raise_for_status()
        print(f"[OK] Server healthy at {SERVER_URL}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {SERVER_URL}: {e}")
        print("  Start the server first:  uvicorn server.app:app --port 7860")
        sys.exit(1)

    # ---- Build per-step prompt dataset (real GRPO) ----
    print(f"\n[1/4] Building per-step dataset ({len(SEEDS)} seeds × {len(tasks)} tasks)...")
    raw_dataset = build_step_dataset(client, tasks, SEEDS, role)
    print(f"      {len(raw_dataset)} per-step prompts generated")

    # Shuffle and hold out the first 10% from training. (Held-out *evaluation*
    # is done on disjoint seeds 100-114 in benchmark.py, not on this slice.)
    random.shuffle(raw_dataset)
    split = max(1, int(len(raw_dataset) * 0.1))
    train_data = raw_dataset[split:]

    # ---- Load model ----
    print(f"\n[2/4] Loading model: {model_name}...")
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_prompt_length + max_completion_length,
                dtype=None,  # auto-detect
                load_in_4bit=True,
                token=HF_TOKEN,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print("  [OK] Unsloth 4-bit model loaded with LoRA")
        except ImportError:
            print("  [WARN] Unsloth not available, falling back to standard transformers")
            use_unsloth = False

    if not use_unsloth:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype="auto",
            device_map="auto",
        )
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print("  [OK] Standard transformers model loaded with LoRA")

    # ---- Build reward function ----
    print(f"\n[3/4] Wiring reward function (role={role})...")
    reward_fn = make_reward_fn(client, role)

    # ---- Run GRPO training ----
    print("\n[4/4] Starting GRPO training...")

    try:
        import inspect

        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        hf_train = Dataset.from_list([{"prompt": d["prompt"],
                                        "task_id": d["task_id"],
                                        "seed": d["seed"],
                                        "step_index": d.get("step_index", 0)} for d in train_data])

        _grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters)
        config_kwargs: dict = dict(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=50,
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
            run_name=f"soc-grpo-{role}",
            temperature=1.0,
            seed=42,
        )
        # KL penalty param renamed across TRL versions
        if "beta" in _grpo_params:
            config_kwargs["beta"] = 0.04
        elif "kl_coef" in _grpo_params:
            config_kwargs["kl_coef"] = 0.04

        grpo_config = GRPOConfig(**config_kwargs)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=hf_train,
            processing_class=tokenizer,
        )

        print(f"  Training {len(hf_train)} examples × {num_train_epochs} epochs")
        print(f"  GRPO group size: {num_generations} (plan: group=8)")
        print(f"  Output: {output_dir}")

        trainer.train()

        # Unsloth + 4-bit: use merged-16bit save path. Plain trainer.save_model on a
        # 4-bit-loaded PEFT model can produce a damaged merge (Unsloth README warning).
        if use_unsloth and hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
            model.save_pretrained_merged(f"{output_dir}-lora", tokenizer, save_method="lora")
            print(f"\n[DONE] Merged 16-bit: {output_dir}  |  LoRA-only: {output_dir}-lora")
        else:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\n[DONE] Model saved to {output_dir}")

    except ImportError as e:
        print(f"\n[ERROR] TRL import failed: {e}")
        print("Install: pip install trl>=0.9.0 datasets")
        print("\nFalling back to reward-curve dry-run (no model update)...")
        _dry_run_reward_curve(client, tasks, role)


def _random_action(obs: dict, role: str) -> dict:
    """Uniform random policy over the valid action types for `role`.

    Used as the untrained baseline in --compare mode. This is deliberately
    dumb: it does not inspect alert_id/host validity, so it produces the
    same kind of mistakes an untrained LLM would make (wrong IDs, wrong
    action for the current phase, over-escalation, etc.).
    """
    tier1_actions = ["classify_alert", "enrich_indicator", "escalate_to_tier2",
                     "phase_complete", "noop"]
    tier2_actions = ["isolate_host", "block_ioc", "close_case",
                     "phase_complete", "noop"]
    manager_actions = ["review_decision", "explain_team_behavior", "noop"]
    pool = {"tier1": tier1_actions, "tier2": tier2_actions,
            "manager": manager_actions}[role]
    action_type = random.choice(pool)
    alerts = obs.get("alert_queue") or []
    aid = alerts[0]["alert_id"] if alerts else "ALT-UNKNOWN"
    action: dict = {"action_type": action_type, "role": role}
    if action_type == "classify_alert":
        action.update({"alert_id": aid,
                       "classification": random.choice(["true_positive", "false_positive"]),
                       "confidence": round(random.random(), 2)})
    elif action_type == "enrich_indicator":
        action.update({"indicator": "0.0.0.0", "indicator_type": "ip",
                       "query_alert_id": aid})
    elif action_type == "escalate_to_tier2":
        action.update({"alert_id": aid, "justification": "random"})
    elif action_type == "isolate_host":
        action["hostname"] = "RAND-HOST"
    elif action_type == "block_ioc":
        action.update({"indicator": "0.0.0.0", "indicator_type": "ip"})
    elif action_type == "close_case":
        action.update({"alert_id": aid, "justification": "random"})
    elif action_type == "review_decision":
        action["alert_id"] = aid
    elif action_type == "explain_team_behavior":
        action["explanation_text"] = "random"
    return action


def run_random_episode(
    client: httpx.Client,
    task_id: str,
    seed: int,
    role_to_train: str = "tier1",
    max_steps: int = 80,
) -> float:
    """Run an episode where `role_to_train` uses the random policy and the
    other roles use the oracle. Mirrors run_episode() so scores are comparable."""
    reset_resp = client.post(
        "/reset", json={"task_id": task_id, "seed": seed, "mode": "team"}
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()
    step = 0
    while not obs.get("done", False) and step < max_steps:
        step += 1
        role = obs.get("current_role") or "tier1"
        action = _random_action(obs, role) if role == role_to_train else oracle_action(obs)
        step_resp = client.post(
            "/step", content=json.dumps(action),
            headers={"Content-Type": "application/json"},
        )
        if step_resp.status_code != 200:
            break
        obs = step_resp.json()
    return float(obs.get("task_score") or obs.get("cumulative_reward", 0.0))


def _compare_baselines(client, tasks, role, n_seeds: int = 10):
    """Produce reward_comparison_baseline.{png,csv}: random vs oracle per episode.

    This is the artifact for the 20%-weighted "Showing Improvement in Rewards"
    judging criterion. Even without GPU training it establishes the measurable
    gap an RL-trained policy is expected to close.
    """
    import csv as _csv

    print("\n--- Baseline comparison: random vs oracle ---")
    rows = []
    ep = 0
    random_scores: list[float] = []
    oracle_scores: list[float] = []
    for task_id in tasks:
        for seed in SEEDS[:n_seeds]:
            ep += 1
            r_score = run_random_episode(client, task_id, seed, role_to_train=role)
            o_score, _ = run_episode(client, task_id=task_id, seed=seed,
                                     role_to_train=role)
            random_scores.append(r_score)
            oracle_scores.append(o_score)
            rows.append({"episode": ep, "task_id": task_id, "seed": seed,
                         "random": r_score, "oracle": o_score,
                         "delta": o_score - r_score})
            print(f"  Ep {ep:3d} | {task_id} seed={seed} | "
                  f"random={r_score:.3f} oracle={o_score:.3f} Δ={o_score - r_score:+.3f}")

    csv_path = f"reward_comparison_baseline_{role}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    r_mean = sum(random_scores) / len(random_scores)
    o_mean = sum(oracle_scores) / len(oracle_scores)
    gap = o_mean - r_mean
    print(f"Random mean:  {r_mean:.4f}")
    print(f"Oracle mean:  {o_mean:.4f}")
    print(f"Learnable gap: {gap:+.4f}  (headroom for an RL-trained policy)")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(1, ep + 1)
        ax.plot(x, random_scores, alpha=0.35, color="#b64a3b",
                label="Random policy (untrained baseline)")
        ax.plot(x, oracle_scores, alpha=0.35, color="#2b4a3a",
                label="Oracle heuristic (ceiling)")
        w = min(5, ep)
        if w >= 2:
            r_smooth = np.convolve(random_scores, np.ones(w) / w, mode="valid")
            o_smooth = np.convolve(oracle_scores, np.ones(w) / w, mode="valid")
            ax.plot(range(w, ep + 1), r_smooth, linewidth=2, color="#b64a3b")
            ax.plot(range(w, ep + 1), o_smooth, linewidth=2, color="#2b4a3a")
        ax.axhline(r_mean, linestyle=":", color="#b64a3b", alpha=0.6,
                   label=f"Random μ={r_mean:.3f}")
        ax.axhline(o_mean, linestyle="--", color="#2b4a3a", alpha=0.6,
                   label=f"Oracle μ={o_mean:.3f}")
        ax.fill_between(x, random_scores, oracle_scores,
                        where=[o >= r for o, r in zip(oracle_scores, random_scores, strict=False)],
                        alpha=0.08, color="#2b4a3a")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Task score")
        ax.set_title(f"SOC-Triage-Gym — {role.upper()} baseline gap "
                     f"(random vs oracle) | learnable Δ={gap:+.3f}")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        png_path = f"reward_comparison_baseline_{role}.png"
        fig.savefig(png_path, dpi=150)
        print(f"Saved plot: {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot")


def _dry_run_reward_curve(client, tasks, role):
    """Plot a reward curve using oracle rollouts (no model needed — for CI/demo)."""

    print("\n--- Dry-run reward curve (oracle heuristic) ---")
    episode_scores = []
    for episode_num, (task_id, seed) in enumerate(
        [(t, s) for t in tasks for s in SEEDS[:10]], start=1
    ):
        score, traj = run_episode(client, task_id=task_id, seed=seed,
                                  role_to_train=role)
        episode_scores.append(score)
        print(f"  Episode {episode_num:3d} | {task_id} seed={seed} | score={score:.4f}")

    avg = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
    print(f"\nOracle avg score: {avg:.4f} (target for trained model: >{avg:.4f})")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(1, len(episode_scores) + 1)
        ax.plot(x, episode_scores, alpha=0.4, label="Episode score")
        window = min(5, len(episode_scores))
        smoothed = np.convolve(episode_scores, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(episode_scores) + 1), smoothed, linewidth=2,
                label=f"Rolling avg (n={window})")
        ax.axhline(avg, linestyle="--", color="gray", label=f"Mean={avg:.3f}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title(f"SOC-Triage-Gym v2 — {role.upper()} Oracle Reward Curve")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(f"reward_curve_{role}_oracle.png", dpi=150)
        print(f"Saved: reward_curve_{role}_oracle.png")
    except ImportError:
        print("matplotlib not installed — skipping plot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOC-Triage-Gym v2 GRPO Training")
    parser.add_argument("--role", choices=["tier1", "tier2", "manager"], default="tier1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8,
                        help="GRPO group size (plan: 8)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--unsloth", action="store_true",
                        help="Use Unsloth for 4-bit training (recommended for T4/A100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model loading; just plot oracle reward curve")
    parser.add_argument("--compare", action="store_true",
                        help="Plot random-policy vs oracle baseline gap "
                             "(artifact for the 20%% 'Showing Improvement' judging criterion)")
    parser.add_argument("--compare-seeds", type=int, default=10,
                        help="Seeds per task for --compare (default: 10)")
    args = parser.parse_args()

    if args.compare:
        client = httpx.Client(base_url=SERVER_URL, timeout=30.0)
        tasks = {"tier1": TIER1_TASKS, "tier2": TIER2_TASKS, "manager": MANAGER_TASKS}[args.role]
        _compare_baselines(client, tasks, args.role, n_seeds=args.compare_seeds)
    elif args.dry_run:
        client = httpx.Client(base_url=SERVER_URL, timeout=30.0)
        tasks = {"tier1": TIER1_TASKS, "tier2": TIER2_TASKS, "manager": MANAGER_TASKS}[args.role]
        _dry_run_reward_curve(client, tasks, args.role)
    else:
        train(
            role=args.role,
            model_name=args.model,
            num_train_epochs=args.epochs,
            num_generations=args.group_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            use_unsloth=args.unsloth,
        )


if __name__ == "__main__":
    main()
