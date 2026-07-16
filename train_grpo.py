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

from training.curriculum import default_team_curriculum
from training.evaluation import DEFAULT_EVAL_SEEDS, evaluate_policy
from training.rewards import (
    CompletionItem,
    ParallelRewardEvaluator,
    score_completion,
)
from training.rewards import (
    parse_action_from_text as parse_action_from_text,
)
from training.run_manager import TrainingRunManager

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
# Parsing, shaping, and per-completion scoring live in training/rewards.py
# (imported above as parse_action_from_text / score_completion) so they are
# unit-testable and shared with scripts/. parse_action_from_text is
# re-exported here for backward compatibility — scripts/train_and_evaluate.py
# imports it from train_grpo.


def make_reward_fn(
    client: httpx.Client,
    role: str,
    evaluator: ParallelRewardEvaluator | None = None,
    curriculum=None,
    run_manager: TrainingRunManager | None = None,
):
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

    Optional hooks:
      evaluator   — ParallelRewardEvaluator: scores the completion group
                    concurrently across isolated server sessions.
      curriculum  — CurriculumScheduler: records rewards for promotion gates.
      run_manager — TrainingRunManager: streams batch stats to metrics.jsonl.
    """
    def reward_fn(prompts, completions, **kwargs):
        task_ids = kwargs.get("task_id", ["team_phishing_escalation"] * len(completions))
        seeds = kwargs.get("seed", [42] * len(completions))
        step_indices = kwargs.get("step_index", [0] * len(completions))

        items = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            items.append(CompletionItem(
                text=text,
                task_id=task_ids[i] if i < len(task_ids) else "team_phishing_escalation",
                seed=seeds[i] if i < len(seeds) else 42,
                step_index=step_indices[i] if i < len(step_indices) else 0,
            ))

        if evaluator is not None:
            rewards, quality_counts = evaluator.score_batch(items, role, replay_to_step)
        else:
            rewards = []
            quality_counts = {"strict": 0, "loose": 0, "fallback": 0}
            for item in items:
                reward, quality = score_completion(
                    client,
                    text=item.text,
                    role=role,
                    task_id=item.task_id,
                    seed=item.seed,
                    step_index=item.step_index,
                    replay_fn=replay_to_step,
                )
                rewards.append(reward)
                quality_counts[quality] += 1

        if curriculum is not None:
            curriculum.record_batch(rewards)

        # Periodic visibility into JSON parse quality — most informative signal
        # for diagnosing why a small model is or isn't learning. Prints once per
        # batch (group of completions for the same prompt).
        total = sum(quality_counts.values()) or 1
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        if run_manager is not None:
            run_manager.log(
                "reward_batch",
                mean_reward=round(mean_reward, 4),
                batch=total,
                **quality_counts,
            )
        if total >= 4:  # batch of meaningful size
            def pct(k):
                return 100 * quality_counts[k] / total
            print(
                f"[reward_fn] parse: strict {pct('strict'):.0f}% "
                f"loose {pct('loose'):.0f}% fallback {pct('fallback'):.0f}% "
                f"| mean reward {mean_reward:+.3f}",
                flush=True,
            )
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def _prime_red_team(client: httpx.Client, seed: int = 42, episode_count: int = 0) -> None:
    """Generate + load an adaptive red-team scenario so that
    reset(task_id='red_team_generated') has something to run."""
    try:
        client.post("/generate_scenario", json={"seed": seed, "episode_count": episode_count})
    except Exception as e:
        print(f"  [WARN] red-team scenario generation failed: {e}")


def _make_model_policy(model, tokenizer, role: str, max_new_tokens: int = 128):
    """Wrap a (trained) HF model as an obs→action policy for held-out eval."""
    import torch

    def policy(obs: dict) -> dict:
        prompt = format_obs_prompt(obs, role, step=int(obs.get("step_count") or 0))
        messages = [
            {"role": "system", "content": ROLE_SYSTEM_PROMPTS[role]},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_action_from_text(completion, role) or {"action_type": "noop", "role": role}

    return policy


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
    curriculum: bool = False,
    parallel_workers: int = 1,
    runs_dir: str = "runs",
    early_stop_patience: int = 0,
    eval_episodes: int = 0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    save_total_limit: int = 3,
):
    """
    Full GRPO training for a SOC analyst role.
    Call this from the Colab notebook or CLI.

    v0.3 training enhancements (all opt-in, defaults preserve old behaviour):
      curriculum          — staged easy→hard task curriculum with promotion
                            gates (training/curriculum.py); final stage mixes
                            in adaptive red_team_generated scenarios (RLVE).
      parallel_workers    — score each GRPO completion group concurrently
                            across isolated server sessions (needs server
                            >= 0.2.0). Reward evaluation is the wall-clock
                            bottleneck; 4-8 workers ≈ 4-8× faster rewards.
      runs_dir            — every run gets a structured runs/<run_id>/ dir:
                            config + git provenance, metrics.jsonl, rotated
                            checkpoints, eval reports, auto model card.
      early_stop_patience — stop when train reward plateaus for N log events.
      eval_episodes       — after training, run held-out eval (seeds 100+)
                            with the trained model and record it in the run.
      lr_scheduler_type / warmup_ratio / save_total_limit — trainer knobs.
    """
    tasks = {"tier1": TIER1_TASKS, "tier2": TIER2_TASKS, "manager": MANAGER_TASKS}[role]

    # ---- Structured run directory (config, metrics, checkpoints, model card) ----
    run_manager = TrainingRunManager(role=role, model_name=model_name, runs_dir=runs_dir)
    output_dir = output_dir or str(run_manager.checkpoints_dir)

    print(f"\n{'='*60}")
    print("SOC-Triage-Gym — GRPO Training")
    print(f"  role={role}  model={model_name}  epochs={num_train_epochs}")
    print(f"  tasks={tasks}  group_size={num_generations}")
    print(f"  run={run_manager.run_id}  curriculum={curriculum}  parallel_workers={parallel_workers}")
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

    # ---- Optional training enhancements ----
    scheduler = default_team_curriculum() if curriculum else None
    evaluator = None
    if parallel_workers > 1:
        evaluator = ParallelRewardEvaluator(
            client_factory=lambda sid: httpx.Client(
                base_url=SERVER_URL, timeout=180.0, headers={"X-Session-ID": sid}
            ),
            workers=parallel_workers,
        )
        print(f"[OK] Parallel reward evaluation: {parallel_workers} isolated server sessions")

    run_manager.write_config({
        "tasks": tasks,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "save_total_limit": save_total_limit,
        "use_unsloth": use_unsloth,
        "curriculum": [s.name for s in scheduler.stages] if scheduler else None,
        "parallel_workers": parallel_workers,
        "early_stop_patience": early_stop_patience,
        "eval_episodes": eval_episodes,
        "output_dir": output_dir,
    })

    # ---- Build per-step prompt dataset (real GRPO) ----
    # With a curriculum, per-stage datasets are built inside the stage loop;
    # this initial build covers the non-curriculum path.
    initial_tasks = scheduler.current.tasks if scheduler else tasks
    print(f"\n[1/4] Building per-step dataset ({len(SEEDS)} seeds × {len(initial_tasks)} tasks)...")
    if "red_team_generated" in initial_tasks:
        _prime_red_team(client, seed=SEEDS[0])
    raw_dataset = build_step_dataset(client, initial_tasks, SEEDS, role)
    print(f"      {len(raw_dataset)} per-step prompts generated")
    run_manager.log("dataset_built", tasks=initial_tasks, size=len(raw_dataset))

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
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            # CPU-only dev machine without the training stack: degrade to the
            # oracle reward-curve dry-run and still finalize the run record.
            print(f"\n[ERROR] Training stack unavailable: {e}")
            print("Install: pip install 'trl>=0.11' 'peft>=0.14' 'transformers<5' datasets accelerate")
            print("\nFalling back to reward-curve dry-run (no model update)...")
            _dry_run_reward_curve(client, tasks, role)
            run_manager.finalize({
                "tasks": tasks,
                "dataset_size": len(train_data),
                "seed_range": f"{SEEDS[0]}..{SEEDS[-1]}",
                "status": "dry_run_fallback",
                "error": str(e),
            })
            return
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
    reward_fn = make_reward_fn(
        client, role,
        evaluator=evaluator,
        curriculum=scheduler,
        run_manager=run_manager,
    )

    # ---- Run GRPO training ----
    print("\n[4/4] Starting GRPO training...")

    try:
        import inspect

        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        _grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters)

        def _to_hf_dataset(rows: list[dict]) -> Dataset:
            return Dataset.from_list([{"prompt": d["prompt"],
                                       "task_id": d["task_id"],
                                       "seed": d["seed"],
                                       "step_index": d.get("step_index", 0)} for d in rows])

        def _build_config(run_tag: str) -> GRPOConfig:
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
                run_name=run_tag,
                temperature=1.0,
                seed=42,
            )
            # KL penalty param renamed across TRL versions
            if "beta" in _grpo_params:
                config_kwargs["beta"] = 0.04
            elif "kl_coef" in _grpo_params:
                config_kwargs["kl_coef"] = 0.04
            # Newer trainer knobs — guarded so older TRL versions still work.
            for key, value in (
                ("lr_scheduler_type", lr_scheduler_type),
                ("warmup_ratio", warmup_ratio),
                ("save_total_limit", save_total_limit),
                ("max_grad_norm", 1.0),
            ):
                if key in _grpo_params:
                    config_kwargs[key] = value
            # bf16 when the GPU supports it (better GRPO numerics than fp16)
            if "bf16" in _grpo_params:
                try:
                    import torch
                    config_kwargs["bf16"] = bool(
                        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    )
                except ImportError:
                    pass
            return GRPOConfig(**config_kwargs)

        # Trainer callback: streams logs into runs/<id>/metrics.jsonl and
        # applies reward-plateau early stopping.
        callbacks = []
        try:
            from training.callbacks import SOCGymCallback
            callbacks.append(SOCGymCallback(run_manager, early_stop_patience=early_stop_patience))
        except ImportError:
            pass

        def _fit(rows: list[dict], run_tag: str) -> None:
            hf_train = _to_hf_dataset(rows)
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=[reward_fn],
                args=_build_config(run_tag),
                train_dataset=hf_train,
                processing_class=tokenizer,
                callbacks=callbacks or None,
            )
            print(f"  Training {len(hf_train)} examples × {num_train_epochs} epochs [{run_tag}]")
            print(f"  GRPO group size: {num_generations} (plan: group=8)")
            print(f"  Output: {output_dir}")
            trainer.train()
            _fit.last_trainer = trainer

        total_examples = 0
        if scheduler is not None:
            # ---- Staged curriculum: easy → hard, promote on reward gates ----
            round_num = 0
            stage_rows = train_data  # stage-0 dataset was built above
            while not scheduler.finished:
                stage = scheduler.current
                round_num += 1
                print(f"\n--- Curriculum stage '{stage.name}' "
                      f"(round {round_num}, tasks={stage.tasks}, "
                      f"gate={stage.promotion_threshold}) ---")
                if stage_rows is None:
                    if "red_team_generated" in stage.tasks:
                        _prime_red_team(client, seed=SEEDS[0], episode_count=round_num)
                    raw = build_step_dataset(client, stage.tasks, SEEDS, role)
                    random.shuffle(raw)
                    stage_rows = raw[max(1, int(len(raw) * 0.1)):]
                run_manager.log("stage_start", stage=stage.name, round=round_num,
                                dataset=len(stage_rows), tasks=stage.tasks)
                total_examples += len(stage_rows)
                _fit(stage_rows, f"soc-grpo-{role}-{stage.name}-r{round_num}")
                decision = scheduler.end_round()
                run_manager.log("curriculum_decision", **decision)
                verdict = ("PROMOTED" if decision["promoted"]
                           else "force-advanced" if decision["forced_advance"]
                           else "repeating stage")
                mean = decision["rolling_mean"]
                print(f"  Stage '{stage.name}': mean reward "
                      f"{mean if mean is None else round(mean, 3)} "
                      f"vs gate {stage.promotion_threshold} → {verdict}")
                if decision["promoted"] or decision["forced_advance"]:
                    stage_rows = None  # build the next stage's dataset next loop
        else:
            total_examples = len(train_data)
            _fit(train_data, f"soc-grpo-{role}")

        trainer = getattr(_fit, "last_trainer", None)

        # Unsloth + 4-bit: use merged-16bit save path. Plain trainer.save_model on a
        # 4-bit-loaded PEFT model can produce a damaged merge (Unsloth README warning).
        if use_unsloth and hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
            model.save_pretrained_merged(f"{output_dir}-lora", tokenizer, save_method="lora")
            print(f"\n[DONE] Merged 16-bit: {output_dir}  |  LoRA-only: {output_dir}-lora")
        elif trainer is not None:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\n[DONE] Model saved to {output_dir}")

        # ---- Held-out evaluation with the trained model (seeds 100+) ----
        eval_report = None
        if eval_episodes > 0:
            print(f"\n[eval] Held-out evaluation ({eval_episodes} seeds × {len(tasks)} tasks)...")
            policy = _make_model_policy(model, tokenizer, role)
            eval_report = evaluate_policy(
                client, role, policy, oracle_action, tasks,
                seeds=DEFAULT_EVAL_SEEDS[:eval_episodes],
            )
            run_manager.record_eval(eval_report, tag="held_out")
            is_best = run_manager.consider_best(eval_report["mean_score"], checkpoint_dir=output_dir)
            print(f"  mean={eval_report['mean_score']:.3f} ± {eval_report['std_score']:.3f}"
                  f"{'  (new best)' if is_best else ''}")

        # ---- Finalize the run: MANIFEST.json + auto model card ----
        batches = run_manager.read_metrics("reward_batch")
        final_mean = batches[-1]["mean_reward"] if batches else None
        run_manager.finalize({
            "tasks": tasks,
            "dataset_size": total_examples,
            "seed_range": f"{SEEDS[0]}..{SEEDS[-1]}",
            "final_mean_reward": final_mean,
            "eval_mean_score": eval_report["mean_score"] if eval_report else None,
            "output_dir": output_dir,
            "curriculum_history": scheduler.history if scheduler else None,
        })
        print(f"\n[run] Artifacts: {run_manager.run_dir}")
        print("      config.json · metrics.jsonl · checkpoints/ · eval/ · MODEL_CARD.md · MANIFEST.json")

    except ImportError as e:
        print(f"\n[ERROR] TRL import failed: {e}")
        print("Install: pip install trl>=0.9.0 datasets")
        print("\nFalling back to reward-curve dry-run (no model update)...")
        _dry_run_reward_curve(client, tasks, role)
        run_manager.finalize({
            "tasks": tasks,
            "dataset_size": len(train_data),
            "seed_range": f"{SEEDS[0]}..{SEEDS[-1]}",
            "status": "dry_run_fallback",
            "error": str(e),
        })


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
    parser.add_argument("--curriculum", action="store_true",
                        help="Staged easy→hard task curriculum with reward promotion gates; "
                             "final stage mixes in adaptive red-team scenarios (RLVE)")
    parser.add_argument("--parallel-rewards", type=int, default=1, metavar="N",
                        help="Score GRPO completion groups across N isolated server "
                             "sessions concurrently (default: 1 = sequential)")
    parser.add_argument("--runs-dir", default="runs",
                        help="Root for structured run artifacts (default: runs/)")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop when train reward plateaus for N log events (0=off)")
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="Held-out eval episodes per task after training (seeds 100+, 0=off)")
    parser.add_argument("--lr-scheduler", default="cosine",
                        help="LR schedule: cosine (default), linear, constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
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
            curriculum=args.curriculum,
            parallel_workers=args.parallel_rewards,
            runs_dir=args.runs_dir,
            early_stop_patience=args.early_stop_patience,
            eval_episodes=args.eval_episodes,
            lr_scheduler_type=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
        )


if __name__ == "__main__":
    main()
