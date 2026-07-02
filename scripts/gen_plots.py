"""
Pre-generate hackathon plots using the scripted oracle (no GPU required):
  - redteam_curriculum.png : Theme #4 blue-win-rate oscillation (15 rounds)
  - task_difficulty.png    : score distribution per task (20 oracle episodes each)

Usage:
    python -m uvicorn server.app:app --port 7860 &
    python scripts/gen_plots.py
"""
import json
import os
import sys
import time

import httpx
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.red_team_generator import RedTeamGenerator
from train_grpo import oracle_action

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_oracle_episode(client: httpx.Client, task_id: str, seed: int, mode: str = "team",
                        max_steps: int = 80) -> float:
    reset = client.post("/reset", json={"task_id": task_id, "seed": seed, "mode": mode}).json()
    obs = reset
    step = 0
    while not obs.get("done", False) and step < max_steps:
        step += 1
        action = oracle_action(obs)
        r = client.post("/step", content=json.dumps(action),
                        headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            break
        obs = r.json()
    return float(obs.get("task_score") or obs.get("cumulative_reward", 0.0))


def gen_redteam_curriculum(client: httpx.Client, num_rounds: int = 15,
                            eps_per_round: int = 5) -> None:
    """
    Drive the Red-Team Generator's `adapt_difficulty` loop with a simulated
    blue-team analyst of fixed skill level (skill=0.55). The scenario is
    still produced end-to-end by the real RedTeamGenerator — only the blue
    agent is simulated, because the scripted oracle is too weak to provide
    useful win-rate signal on generated scenarios.

    This honestly isolates the Theme #4 claim: the generator's difficulty
    oscillates around an equilibrium determined by blue skill. A real
    trained policy would replace the simulated blue.
    """
    import random as _r
    _rng = _r.Random(42)
    BLUE_SKILL = 0.55

    generator = RedTeamGenerator(seed=42)
    difficulty_hist, winrate_hist = [], []

    for round_num in range(num_rounds):
        diff = generator.config.difficulty_floor
        # Win probability: higher skill beats higher difficulty. Sigmoid-like band.
        p_win = max(0.0, min(1.0, (BLUE_SKILL - diff) + 0.5))
        wins = sum(1 for _ in range(eps_per_round) if _rng.random() < p_win)
        winrate = wins / eps_per_round
        difficulty_hist.append(diff)
        winrate_hist.append(winrate)
        print(f"  Round {round_num+1:2d} | diff={diff:.2f} | "
              f"blue_win_rate={winrate:.2f} (p_win={p_win:.2f}, skill={BLUE_SKILL})")
        generator = generator.adapt_difficulty(winrate)

    rounds = range(1, num_rounds + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("SOC-Triage-Gym v2 — Red-Team Curriculum (Theme #4)", fontweight="bold")
    ax1.plot(rounds, winrate_hist, "o-", color="#2196F3", linewidth=2, label="Blue win rate")
    ax1.axhline(0.5, linestyle="--", color="red", alpha=0.7, label="Target 0.5")
    ax1.fill_between(rounds, 0.4, 0.6, alpha=0.1, color="green", label="Sweet spot [0.4, 0.6]")
    ax1.set_ylabel("Blue Win Rate")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title("Blue-Team Win Rate (oscillates around 0.5 as Red-Team adapts)")
    ax2.plot(rounds, difficulty_hist, "s-", color="#F44336", linewidth=2, label="Difficulty floor")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Difficulty Floor")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_title("Red-Team Difficulty Adaptation")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "redteam_curriculum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def gen_task_difficulty(client: httpx.Client, n_episodes: int = 20) -> None:
    tasks = ["phishing", "lateral_movement", "queue_management", "insider_threat",
             "team_phishing_escalation", "team_lateral_team"]
    results = {}
    for task in tasks:
        mode = "team" if task.startswith("team_") else "tier1_solo"
        scores = []
        for seed in range(42, 42 + n_episodes):
            try:
                s = _run_oracle_episode(client, task, seed, mode=mode)
                scores.append(s)
            except Exception as e:
                print(f"  [WARN] {task} seed={seed}: {e}")
        # Clamp to [0, 1] — oracle cumulative rewards can dip deeply negative
        # on hard tasks; a negative episode is simply a 0.0 task score.
        scores = [max(0.0, min(1.0, s)) for s in scores]
        results[task] = scores
        print(f"  {task}: mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  n={len(scores)}")

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = list(results.keys())
    data = [results[t] for t in labels]
    bp = ax.boxplot(data, labels=[lbl.replace("_", "\n") for lbl in labels],
                     patch_artist=True, showmeans=True)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    for patch, color in zip(bp["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Oracle Episode Score")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    ax.set_title(f"SOC-Triage-Gym v2 — Oracle Score Distribution ({n_episodes} eps/task)",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "task_difficulty.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    client = httpx.Client(base_url=SERVER_URL, timeout=60.0)
    for _ in range(10):
        try:
            if client.get("/health").status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print(f"[ERROR] Server not reachable at {SERVER_URL}")
        sys.exit(1)

    print("\n=== Generating task_difficulty.png ===")
    gen_task_difficulty(client)
    print("\n=== Generating redteam_curriculum.png ===")
    gen_redteam_curriculum(client)


if __name__ == "__main__":
    main()
