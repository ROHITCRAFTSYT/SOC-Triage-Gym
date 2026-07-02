"""
Generate `soc_grpo_results.png` and `redteam_curriculum.png` on your Mac
from the executed Kaggle notebook outputs — no GPU, no cloud, no model load.

What this does:
  1. Reads `soc_triage_gym_v2_training.ipynb` and extracts the captured
     stream output of the training cell (every `[reward_fn]` log line).
  2. Plots the real training-reward curve and JSON-parse-rate curve.
  3. Reads the captured `Oracle baseline avg` from the baseline cell.
  4. For the red-team curriculum (cell 26 didn't run), simulates the same
     adaptive oscillation that the env's RedTeamGenerator would produce —
     noted as "simulated" in the figure title.

Run:
    python3 scripts/finalize_outputs.py

Outputs (in repo root):
    soc_grpo_results.png
    redteam_curriculum.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NB   = REPO / "soc_triage_gym_v2_training.ipynb"

REWARD_RE       = re.compile(r"mean reward ([+-][\d.]+)")
PARSE_RE        = re.compile(r"strict (\d+)%\s+loose\s+(\d+)%\s+fallback\s+(\d+)%")
BASELINE_RE     = re.compile(r"Oracle baseline avg:\s*([\d.]+)")


def extract_from_notebook():
    """Return (rewards, strict_pcts, baseline_avg) parsed from the notebook."""
    if not NB.exists():
        sys.exit(f"notebook not found: {NB}")
    nb = json.loads(NB.read_text())
    rewards, strict_pcts = [], []
    baseline_avg = None

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            txt = "".join(out.get("text", []))
            if not txt:
                continue
            if baseline_avg is None:
                m = BASELINE_RE.search(txt)
                if m:
                    baseline_avg = float(m.group(1))
            for m in REWARD_RE.finditer(txt):
                rewards.append(float(m.group(1)))
            for m in PARSE_RE.finditer(txt):
                strict_pcts.append(int(m.group(1)))

    return rewards, strict_pcts, baseline_avg


def plot_grpo_results(rewards, strict_pcts, baseline_avg, trained_estimate):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SOC-Triage-Gym v2 — GRPO Training Results",
                 fontsize=14, fontweight="bold")

    # Panel 1 — training reward (REAL)
    ax = axes[0]
    x = np.arange(1, len(rewards) + 1)
    ax.plot(x, rewards, "o-", color="#55A868", alpha=0.35,
            label=f"Per-log-step reward (n={len(rewards)})")
    if len(rewards) >= 11:
        w = 11
        sm = np.convolve(rewards, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w, len(rewards) + 1), sm,
                linewidth=2.5, color="#1F6B3A", label=f"Smoothed (w={w})")
    ax.set_xlabel("Logging step (×2 training steps)")
    ax.set_ylabel("Mean GRPO group reward")
    ax.set_title("Training Reward Curve (real data from Kaggle run)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2 — JSON parse health (REAL)
    ax2 = axes[1]
    if strict_pcts:
        x2 = np.arange(1, len(strict_pcts) + 1)
        ax2.plot(x2, strict_pcts, "o-", color="#4C72B0", alpha=0.4,
                 label="strict-JSON parse rate")
        if len(strict_pcts) >= 11:
            w = 11
            sm = np.convolve(strict_pcts, np.ones(w) / w, mode="valid")
            ax2.plot(np.arange(w, len(strict_pcts) + 1), sm,
                     linewidth=2.5, color="#1F3D6B", label=f"Smoothed (w={w})")
        ax2.axhline(60, linestyle="--", color="orange", alpha=0.6,
                    label="60% health threshold")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Logging step")
    ax2.set_ylabel("Strict JSON parse rate (%)")
    ax2.set_title("Output Format Stability")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Panel 3 — reward distribution histogram (REAL — shows training stability)
    ax3 = axes[2]
    ax3.hist(rewards, bins=20, color="#DD8452", alpha=0.85, edgecolor="black")
    mean_r = float(np.mean(rewards))
    last_window = float(np.mean(rewards[-min(20, len(rewards)):]))
    ax3.axvline(mean_r, linestyle="--", color="black", alpha=0.7,
                label=f"overall mean = {mean_r:.3f}")
    ax3.axvline(last_window, linestyle="--", color="#C44E52", alpha=0.9,
                label=f"last-20 mean = {last_window:.3f}")
    ax3.set_xlabel("Per-log-step reward")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Reward Distribution (n={len(rewards)})")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = REPO / "soc_grpo_results.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")


def plot_redteam_curriculum():
    """
    Run the heuristic red-team adaptive-difficulty algorithm — pure logic,
    no GPU, no env server. Mirrors what cell 26 would have produced.
    """
    rng = np.random.default_rng(42)
    NUM_ROUNDS = 40
    EPISODES_PER_ROUND = 15

    difficulty = 0.30  # initial floor
    win_rates, difficulties = [], []

    for _r in range(NUM_ROUNDS):
        # Blue oracle wins more often when difficulty is below ~0.5; below it
        # wins ~70%, above it drops fast. Add per-round noise.
        prob = float(np.clip(0.95 - 1.4 * difficulty + rng.normal(0, 0.05), 0.05, 0.95))
        wins = rng.binomial(EPISODES_PER_ROUND, prob)
        wr = wins / EPISODES_PER_ROUND

        win_rates.append(wr)
        difficulties.append(difficulty)

        # adapt: same rule as scenarios/red_team_generator.py
        if wr > 0.75:
            difficulty = min(0.95, difficulty + 0.10)
        elif wr < 0.45:
            difficulty = max(0.05, difficulty - 0.10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("SOC-Triage-Gym v2 — Red-Team Curriculum (Theme #4)\n"
                 "Self-improving difficulty adaptation (heuristic simulation)",
                 fontweight="bold")

    rounds = np.arange(1, NUM_ROUNDS + 1)
    ax1.plot(rounds, win_rates, "o-", color="#2196F3",
             linewidth=1.5, alpha=0.5, label="Blue win rate (raw)")
    sw = 5
    if NUM_ROUNDS >= sw:
        sm = np.convolve(win_rates, np.ones(sw) / sw, mode="valid")
        ax1.plot(rounds[sw - 1:], sm, color="#0D47A1",
                 linewidth=2.5, label=f"Smoothed (w={sw})")
    ax1.axhline(0.5, linestyle="--", color="red", alpha=0.7, label="Target 0.5")
    ax1.fill_between(rounds, 0.4, 0.6, alpha=0.1, color="green",
                     label="Sweet spot [0.4, 0.6]")
    ax1.set_ylabel("Blue Win Rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Blue-Team Win Rate (oscillates around 0.5 as Red-Team adapts)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(rounds, difficulties, "s-", color="#F44336",
             linewidth=2, label="Difficulty floor")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Difficulty Floor")
    ax2.set_ylim(0, 1)
    ax2.set_title("Red-Team Difficulty Adaptation")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = REPO / "redteam_curriculum.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")


def main():
    rewards, strict_pcts, baseline_avg = extract_from_notebook()
    if not rewards:
        sys.exit("no reward data found in notebook — was training cell executed?")
    print(f"extracted: {len(rewards)} reward samples, "
          f"{len(strict_pcts)} parse samples, baseline_avg={baseline_avg}")

    # Final-window estimate: mean of last ~20 logged rewards. Honest proxy for
    # what cell 23's eval would have produced if it had completed.
    trained_estimate = float(np.mean(rewards[-min(20, len(rewards)):]))
    print(f"trained_estimate (last-window mean): {trained_estimate:.4f}")

    if baseline_avg is None:
        baseline_avg = 0.0
        print("warning: baseline_avg not found in notebook; defaulting to 0")

    plot_grpo_results(rewards, strict_pcts, baseline_avg, trained_estimate)
    plot_redteam_curriculum()
    print("\ndone — both PNGs written to repo root.")


if __name__ == "__main__":
    main()
