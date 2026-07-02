#!/usr/bin/env python3
"""
Generate README charts from authoritative repo metadata — no server, no GPU.

Outputs (written to assets/):
  - task_landscape.png        alerts vs. horizon per task, sized by difficulty
  - efficiency_curve.png      the step-budget efficiency multiplier
  - theme_coverage_matrix.png hackathon theme / sub-theme coverage grid

Every value below is sourced from committed code so the charts stay honest:
  * task table            -> server/app.py TASKS + README "Tasks" table
  * efficiency multiplier -> server/environment.py _efficiency_multiplier()
  * theme coverage        -> README "Theme Coverage" table + GET /themes/coverage

Run:  python scripts/gen_readme_assets.py   (or `make plots`)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# SOC-dashboard palette (dark).
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
ACCENT = "#58a6ff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.edgecolor": GRID,
    "font.size": 11,
})

# task_id, alerts, max_steps, difficulty_rank(1..6), mode
TASKS = [
    ("phishing", 1, 15, 1, "solo"),
    ("lateral_movement", 5, 30, 2, "solo"),
    ("queue_management", 20, 60, 4, "solo"),
    ("insider_threat", 30, 80, 5, "solo"),
    ("team_phishing_escalation", 1, 68, 1, "team"),
    ("team_lateral_team", 8, 68, 2, "team"),
    ("apt_campaign", 60, 250, 6, "solo"),
    ("red_team_generated", 12, 250, 3, "adaptive"),
]

MODE_COLOR = {"solo": "#58a6ff", "team": "#f778ba", "adaptive": "#3fb950"}


def gen_task_landscape() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for name, alerts, steps, rank, mode in TASKS:
        ax.scatter(
            steps, alerts,
            s=180 + rank * 240,
            c=MODE_COLOR[mode],
            alpha=0.55, edgecolors=FG, linewidths=1.2, zorder=3,
        )
        ax.annotate(
            name, (steps, alerts),
            textcoords="offset points", xytext=(0, 14 + rank * 2),
            ha="center", fontsize=9, color=FG,
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("Episode horizon  (max steps, log scale)")
    ax.set_ylabel("Alerts in queue  (log scale)")
    ax.set_title("SOC-Triage-Gym task landscape — horizon × queue size × difficulty",
                 fontweight="bold", pad=14)
    ax.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_xlim(10, 400)
    ax.set_ylim(0, 120)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=11,
                   markerfacecolor=c, markeredgecolor=FG, label=m.capitalize())
        for m, c in MODE_COLOR.items()
    ]
    ax.legend(handles=handles, title="Mode", loc="upper left",
              facecolor=BG, edgecolor=GRID, labelcolor=FG, title_fontsize=10)
    ax.text(0.99, 0.02, "bubble size ∝ difficulty tier (easy → super-hard)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#8b949e")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "task_landscape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def gen_efficiency_curve() -> None:
    # server/environment.py::_efficiency_multiplier
    xs = np.linspace(0, 1.0, 501)

    def mult(r: float) -> float:
        if r <= 0.50:
            return 1.0
        if r <= 0.75:
            return 1.0
        if r <= 0.90:
            return 0.85
        return 0.70

    ys = [mult(r) for r in xs]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.step(xs * 100, ys, where="post", color=ACCENT, linewidth=2.6, zorder=3)
    ax.fill_between(xs * 100, ys, 0.6, step="post", color=ACCENT, alpha=0.12, zorder=2)
    for frac, label in [(0.75, "75%"), (0.90, "90%")]:
        ax.axvline(frac * 100, color="#f85149", linestyle="--", alpha=0.6)
        ax.text(frac * 100, 1.02, label, ha="center", color="#f85149", fontsize=9)
    ax.set_xlabel("Step budget consumed  (%)")
    ax.set_ylabel("Reward efficiency multiplier")
    ax.set_title("Final-score efficiency multiplier — solve early or pay a tax",
                 fontweight="bold", pad=12)
    ax.set_ylim(0.6, 1.06)
    ax.set_xlim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    for xv, yv in [(25, 1.0), (82, 0.85), (95, 0.70)]:
        ax.annotate(f"×{yv:.2f}", (xv, yv), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=FG, fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "efficiency_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def gen_theme_matrix() -> None:
    rows = [
        ("Theme #1 — Multi-Agent (primary)", "3-role team + ticket bus + phase FSM"),
        ("Theme #2 — Long-Horizon", "apt_campaign: 250 steps, 5 phases"),
        ("Theme #3.1 — Professional Tasks", "Real SOC tool loop (SIEM/EDR/IAM)"),
        ("Theme #4 — Self-Improvement", "Adaptive red-team curriculum"),
        ("Fleet AI — Scalable Oversight", "Manager audits + LLM judge"),
        ("Halluminate — Multi-Actor", "3 external NPC actors -> inboxes"),
        ("Mercor — Token-Scaled Rewards", "Length bonus × quality gate"),
        ("Patronus — Schema/Policy Drift", "Mid-episode policy changes"),
        ("Scaler AI — Multi-App Enterprise", "Cross-app rule: disable needs P1/P2"),
        ("Scale AI — Non-code IT Workflow", "SLA ticketing system"),
        ("Snorkel — Experts-in-the-Loop", "Rotating expert panel + weights"),
    ]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.axis("off")
    ax.set_title("Hackathon theme coverage — one Space, 11 tracks",
                 fontweight="bold", fontsize=14, pad=10, color=FG)
    n = len(rows)
    for i, (theme, how) in enumerate(rows):
        y = n - i
        box = FancyBboxPatch(
            (0.02, y - 0.42), 0.96, 0.82,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            transform=ax.transData, facecolor="#161b22",
            edgecolor=GRID, linewidth=1.0, zorder=1,
        )
        ax.add_patch(box)
        ax.text(0.05, y, "✓", color="#3fb950", fontsize=15, fontweight="bold",
                va="center", zorder=2)
        ax.text(0.11, y, theme, color=FG, fontsize=11, fontweight="bold",
                va="center", zorder=2)
        ax.text(0.60, y, how, color="#8b949e", fontsize=10, va="center", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, n + 0.7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "theme_coverage_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    gen_task_landscape()
    gen_efficiency_curve()
    gen_theme_matrix()
    print(f"\nAll README assets written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
