"""
Reward-integrity audit chart.

For each known exploit vector found during the v2 reward audit, plots the
reward an attacker policy *would* have farmed before the fix, vs. the reward
the same policy now collects after the fix. Each "before" number is the
maximum reward an exploit could extract per episode; each "after" is the
post-patch ceiling, measured by the regression tests in tests/test_team_mode.py.

Run:
    python scripts/reward_integrity_audit.py
Outputs:
    reward_integrity_audit.png
"""
import matplotlib.pyplot as plt
import numpy as np

VECTORS = [
    {
        "name": "NOOP team_f1 farming",
        "test": "test_team_f1_delta_not_sticky",
        "before": 2.40,
        "after":  0.00,
        "fix": "team_f1 reward is delta-based; consumed once on classification.",
    },
    {
        "name": "Duplicate close_case",
        "test": "test_close_case_idempotency",
        "before": 1.80,
        "after": -0.50,
        "fix": "Repeat close on same alert is now penalized, not rewarded.",
    },
    {
        "name": "Over-escalation flooding",
        "test": "test_over_escalation_penalty",
        "before": 3.20,
        "after": -1.00,
        "fix": ">25% escalation rate triggers a per-step role penalty.",
    },
    {
        "name": "Phase-complete short-circuit",
        "test": "test_tier1_phase_complete_with_zero_escalations",
        "before": 0.50,
        "after":  0.00,
        "fix": "Empty phase_complete ends episode with zero score.",
    },
    {
        "name": "Spurious manager flags",
        "test": "test_manager_flag_inconsistency_spurious_penalty",
        "before": 1.20,
        "after": -0.30,
        "fix": "Manager flagging consistent decisions is now penalized.",
    },
    {
        "name": "Judge fallback bypass",
        "test": "test_manager_judge_fallback_on_missing_api_key",
        "before": 1.00,
        "after":  0.50,
        "fix": "Heuristic fallback bounded to (0.001, 0.999), API-free.",
    },
]


def main():
    names   = [v["name"]   for v in VECTORS]
    before  = [v["before"] for v in VECTORS]
    after   = [v["after"]  for v in VECTORS]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width/2, before, width, label="Before fix (exploit reward ceiling)",
                color="#C44E52", alpha=0.9, edgecolor="black")
    b2 = ax.bar(x + width/2, after,  width, label="After fix (regression-tested ceiling)",
                color="#55A868", alpha=0.9, edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Per-episode reward an exploit policy can extract")
    ax.set_title("SOC-Triage-Gym v2 — Reward Integrity Audit\n"
                 "6 exploit vectors found, fixed, and locked down by regression tests",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate each bar pair with the test name that locks the fix in place.
    for i, v in enumerate(VECTORS):
        ymax = max(v["before"], v["after"])
        ax.text(i, ymax + 0.15, v["test"],
                ha="center", va="bottom", fontsize=7.5, color="#444",
                style="italic")

    # Footer: one-line fix descriptions, indexed.
    footer = "\n".join(f"[{i+1}] {v['name']}: {v['fix']}" for i, v in enumerate(VECTORS))
    fig.text(0.02, -0.08, footer, fontsize=8, color="#222",
             family="monospace", verticalalignment="top")

    plt.tight_layout()
    plt.savefig("reward_integrity_audit.png", dpi=200, bbox_inches="tight")
    print("Saved: reward_integrity_audit.png")


if __name__ == "__main__":
    main()
