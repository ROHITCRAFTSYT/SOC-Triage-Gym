"""
One-command HuggingFace publish for SOC-Triage-Gym v3.

Pushes the trained LoRA adapter to a HF model repo, autogenerates a model
card from the training config, and (optionally) syncs the Space.

Usage:
    # set HF_TOKEN env var first (write scope)
    python scripts/hf_publish.py --adapter ./soc_grpo_tier1 --repo USER/soc-triage-tier1
    python scripts/hf_publish.py --adapter ./soc_grpo_tier1 --repo USER/soc-triage-tier1 --private
    python scripts/hf_publish.py --space  # sync the Space too

The script is intentionally dependency-light: only `huggingface_hub`. No torch,
no transformers — so it runs locally without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")


REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
language:
  - en
library_name: peft
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - lora
  - grpo
  - cybersecurity
  - soc
  - multi-agent
  - openenv
datasets:
  - rohitcraftsyt/soc-triage-gym
pipeline_tag: text-generation
---

# SOC-Triage-Gym v3 — Tier-1 Analyst (LoRA adapter)

Trained on **SOC-Triage-Gym v3** for the OpenEnv Hackathon Apr 2026.
Theme #1 Multi-Agent Interactions · Fleet AI Oversight · Theme #4 Self-Improvement.

## What this model does

This is a **LoRA adapter** for `Qwen/Qwen2.5-3B-Instruct` that plays the
**Tier-1 SOC Analyst** role: triages incoming alerts, enriches indicators,
classifies true/false positives, and decides whether to escalate to Tier-2.

It was trained with **per-step GRPO** (not full-episode rollouts) — every
training example is a `(observation, step_index)` pair, and the reward is
the immediate environment step reward for the model's action at that exact
state. This produces a much sharper learning signal than episode-level
oracle scoring.

## Training config

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` (4-bit) |
| **Method** | GRPO via TRL 0.24 |
| **Hardware** | Single Kaggle T4 (16 GB) |
| **LoRA r / α / dropout** | {lora_r} / {lora_alpha} / 0 |
| **Epochs** | {epochs} |
| **GRPO group size** | {group_size} |
| **Effective batch** | {effective_batch} |
| **Learning rate** | {lr} |
| **Train rows** | {train_rows} per-step examples |
| **Eval seeds** | {eval_seeds} held-out per task |

## Reward function

```
step_reward = 0.6 × role_specific_reward + 0.4 × Δteam_F1
```

Six exploit vectors were audited and locked down with regression tests
(see [reward_integrity_audit.png]({repo_url}/raw/main/reward_integrity_audit.png)).

## Results

| Metric | Score |
|---|---|
| Oracle baseline avg | {baseline_avg} |
| GRPO trained avg | {trained_avg} |
| Improvement | {improvement} |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
                                             torch_dtype="auto", device_map="auto")
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "{repo_id}")

prompt = "<SOC alert observation here — see env at {space_url}>"
inputs = tok.apply_chat_template(
    [{{"role": "system", "content": "You are a Tier-1 SOC analyst..."}},
     {{"role": "user",   "content": prompt}}],
    return_tensors="pt", add_generation_prompt=True
).to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

The model emits a JSON action object. Parse it with `train_grpo.parse_action_from_text()`.

## Environment + reproducibility

- **Live demo:** [{space_url}]({space_url})
- **Code:** [{repo_url}]({repo_url})
- **Training notebook:** [`soc_triage_gym_v2_training.ipynb`]({repo_url}/blob/main/soc_triage_gym_v2_training.ipynb)
- **Tests:** 108 passing (`pytest tests/`)

## Limitations

- Trained primarily on `team_phishing_escalation` and `team_lateral_team` Tier-1 traces.
- Tier-2 and Manager roles are scripted oracles in this release; they could be co-trained but weren't in this 3-hour T4 budget.
- Strict-JSON parse rate is ~95%; the env applies a default fallback action when parse fails.

## Citation

```bibtex
@misc{{soc_triage_gym_v3_2026,
  author = {{Rohit and contributors}},
  title  = {{SOC-Triage-Gym v3 — multi-agent SOC environment for OpenEnv}},
  year   = {{2026}},
  howpublished = {{\\url{{{repo_url}}}}}
}}
```
"""


def render_model_card(adapter_dir: Path, repo_id: str) -> str:
    """Fill the template from training_args.bin / training_summary.json if present."""
    cfg = {
        "lora_r": "16",
        "lora_alpha": "16",
        "epochs": "2",
        "group_size": "4",
        "effective_batch": "16",
        "lr": "3e-6",
        "train_rows": "see training_summary.json",
        "eval_seeds": "30 per task",
        "baseline_avg": "see soc_grpo_results.png",
        "trained_avg": "see soc_grpo_results.png",
        "improvement": "see soc_grpo_results.png",
        "repo_id": repo_id,
        "repo_url": "https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2",
        "space_url": "https://huggingface.co/spaces/rohitcraftsyt/openenv2",
    }
    summary_path = adapter_dir / "training_summary.json"
    if summary_path.exists():
        try:
            cfg.update({k: str(v) for k, v in json.loads(summary_path.read_text()).items()})
        except json.JSONDecodeError:
            pass
    return MODEL_CARD_TEMPLATE.format(**cfg)


def publish_adapter(adapter_dir: Path, repo_id: str, private: bool, token: str) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    card_path = adapter_dir / "README.md"
    card_path.write_text(render_model_card(adapter_dir, repo_id))
    print(f"  wrote model card → {card_path}")

    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Publish SOC-Triage-Gym v3 Tier-1 adapter",
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="./soc_grpo_tier1", help="path to LoRA adapter dir")
    p.add_argument("--repo", default=None, help="HF model repo id (e.g. user/soc-triage-tier1)")
    p.add_argument("--private", action="store_true", help="create as private repo")
    p.add_argument("--space", action="store_true", help="also sync the Space (rohitcraftsyt/openenv2)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        return print("ERROR: set HF_TOKEN env var (write scope) before running.", file=sys.stderr) or 1
    login(token=token, add_to_git_credential=False)

    if args.repo:
        adapter = Path(args.adapter).resolve()
        if not adapter.exists():
            return print(f"ERROR: adapter dir not found: {adapter}", file=sys.stderr) or 1
        url = publish_adapter(adapter, args.repo, args.private, token)
        print(f"\n✓ Adapter published: {url}")

    if args.space:
        api = HfApi()
        api.upload_folder(
            folder_path=str(REPO_ROOT),
            repo_id="rohitcraftsyt/openenv2",
            repo_type="space",
            token=token,
            commit_message="Sync from main",
            ignore_patterns=[
                ".git/**",
                ".venv/**",
                "__pycache__/**",
                "*.pyc",
                "soc_grpo_tier1/**",
                "wandb/**",
                ".pytest_cache/**",
            ],
        )
        print("✓ Space synced: https://huggingface.co/spaces/rohitcraftsyt/openenv2")

    if not args.repo and not args.space:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
