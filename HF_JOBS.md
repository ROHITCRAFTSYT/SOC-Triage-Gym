# Training on Hugging Face Jobs

Run GRPO training as a one-shot HF Job — no Pro subscription required (uses
your monthly free-tier compute allowance). Results are pushed back to your
HF model + Space automatically.

## Prerequisites

- HF account with a write-scope token: <https://huggingface.co/settings/tokens>
- `huggingface_hub` CLI installed locally:
  ```bash
  pip install -U "huggingface_hub[cli]"
  hf auth login
  ```

## Launch the job

One command, GPU runs in the cloud, you watch logs locally:

```bash
hf jobs run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --env GIT_REF=main \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c 'apt-get update -qq && apt-get install -y -qq curl ca-certificates && curl -fsSL https://raw.githubusercontent.com/ROHITCRAFTSYT/-Metas-OpenEnv-2/main/scripts/hf_job_entrypoint.sh | bash'
```

**What this does** (≈30 min on `a10g-small`):

1. Pulls the latest CUDA pytorch image
2. Fetches `scripts/hf_job_entrypoint.sh` from your repo's main branch
3. The entrypoint clones the repo, installs the GRPO stack, starts the env
   server, runs `scripts/train_and_evaluate.py`, and uploads:
   - **Model:** trained LoRA → `rohitcraftsyt/soc-grpo-tier1`
   - **Space:** `training_loss.png`, `trained_vs_baseline.png`,
     `trained_vs_baseline.csv`, `training_summary.json` →
     `rohitcraftsyt/openenv2`

## Tunables

Pass them as `--env KEY=VALUE` to the launch command. Defaults shown.

| Variable | Default | Notes |
|---|---|---|
| `ROLE` | `tier1` | `tier1` / `tier2` / `manager` — staged training |
| `NUM_EPOCHS` | `2` | Bump to 3–4 if `a10g-large` flavor |
| `NUM_GENERATIONS` | `4` | Per-prompt completions for GRPO group sampling |
| `SOC_TRAIN_N_SEEDS` | `12` | Seeds × tasks × steps = total training examples |
| `SOC_TRAIN_TASKS` | (defaults from `train_grpo.py`) | Comma-separated, e.g. `phishing` for solo dense reward |
| `GIT_REF` | `main` | Branch / tag / commit SHA to train against |
| `HF_MODEL_REPO` | `rohitcraftsyt/soc-grpo-tier1` | Where the LoRA adapter lands |
| `HF_SPACE_REPO` | `rohitcraftsyt/openenv2` | Where the artifact files land |

Example with overrides:

```bash
hf jobs run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --env GIT_REF=main \
    --env NUM_EPOCHS=3 \
    --env SOC_TRAIN_TASKS=team_phishing_escalation \
    --env SOC_TRAIN_N_SEEDS=20 \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c 'apt-get update -qq && apt-get install -y -qq curl ca-certificates && curl -fsSL https://raw.githubusercontent.com/ROHITCRAFTSYT/-Metas-OpenEnv-2/main/scripts/hf_job_entrypoint.sh | bash'
```

## Watching progress

The launch command streams logs to your terminal. Look for:

- `[gpu] NVIDIA A10G | bf16=True` — confirms GPU attached
- `[reward_fn] parse: strict X% loose Y% fallback Z%` — JSON-validity health
  check during training. `strict` should rise above 60% by mid-epoch 1; if
  it stays under 30%, the model isn't learning to emit JSON.
- `[upload] pushing artifacts to HF` — final stage
- `[done] training + upload complete`

## Flavors and approximate cost

HF gives every account some free monthly compute. Beyond that:

| Flavor | GPU | $/hr (approx) | Use for |
|---|---|---|---|
| `cpu-basic` | none | free | n/a (GRPO needs GPU) |
| `t4-small` | T4 (16 GB) | ~$0.40 | Quick smoke test, 1 epoch |
| `t4-medium` | T4 (16 GB) | ~$0.60 | 2 epoch run, fine for tier1 |
| **`a10g-small`** | A10G (24 GB) | **~$1.05** | **Recommended — fits Qwen-1.5B comfortably, 30 min/run** |
| `a10g-large` | A10G x4 (96 GB) | ~$3.15 | Qwen-7B, full team training |
| `a100-large` | A100 (80 GB) | ~$4.50 | Overkill for this scale |

Check `hf jobs ps` for running jobs and `hf jobs logs <job-id>` for past runs.

## Troubleshooting

**"HF_TOKEN secret not set"** — your token wasn't picked up. Either run `hf auth login` first then use `--secrets HF_TOKEN` (no value, picks up your stored token), or pass explicitly:
```bash
--secrets HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**"Job image specification error: Invalid image name `--secret`"** — you used `--secret` (singular). The correct flag is `--secrets` (plural).

**"bash: curl ... | bash: No such file or directory"** — you used `bash -lc` instead of `bash -c`. The `hf jobs` CLI parses `-l` as its `--label` flag (consuming `c` as the label value), so the remote container only ran `bash 'curl ...'` with no `-c` flag. Use `bash -c` and the `-l` flag conflict goes away.

**"bash: line 1: curl: command not found"** — the pytorch base image doesn't ship with curl. Install it inside the bash command before using it: `bash -c 'apt-get update -qq && apt-get install -y -qq curl ca-certificates && curl -fsSL ... | bash'` (already done in the launch commands above).

**Job runs but no upload** — your token doesn't have write scope. Regenerate
at <https://huggingface.co/settings/tokens> with **Write** permission.

**`unsloth` install fails** — the base CUDA pytorch image may be too new or
too old. Try `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` instead of 2.6.0.

**Server never becomes healthy** — check the `[fatal] server died` log. Usually
a missing dep on Python 3.12; try the `python:3.11-slim` image with explicit
CUDA install instead.

## Alternative: free Colab / Kaggle

If you don't want to use HF Jobs compute hours, the same training runs on:

- **Google Colab** (free T4): open `soc_triage_gym_v2_training.ipynb`
- **Kaggle** (free T4, 30 hr/week): clone the repo and run
  `python scripts/train_and_evaluate.py`

Both push to HF the same way — see the upload block at the bottom of
`scripts/hf_job_entrypoint.sh` for the `huggingface_hub` API calls.
