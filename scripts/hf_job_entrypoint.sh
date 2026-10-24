#!/usr/bin/env bash
# scripts/hf_job_entrypoint.sh
#
# Self-contained training entrypoint for `hf jobs run`. Designed to be
# launched against a CUDA pytorch base image — clones the repo, installs
# the GRPO stack, runs training, and uploads the artifacts back to HF.
#
# Required env vars (pass via `hf jobs run --secret`):
#   HF_TOKEN              — write-scope token used to push model + space artifacts
#
# Optional env vars (pass via `hf jobs run --env`):
#   GIT_REPO              — github clone URL (default: ROHITCRAFTSYT/-Metas-OpenEnv-2)
#   GIT_REF               — branch/tag/sha to check out (default: main)
#   HF_MODEL_REPO         — target model repo for the LoRA (default: rohitcraftsyt/soc-grpo-tier1)
#   HF_SPACE_REPO         — target space repo for plot/csv/json artifacts (default: rohitcraftsyt/openenv2)
#   NUM_EPOCHS            — GRPO epochs (default: 2)
#   NUM_GENERATIONS       — completions per prompt (default: 4 — fits T4/A10g-small)
#   SOC_TRAIN_N_SEEDS     — seeds per task (default: 12)
#   SOC_TRAIN_TASKS       — comma-separated task IDs (default: train_grpo.py defaults)
#   ROLE                  — tier1 | tier2 | manager (default: tier1)

set -euo pipefail

# --- defaults --------------------------------------------------------------
GIT_REPO="${GIT_REPO:-https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2.git}"
GIT_REF="${GIT_REF:-main}"
HF_MODEL_REPO="${HF_MODEL_REPO:-rohitcraftsyt/soc-grpo-tier1}"
HF_SPACE_REPO="${HF_SPACE_REPO:-rohitcraftsyt/openenv2}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export SOC_TRAIN_N_SEEDS="${SOC_TRAIN_N_SEEDS:-12}"
export ROLE="${ROLE:-tier1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[fatal] HF_TOKEN secret not set. Re-run with:" >&2
  echo "  hf jobs run --secret HF_TOKEN=hf_xxx ..." >&2
  exit 1
fi

# --- system packages -------------------------------------------------------
echo "[setup] apt-get install git curl"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates >/dev/null

# --- repo ------------------------------------------------------------------
WORKDIR=/repo
echo "[setup] clone $GIT_REPO @ $GIT_REF -> $WORKDIR"
git clone --depth=1 --branch "$GIT_REF" "$GIT_REPO" "$WORKDIR"
cd "$WORKDIR"

# --- python deps -----------------------------------------------------------
echo "[setup] pip install env + GRPO stack"
# Upgrade pip + setuptools + wheel before any editable install: the pytorch
# base image ships with old setuptools that lacks setuptools.backends.legacy,
# which our pyproject.toml build backend imports. Without this, the next
# `pip install -e` fails with "BackendUnavailable: Cannot import
# 'setuptools.backends.legacy'".
pip install --upgrade --quiet "pip>=24" "setuptools>=70" "wheel>=0.43"
pip install -e ".[dev]" --quiet
pip install --quiet \
    "transformers<5" \
    "trl>=0.11,<0.25" \
    "peft>=0.14" \
    "accelerate>=0.34" \
    "datasets>=3" \
    "bitsandbytes>=0.43" \
    matplotlib \
    huggingface_hub
# unsloth installs a bunch of pinned wheels — must come last so it can pin its
# own torch/triton without earlier packages overriding.
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# --- env server ------------------------------------------------------------
echo "[setup] start FastAPI server on :7860"
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860 \
    > /tmp/uvicorn.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

# wait for /health
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:7860/health >/dev/null 2>&1; then
        echo "[setup] server ready ($i s)"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[fatal] server died — last log:" >&2
        tail -50 /tmp/uvicorn.log >&2
        exit 1
    fi
    sleep 1
done

# --- gpu check -------------------------------------------------------------
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — relaunch with --flavor a10g-small or t4-small"
print(f"[gpu] {torch.cuda.get_device_name(0)} | bf16={torch.cuda.is_bf16_supported()}")
PY

# --- train + evaluate ------------------------------------------------------
echo "[train] starting GRPO ROLE=$ROLE epochs=$NUM_EPOCHS generations=$NUM_GENERATIONS seeds=$SOC_TRAIN_N_SEEDS"
python scripts/train_and_evaluate.py

# --- upload artifacts ------------------------------------------------------
echo "[upload] pushing artifacts to HF"
python - <<PY
import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()

ckpt = f"checkpoints/soc_grpo_${ROLE}"
if os.path.isdir(ckpt):
    print(f"  -> model: ${HF_MODEL_REPO} (folder {ckpt})")
    api.create_repo("${HF_MODEL_REPO}", repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=ckpt,
        repo_id="${HF_MODEL_REPO}",
        repo_type="model",
        commit_message=f"GRPO ${ROLE} checkpoint",
    )

artifacts = [
    "training_loss.png",
    "trained_vs_baseline.png",
    "trained_vs_baseline.csv",
    "training_summary.json",
    "reward_curve_tier1_oracle.png",
    "reward_comparison_baseline_tier1.png",
]
for f in artifacts:
    if os.path.exists(f):
        print(f"  -> space: ${HF_SPACE_REPO}/{f}")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f,
            repo_id="${HF_SPACE_REPO}",
            repo_type="space",
            commit_message=f"hf-job artifact: {f}",
        )
PY

echo "[done] training + upload complete"
