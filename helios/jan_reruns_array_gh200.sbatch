#!/usr/bin/env bash
#SBATCH --job-name=jan-reruns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/net/storage/pr3/plgrid/plggcontrastiverl/jl/work/plan-crl-jan-reruns-20260525}"
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_vllm0210_cu129_flash_py311}"
export CONFIG_DIR="${CONFIG_DIR:-}"
export CONFIG_GLOB="${CONFIG_GLOB:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"

JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
export PLANCRL_RUN_TMP_ROOT="${PLANCRL_RUN_TMP_ROOT:-/tmp/p-${USER:-u}}"
export RAY_TMPDIR="$PLANCRL_RUN_TMP_ROOT/ray-$JOB_TAG"
export TMPDIR="$PLANCRL_RUN_TMP_ROOT/tmp-$JOB_TAG"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

rm -rf "$RAY_TMPDIR" "$TMPDIR"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

cd "$REPO_ROOT"
source "$REPO_ROOT/helios/code_env_gh200.sh"
source "$VENV_DIR/bin/activate"

python - <<'PY'
import plan_crl
print("plan_crl", plan_crl.__file__)
PY

if [ -n "$CONFIG_DIR" ]; then
  mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "config_*" | sort -V)
elif [ -n "$CONFIG_GLOB" ]; then
  CONFIGS=( $CONFIG_GLOB )
else
  echo "Set CONFIG_DIR or CONFIG_GLOB to materialized mrunner configs." >&2
  exit 2
fi

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "No configs matched CONFIG_DIR=$CONFIG_DIR CONFIG_GLOB=$CONFIG_GLOB" >&2
  exit 2
fi

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "job=$SLURM_JOB_ID array_task=${SLURM_ARRAY_TASK_ID:-0} config=$CONFIG"
python mrunner_run.py --config "$CONFIG"
