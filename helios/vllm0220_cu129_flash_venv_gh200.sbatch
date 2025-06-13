#!/bin/bash -l
#SBATCH --job-name=plancrl-vllm022-cu129-flash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"

export REPO_ROOT
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_vllm0220_cu129_flash_py311}"
export PLANCRL_BUILD_ROOT="${PLANCRL_BUILD_ROOT:-$REPO_ROOT/.cache/plan_crl_build_vllm0220_cu129_flash}"
export VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
export VLLM_GIT_REF="${VLLM_GIT_REF:-v${VLLM_VERSION}}"
export CUDA_BACKEND="${CUDA_BACKEND:-cu129}"
export INSTALL_CUDA13_RUNTIME="${INSTALL_CUDA13_RUNTIME:-1}"
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
export INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-1}"
export INSTALL_XGRAMMAR="${INSTALL_XGRAMMAR:-1}"
export INSTALL_PLAN_CRL_EDITABLE="${INSTALL_PLAN_CRL_EDITABLE:-0}"
export RESET_VENV="${RESET_VENV:-1}"

exec bash "$REPO_ROOT/helios/vllm0210_cu129_flash_venv_gh200.sbatch"
