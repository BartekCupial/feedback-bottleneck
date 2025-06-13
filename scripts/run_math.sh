#!/usr/bin/env bash
# Omni-MATH: olympiad-level mathematical reasoning.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m scripts.collect_episodes \
  --exp-name math \
  --env-name math \
  --math-args.dataset-path LLParallax/Omni-MATH-filtered \
  --math-args.max-turns 2 \
  --math-args.verification-mode algorithmic \
  --num-eval-episodes 256 \
  --output-dir ./runs/math \
  "${COMMON_ARGS[@]}" \
  "$@"
