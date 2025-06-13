#!/usr/bin/env bash
# ARC-AGI: grid-transformation puzzles.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m scripts.collect_episodes \
  --exp-name arc_agi \
  --env-name arc_agi \
  --task arc_agi_public \
  --arc-agi-args.split evaluation \
  --arc-agi-args.max-tasks 120 \
  --arc-agi-args.max-turns 2 \
  --num-eval-episodes 120 \
  --output-dir ./runs/arc_agi \
  "${COMMON_ARGS[@]}" \
  "$@"
