#!/usr/bin/env bash
# Linguini (from BIG-Bench Extra Hard / BBEH): learning new rules.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m scripts.collect_episodes \
  --exp-name linguini \
  --env-name bbeh \
  --task linguini \
  --bbeh-args.max-turns 2 \
  --num-eval-episodes 256 \
  --output-dir ./runs/linguini \
  "${COMMON_ARGS[@]}" \
  "$@"
