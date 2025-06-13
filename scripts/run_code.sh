#!/usr/bin/env bash
# Codeforces: competitive programming.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m scripts.collect_episodes \
  --exp-name code \
  --env-name code \
  --code-args.dataset-path codeparrot/apps \
  --code-args.max-turns 2 \
  --code-args.sandbox-case-timeout-s 2.0 \
  --num-eval-episodes 256 \
  --output-dir ./runs/code \
  "${COMMON_ARGS[@]}" \
  "$@"
