# Shared configuration sourced by every run_<env>.sh script.
# Edit values here (model, sampling, agent) to change them for all environments.
# Anything set on the command line after the script name is forwarded and wins.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Model used as both student and teacher (hierarchical agent).
MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"

COMMON_ARGS=(
  --no-log-wandb
  --wandb-mode offline
  --record-stats
  --no-use-distributed
  --num-eval-workers 1
  --llm-agent hierarchical
  --max-history 1
  --refinement-prompt.feedback-mode feedback
  --refinement-prompt.teacher-reference-mode none
  --refinement-prompt.teacher-prompt-style reasoning_feedback_tag
  --refinement-prompt.teacher-parse-feedback-tags
  --llm-actor.actor-type vllm
  --llm-actor.engine-args.model-id "$MODEL_ID"
  --llm-actor.engine-args.tokenizer-id "$MODEL_ID"
  --llm-actor.engine-args.max-model-len 32768
  --llm-actor.engine-args.gpu-memory-utilization 0.82
  --llm-actor.engine-args.dtype bfloat16
  --llm-actor.engine-args.no-enable-thinking
  --llm-actor.sampling-args.max-tokens 2048
  --llm-actor.sampling-args.temperature 0.0
  --llm-actor.sampling-args.top-p 1.0
)
