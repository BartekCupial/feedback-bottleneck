from dataclasses import dataclass, field
from typing import Literal, Optional, Union

@dataclass
class NLEArgs:
    character: str = "@"
    """Character representing the agent in NLE"""
    max_episode_steps: int = 100_000
    """Max steps per episode in NLE"""
    savedir: Optional[str] = None
    """Directory to save NLE data; null disables saving"""
    save_ttyrec_every: int = 0
    """Frequency of saving TTY recordings"""
    skip_more: bool = True
    """Whether to skip the 'more' prompt in NLE"""
    no_progress_abort: int = 150
    """Timeout for no progress in NLE"""
    allow_all_yn_questions: bool = True
    """Controls how the environment handles Yes/No questions.If False The environment will automatically 'press Escape'"""
    allow_all_modes: bool = True
    """If False The environment automatically clears "More" prompts and declines text input prompts/menus."""
    penalty_mode: str = "constant"
    """Penalty mode for NLE"""
    penalty_step: float = -0.0
    """Step penalty in NLE"""
    penalty_time: float = -0.0
    """Time penalty per step in NLE"""
    provide_map: bool = True
    """Whether to provide map observations in NLE"""


@dataclass
class DatasetEnvArgs:
    dataset_path: Optional[str] = None
    """Optional dataset identifier or local path for dataset-backed environments"""
    dataset_config: Optional[str] = None
    """Optional dataset config name used by datasets.load_dataset"""
    dataset_split: str = "train"
    """Dataset split to load for collection/evaluation"""
    problem_split: Literal["all", "train", "test"] = "all"
    """Optional problem_id split applied within the loaded environment dataset"""
    train_problem_split: Literal["all", "train", "test"] = "all"
    """Optional problem_id split applied to the tokenized training dataset"""
    problem_split_seed: int = 0
    """Seed used to derive the deterministic problem_id split"""
    problem_split_test_size: int = 512
    """Exact number of unique problem_ids assigned to the test split"""


@dataclass
class MathArgs(DatasetEnvArgs):
    dataset_path: str = "LLParallax/Omni-MATH-filtered"
    """Optional dataset identifier or local file path overriding the built-in task aliases"""
    max_turns: int = 1
    """Maximum number of student attempts before the episode truncates"""
    max_action_length: int = 65536
    """Soft cap for free-form math answers sent to the environment"""
    verification_mode: Literal["algorithmic", "llm"] = "algorithmic"
    """How to decide whether a math answer is solved"""
    verifier_sampling_args: "SamplingArgs" = field(default_factory=lambda: SamplingArgs())
    """Sampling parameters used by the math LLM verifier"""


@dataclass
class CountdownArgs:
    max_turns: int = 1
    """Maximum number of attempts before truncation"""
    max_action_length: int = 65536
    """Soft cap for free-form answers sent to the countdown environment"""


@dataclass
class BBehArgs(DatasetEnvArgs):
    dataset_path: str = "BBEH/bbeh"
    """Hugging Face dataset id or local cached dataset path for BBEH"""
    max_turns: int = 1
    """Maximum number of attempts before truncation"""
    max_action_length: int = 65536
    """Soft cap for free-form answers sent to the BBEH environment"""


@dataclass
class SciKnowEvalArgs(DatasetEnvArgs):
    dataset_path: str = "hicai-zju/SciKnowEval"
    """Hugging Face dataset id or local cached dataset path for SciKnowEval"""
    dataset_config: str = "v2"
    """SciKnowEval dataset config to load"""
    dataset_split: str = "test"
    """SciKnowEval split to load"""
    domains: tuple[str, ...] = ()
    """Optional domain filter such as ('Biology',) or ('Biology', 'Chemistry')"""
    levels: tuple[str, ...] = ()
    """Optional level filter such as ('L3',)"""
    question_types: tuple[str, ...] = ()
    """Optional question type filter such as ('mcq-4-choices', 'mcq-2-choices')"""
    max_turns: int = 1
    """Maximum number of attempts before truncation"""
    max_action_length: int = 65536
    """Soft cap for free-form answers sent to the SciKnowEval environment"""


@dataclass
class ArcAgiArgs:
    version: Literal["1", "2"] = "1"
    """ARC-AGI dataset version: '1' for the original fchollet/ARC-AGI, '2' for arcprize/ARC-AGI-2."""
    data_dir: Optional[str] = None
    """Optional path to an ARC-AGI `data` directory or split directory."""
    cache_dir: str = "~/.cache/plan-crl/arc-agi"
    """Cache directory used when downloading ARC-AGI from GitHub."""
    split: Literal["training", "evaluation"] = "evaluation"
    """ARC-AGI-2 split to evaluate."""
    download_if_missing: bool = True
    """Download ARC-AGI-2 into cache_dir when data_dir is not provided and the cache is empty."""
    max_tasks: Optional[int] = None
    """Optional cap on loaded ARC tasks."""
    max_turns: int = 2
    """Maximum full-grid attempts before truncation."""
    max_action_length: int = 65536
    """Soft cap for free-form ARC answers sent to the environment."""
    feedback_max_chars: int = 2000
    """Maximum length of structured ARC feedback stored in observations."""


@dataclass
class CodeArgs:
    dataset_path: str = "codeparrot/apps"
    """HF dataset id or local path for programming tasks."""
    dataset_config: Optional[str] = None
    """Optional dataset config/subset name."""
    dataset_split: str = "train"
    """Dataset split to load."""
    source_name: str = ""
    """Optional logical source name stored in metadata."""
    max_turns: int = 10
    """Maximum number of student attempts per task."""
    max_action_length: int = 65536
    """Maximum length of generated code stored in the environment."""
    max_cases: Optional[int] = 16
    """Optional cap on number of tests per task."""
    min_rating: Optional[int] = None
    """Optional minimum difficulty/rating filter."""
    max_rating: Optional[int] = None
    """Optional maximum difficulty/rating filter."""
    require_official_tests_complete: bool = False
    """Keep only tasks with complete official tests when the dataset exposes this flag."""
    require_stdio: bool = False
    """Keep only stdio tasks when input_mode is present."""
    require_no_generated_checker: bool = False
    """Keep only tasks without a generated checker when present."""
    required_tags: tuple[str, ...] = ()
    """Optional tag filters that must all be present."""
    sandbox_case_timeout_s: float = 2.0
    """Per-test timeout for candidate execution."""
    sandbox_task_timeout_s: Optional[float] = None
    """Optional timeout for the whole candidate across all tests. If null, it can be derived from the case timeout."""
    sandbox_use_dataset_time_limit: bool = False
    """Whether to scale execution timeouts from dataset-provided time limits when available."""
    sandbox_time_limit_multiplier: float = 10.0
    """Multiplier applied to dataset time_limit when dataset-aware timeouts are enabled."""
    sandbox_min_case_timeout_s: Optional[float] = None
    """Optional lower clamp for the effective per-case timeout."""
    sandbox_max_case_timeout_s: Optional[float] = None
    """Optional upper clamp for the effective per-case timeout."""
    sandbox_task_timeout_case_budget_multiplier: float = 2.0
    """If task timeout is derived automatically, use case_timeout * num_cases * this multiplier."""
    feedback_max_chars: int = 2000
    """Maximum length of sandbox feedback stored in observations."""
    problem_split: Literal["all", "train", "test"] = "all"
    """Optional split by unique task_id to avoid leakage."""
    train_problem_split: Literal["all", "train", "test"] = "all"
    """Optional problem_id split applied to the tokenized code training dataset"""
    problem_split_seed: int = 0
    """Seed for task split shuffling."""
    problem_split_test_size: Optional[int] = None
    """Number of unique tasks reserved for the held-out split."""


@dataclass
class EnterpriseOpsArgs:
    dataset_name: str = "ServiceNow-AI/EnterpriseOps-Gym"
    """HuggingFace dataset containing EnterpriseOps task prompts and metadata."""
    seed_db_root: str = "~/.cache/enterprise_ops"
    """Root directory containing EnterpriseOps seed database assets."""
    domains: tuple[str, ...] = ()
    """Optional domain filter, for example ('calendar',)."""
    modes: tuple[str, ...] = ("oracle",)
    """Optional task mode filter, for example ('oracle',)."""
    server_url_overrides: dict[str, str] = field(default_factory=dict)
    """MCP server URL overrides keyed by EnterpriseOps gym name."""
    max_turns: int = 50
    """Maximum tool-use turns per task."""
    max_action_length: int = 65536
    """Maximum length of generated enterprise ops actions."""


@dataclass
class RefinementPromptArgs:
    feedback_mode: Literal["feedback", "no_feedback"] = "feedback"
    """Whether refinement prompts include teacher feedback text or use only the previous attempt."""
    teacher_reference_mode: Literal["solution", "answer", "none"] = "solution"
    """Whether the teacher sees the full solution, only the final answer, or no ground truth at all."""
    teacher_feedback_word_limit: int = 0
    """Optional word limit for the teacher's final feedback tag; 0 disables the limit."""
    teacher_execution_context_mode: Literal["structured", "feedback", "traceback", "none"] = "structured"
    """Which execution-side signals the teacher sees when the environment exposes them."""
    teacher_prompt_style: Literal["plain", "feedback_tag", "reasoning_feedback_tag"] = "reasoning_feedback_tag"
    """Prompt format for teacher feedback."""
    teacher_parse_feedback_tags: bool = True
    """When teacher output contains <feedback> tags, pass only the tagged feedback onward."""


@dataclass
class LLMEngineArgs:
    """Configuration for a vLLM engine instance."""

    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    """The HuggingFace model ID to use"""
    tokenizer_id: Optional[str] = None
    """The tokenizer ID (if different from model_id)"""
    enable_thinking: bool = True
    """Whether to pass enable_thinking to tokenizer.apply_chat_template"""
    max_model_len: int = 8192
    """Maximum context length for the model"""
    tensor_parallel_size: int = 1
    """Number of GPUs to use for tensor parallelism"""
    pipeline_parallel_size: int = 1
    """Number of pipeline stages"""
    enable_prefix_caching: bool = True
    """Whether to enable prefix caching (highly recommended for agents)"""
    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory to allocate to vLLM"""
    dtype: str = "auto"
    """Data type (auto, float16, bfloat16)"""
    enable_lora: bool = False
    """Whether to enable LoRA adapter support"""
    max_lora_rank: int = 64
    """Max LoRA rank if enabling LoRA"""
    adapter_path: Optional[str] = None
    """Optional path to a LoRA adapter directory to load for inference"""
    enforce_eager: bool = False
    """If true, disable vLLM torch.compile integration and CUDA graphs"""
    max_logprobs: int = 128
    """Maximum number of logprobs the vLLM engine will return per token"""
    enable_thinking: bool = True
    """Whether to enable thinking mode formatting via apply_chat_template (e.g. for Gemma 4)"""
    max_num_batched_tokens: Optional[int] = None
    """Optional vLLM cap for tokens used in scheduler profiling/batching"""
    max_num_seqs: Optional[int] = None
    """Optional vLLM cap for concurrent sequences"""

    def get_tokenizer_id(self) -> str:
        return self.tokenizer_id if self.tokenizer_id else self.model_id


@dataclass
class LLMClientArgs:
    """Configuration for remote API clients (OpenAI, etc.)."""

    client_name: str = "openai"
    """Name of the client (openai, anthropic, etc.)"""
    model_id: str = "gpt-4o-mini-2024-07-18"
    """API model ID"""
    tokenizer_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    """Tokenizer ID to use for APIs that do not return token IDs (e.g., OpenAI)"""
    enable_thinking: bool = True
    """Whether to pass enable_thinking to tokenizer.apply_chat_template"""
    base_url: Optional[str] = None
    """API base URL"""
    timeout: int = 60
    """Request timeout in seconds"""
    max_retries: int = 3
    """Max retries for failed requests"""
    delay: int = 2
    """Delay between retries"""
    alternate_roles: bool = False
    """Whether to alternate user/assistant roles (some APIs require this)"""


@dataclass
class SamplingArgs:
    """Configuration for LLM generation/sampling."""

    max_tokens: int = 2048
    """Maximum number of new tokens to generate"""
    temperature: float = 1.0
    """Sampling temperature"""
    top_p: float = 1.0
    """Nucleus sampling probability"""
    include_stop_str_in_output: bool = True
    """Whether to include the stop string in the output text"""
    stop: Optional[list[str]] = None
    """Optional stop strings passed through to the backend"""
    stop_token_ids: Optional[list[int]] = None
    """Optional stop token ids passed through to the backend"""
    ignore_eos: bool = False
    """Whether to ignore EOS and continue generating"""
    min_tokens: int = 0
    """Minimum number of tokens before EOS or stop tokens can terminate generation"""
    logprobs: Optional[int] = 1
    """Whether to return log probabilities"""


@dataclass
class LLMActorArgs:
    """Configuration for creating the LLM actor."""

    actor_type: Literal["client", "vllm", "hf", "dummy"] = "vllm"
    """Type of actor to create ('client', 'vllm', 'hf', or 'dummy')"""
    client_args: LLMClientArgs = field(default_factory=LLMClientArgs)
    """Arguments for the client actor (if actor_type='client')"""
    engine_args: LLMEngineArgs = field(default_factory=LLMEngineArgs)
    """Arguments for the vLLM engine actor (if actor_type='vllm')"""
    dummy_args: Optional[dict] = None
    """Arguments for the dummy actor (if actor_type='dummy')"""
    sampling_args: SamplingArgs = field(default_factory=SamplingArgs)
    """Sampling parameters for generation"""


@dataclass
class Args:
    # General parameters
    num_eval_episodes: int = 256
    """number of evaluation episodes to run"""
    seed: int = 0
    """seed of the experiment"""
    output_dir: str = "runs"
    """directory for output files"""
    push_to_hub: bool = False
    """whether to push the dataset to Hugging Face Hub"""
    hub_repo: Optional[str] = None
    """repository name on Hugging Face Hub (e.g., username/dataset-name)"""
    use_distributed: bool = False
    """whether to use ray for distributed execution"""

    # Dataset configuration
    dataset_path: str = "LLParallax/crafter-trajectories"
    """path to the dataset of episodes for training"""
    embeddings_path: str = "LLParallax/crafter-embeddings-llama1b"
    """path to the dataset of plan embeddings"""
    gamma: float = 0.99
    """discount factor (gamma) for future rewards"""
    horizon: int = 400
    """horizon for the sampling of the future goals"""
    normalize_inputs: bool = False
    """whether to normalize the inputs (images) in the dataset"""
    num_workers: int = 4
    """number of workers to use for data loading"""
    goal_types: tuple[str, ...] = ("goal_plan_embedding",)
    """goal types to contrast with the current state and action (e.g., 'goal_plan', 'goal_plan_embedding', 'goal_future_state')"""
    max_token_length: int = 128
    """maximum length of the tokenized input for the language model"""
    use_game_states: bool = False
    """whether the dataset stores full game-state objects (render them on the fly)"""
    render_size: tuple[int, int] = (64, 64)
    """render size (width, height) to use when rendering saved game states"""

    # Environment configuration
    env_name: str = "crafter"
    """environment to use"""
    nle_args: NLEArgs = field(default_factory=NLEArgs)
    """arguments for the NLE environment"""
    math_args: MathArgs = field(default_factory=MathArgs)
    """arguments for the math environment"""
    countdown_args: CountdownArgs = field(default_factory=CountdownArgs)
    """arguments for the countdown environment"""
    bbeh_args: BBehArgs = field(default_factory=BBehArgs)
    """arguments for the BBEH environment"""
    sciknoweval_args: SciKnowEvalArgs = field(default_factory=SciKnowEvalArgs)
    """arguments for the SciKnowEval environment"""
    arc_agi_args: ArcAgiArgs = field(default_factory=ArcAgiArgs)
    """arguments for the ARC-AGI environment"""
    code_args: CodeArgs = field(default_factory=CodeArgs)
    """arguments for the code environment"""
    enterprise_ops_args: EnterpriseOpsArgs = field(default_factory=EnterpriseOpsArgs)
    """arguments for the EnterpriseOps-Gym environment"""
    task: str = "default"
    """task to perform in the environment"""
    language_wrapper: bool = False
    """whether to use the language wrapper for the environment"""
    record_stats: bool = False
    """whether to use the stats recorder for the environment"""
    record_videos: bool = False
    """whether to use the video recorder for the environment"""
    render_mode: Optional[Literal["human", "rgb_array"]] = None

    # Logging and experiment tracking
    exp_name: str = "run"
    """the name of this experiment"""
    log_wandb: bool = True
    """if true, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "plan-crl"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "ideas-ncbr"
    """the entity (team) of wandb's project"""
    wandb_group: str = "."
    """the wandb's group name"""
    wandb_mode: Literal["online", "offline"] = "online"
    """wandb logging mode: 'online' for real-time sync, 'offline' for local logging"""
    wandb_tags: Optional[list[str]] = None
    """tags for the wandb run"""
    wandb_job_type: str = "train"
    """job type for the wandb run"""
    wandb_unique_id: Optional[str] = None
    """unique ID for the wandb run, if None, it will be generated automatically"""

    # System and performance settings
    checkpoint_logdir: Optional[str] = None
    """directory to save model checkpoints (if None, no checkpoints saved)"""
    checkpoint_every: int = 10000
    """number of steps between checkpoints"""
    device: str = "auto"
    """whether to use CUDA acceleration if available"""
    dtype: str = "auto"
    """data type for the model (e.g., 'float16', 'bfloat16', 'float32')"""

    # Contrastive Training settings
    train_steps: int = 6_000_000
    """number of training steps to perform"""
    eval_every_steps: int = 10000
    """number of steps between evaluations"""
    num_eval_trajectories: int = 10
    """number of trajectories to sample for evaluation"""
    policy_lr: float = 3e-4
    """learning rate for the policy network"""
    critic_lr: float = 3e-4
    """learning rate for the critic network"""
    alpha_lr: float = 3e-4
    """learning rate for the entropy temperature parameter"""
    batch_size: int = 128
    """batch size for training"""
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    """type of contrastive loss function to use"""
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"
    """energy function for computing similarities"""
    logsumexp_penalty_coeff: float = 0.1
    """coefficient for forward CRL logsumexp penalty"""
    alpha: Union[float, str] = "auto"
    """initial value for the entropy temperature parameter, can be 'auto' or a float"""

    # Plan evaluation settings
    plan_eval_every_steps: int = 100_000
    """number of training steps between plan-completion evaluations (set to 0 to disable)"""
    num_plan_eval_episodes: int = 10
    """number of episodes to run when computing plan-completion accuracy"""

    # Network architecture settings
    conv_downsample_blocks: int = 2
    """number of downsample blocks in the conv encoder"""
    hidden_size: int = 128
    """hidden size size for neural networks"""
    num_blocks: int = 2
    """number of blocks in networks"""
    activation: Literal["relu", "swish", "elu", "tanh"] = "swish"
    """activation function to use in networks ('relu', 'swish', 'elu', 'tanh')"""
    accumulation_steps: int = 1
    """number of gradient accumulation steps before optimizer update"""
    repr_size: int = 64
    """size of the representation vector"""

    # LLM agent configuration
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    """the model to use for the agent"""
    tokenizer_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    """the tokenizer to use for the agent"""
    max_history: int = 4
    """maximum number of observations to keep in the history for the agent"""
    max_image_history: int = 0
    """maximum number of images to keep in the history for the agent"""
    num_eval_workers: int = 8
    """number of workers to use for evaluation"""
    llm_agent: Literal["naive", "hierarchical", "hierarchical_separate", "dummy", "manual"] = "naive"
    """type of agent to use ('naive', 'hierarchical', 'hierarchical_separate', 'dummy' or 'manual')"""
    plan_every_k: Union[int, tuple[int, int]] = 10
    """number of steps after which to plan in the 'plan_every_k_step' agent"""
    diverse_plans: bool = False
    """whether to use diverse plans in the 'plan_every_k_step' agent"""
    plan_scheduler: Literal["random_k", "llm_completion"] = "random_k"
    """planning scheduler for 'plan_every_k_step' agent"""
    live_preview_plans: bool = False
    """print new plans to stdout as they are generated"""
    live_preview_file: Optional[str] = None
    """optional path to a JSONL file where new plans will be appended"""
    debug_action_extraction: bool = False
    """enable debug logging for action extraction processing"""
    ask_for_plan_completion: bool = False
    """whether to ask the judge model for plan completion feedback after executing the plan"""
    ask_for_adaptive_replan: bool = False
    """whether to ask the judge model for adaptive replanning feedback after executing the plan"""
    actor_prompt_layout: Literal["system_first", "prompt_last"] = "system_first"
    """layout for actor prompts: standard chat `system+user` or single `user` with prompt suffix"""
    actor_prompt_tail_style: Literal["full", "compact"] = "full"
    """when `actor_prompt_layout=prompt_last`, use full actor prompt text or compact action-only directive"""
    guidance_scale: float = 1.0
    """Classifier-free guidance scale; 1.0 disables guidance"""
    refinement_prompt: RefinementPromptArgs = field(default_factory=RefinementPromptArgs)
    """Generic prompt controls shared by refinement-style environments."""
    math_dpo_pair_strategy: Literal["adjacent", "correctness_adjacent", "solve_vs_history"] = "adjacent"
    """How DPO preference pairs are constructed for math episodes."""
    math_dpo_max_history_negatives: int = 4
    """Maximum number of prior incorrect attempts to use as negatives in solve_vs_history mode."""

    # LLM actor configuration
    llm_actor: LLMActorArgs = field(default_factory=LLMActorArgs)
    """Configuration for the primary ACTOR model (the one being trained)"""
    llm_judge: LLMActorArgs = field(default_factory=LLMActorArgs)
    """Configuration for the JUDGE model (frozen, used for planning/eval)"""

    # Chunk training (critic on chunks)
    use_chunk_training: bool = False
    """whether to train critic on chunks (sequence of state-action pairs) using a transformer"""
    chunk_len: int = 8
    """maximum chunk length (chunks are padded to this length)"""
    chunk_batch_size: int = 16
    """number of chunks per batch when using chunk training"""
    transformer_nhead: int = 4
    """number of heads in transformer encoder"""
    transformer_nlayers: int = 2
    """number of transformer encoder layers"""
    transformer_dim_feedforward: int = 256
    """transformer feedforward dimension"""
