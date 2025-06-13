# AsyncLLMEngine only works with async ray actors https://github.com/vllm-project/vllm/issues/7904
import asyncio
import inspect
import logging
import uuid
from typing import Any, Dict, List

import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from feedback_bottleneck.config.args import LLMEngineArgs

PROMPT_TRUNCATION_MARGIN_TOKENS = 16
MIN_PROMPT_BUDGET_TOKENS = 256


def flatten_messages(messages):
    prompt_msgs = []
    for m in messages:
        # Flatten OpenAI content types into plain string
        content = ""
        for c in m["content"]:
            # Only support text for now; extend if you want image input etc.
            if c["type"] == "text":
                content += c.get("text", "")
        prompt_msgs.append({"role": m["role"], "content": content})
    return prompt_msgs


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool = True) -> str:
    prompt_msgs = flatten_messages(messages)
    if prompt_msgs and prompt_msgs[-1]["role"] == "assistant":
        return tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=enable_thinking,
        )
    return tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def render_chat_prompt_token_ids(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool = True) -> List[int]:
    prompt_msgs = flatten_messages(messages)
    if prompt_msgs and prompt_msgs[-1]["role"] == "assistant":
        tokenized_prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=enable_thinking,
        )
        return _normalize_prompt_token_ids(tokenized_prompt)
    tokenized_prompt = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return _normalize_prompt_token_ids(tokenized_prompt)


def _normalize_prompt_token_ids(tokenized_prompt: Any) -> List[int]:
    if isinstance(tokenized_prompt, dict) or (
        hasattr(tokenized_prompt, "keys") and "input_ids" in tokenized_prompt
    ):
        tokenized_prompt = tokenized_prompt["input_ids"]
    if hasattr(tokenized_prompt, "tolist"):
        tokenized_prompt = tokenized_prompt.tolist()
    if tokenized_prompt and isinstance(tokenized_prompt[0], list):
        tokenized_prompt = tokenized_prompt[0]
    return [int(token_id) for token_id in tokenized_prompt]


def _add_optional_vllm_limits(llm_kwargs: Dict[str, Any], args: LLMEngineArgs) -> None:
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs


class BaseVLLMActor:
    def _validate_adapter_config(self, args: LLMEngineArgs):
        if args.adapter_path is not None and not args.enable_lora:
            raise ValueError("Received engine_args.adapter_path but engine_args.enable_lora is False")

    def _init_adapter_state(self, args: LLMEngineArgs):
        self.active_adapter_path = args.adapter_path
        self.adapter_version = 1 if args.adapter_path is not None else 0
        self.max_model_len = args.max_model_len

    def _build_lora_request(self):
        if self.active_adapter_path is None:
            return None
        return LoRARequest(
            f"sft_adapter_v{self.adapter_version}",
            self.adapter_version,
            self.active_adapter_path,
        )

    async def _get_tokenizer(self):
        return self.tokenizer

    def _response_from_result(self, result) -> Dict[str, Any]:
        return {
            "text": result.text,
            "logprobs": result.cumulative_logprob,
            "token_ids": result.token_ids,
            "step_logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }

    def _error_response(self) -> Dict[str, Any]:
        return {
            "text": "",
            "logprobs": None,
            "token_ids": [],
            "step_logprobs": None,
            "finish_reason": "error",
        }

    async def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = await self._get_tokenizer()
        prompt = render_chat_prompt(tokenizer, messages, enable_thinking=self.enable_thinking)
        prompt_token_ids = render_chat_prompt_token_ids(tokenizer, messages, enable_thinking=self.enable_thinking)
        prompt, prompt_token_ids = self._truncate_prompt_to_budget(
            tokenizer,
            prompt,
            prompt_token_ids,
            sampling_params,
        )
        # Pass exact token ids to vLLM after truncation. Decoding truncated ids back
        # to text and letting vLLM tokenize again can drift by a token and exceed
        # the model context limit.
        response = await self._generate_from_prompt(prompt_token_ids, sampling_params)
        response["prompt_token_ids"] = prompt_token_ids
        return response

    def _truncate_prompt_to_budget(
        self,
        tokenizer,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> tuple[str, List[int]]:
        max_model_len = int(getattr(self, "max_model_len", 0) or 0)
        if max_model_len <= 0:
            return prompt, prompt_token_ids

        max_new_tokens = int(sampling_params.get("max_tokens") or 0)
        prompt_budget = max_model_len - max_new_tokens - PROMPT_TRUNCATION_MARGIN_TOKENS
        if prompt_budget < MIN_PROMPT_BUDGET_TOKENS:
            prompt_budget = max(1, max_model_len - PROMPT_TRUNCATION_MARGIN_TOKENS)

        if len(prompt_token_ids) <= prompt_budget:
            return prompt, prompt_token_ids

        truncated_prompt_token_ids = prompt_token_ids[-prompt_budget:]
        logging.warning(
            "Truncating overlong prompt from %d to %d tokens "
            "(max_model_len=%d, max_new_tokens=%d)",
            len(prompt_token_ids),
            len(truncated_prompt_token_ids),
            max_model_len,
            max_new_tokens,
        )
        truncated_prompt = tokenizer.decode(truncated_prompt_token_ids, skip_special_tokens=False)
        return truncated_prompt, truncated_prompt_token_ids

    async def _generate_from_prompt(self, prompt: Any, sampling_params_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _generate_from_prompt().")


@ray.remote(num_gpus=1)
class RemoteLLMActor(BaseVLLMActor):
    """
    A dedicated Ray actor that encapsulates the vLLM AsyncLLMEngine.
    It handles inference requests from multiple distributed worker actors.
    """

    def __init__(self, args: LLMEngineArgs):
        logging.info(f"Initializing RemoteLLMActor with model {args.model_id} on GPU {ray.get_gpu_ids()}")
        self._validate_adapter_config(args)
        async_engine_kwargs = {
            "model": args.model_id,
            "tokenizer": args.tokenizer_id,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "enable_prefix_caching": args.enable_prefix_caching,
            "dtype": args.dtype,
            "enable_lora": args.enable_lora,
            "max_lora_rank": args.max_lora_rank,
            "enforce_eager": args.enforce_eager,
            "max_logprobs": args.max_logprobs,
        }
        _add_optional_vllm_limits(async_engine_kwargs, args)
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**async_engine_kwargs)
        )
        # We will get the tokenizer inside the async methods
        self.tokenizer = None
        self.enable_thinking = args.enable_thinking

        self._init_adapter_state(args)

    def ready(self):
        """Simple ping to verify actor is initialized."""
        return True

    def update_adapter(self, adapter_path: str) -> bool:
        """
        Updates the active LoRA adapter.
        vLLM loads this on-the-fly during the next request.
        """
        logging.info(f"Switching vLLM to adapter: {adapter_path}")
        self.active_adapter_path = adapter_path
        self.adapter_version += 1
        return True

    async def _get_tokenizer(self):
        if self.tokenizer is None:
            if hasattr(self.engine, "get_tokenizer"):
                tokenizer = self.engine.get_tokenizer()
                if inspect.isawaitable(tokenizer):
                    tokenizer = await tokenizer
                self.tokenizer = tokenizer
            elif hasattr(self.engine, "engine"):
                self.tokenizer = self.engine.engine.tokenizer.tokenizer
            else:
                self.tokenizer = self.engine.tokenizer
        return self.tokenizer

    async def _generate_from_prompt(self, prompt: Any, sampling_params_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self._get_tokenizer()
            sampling_params = SamplingParams(**sampling_params_dict)
            request_id = str(uuid.uuid4())
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=self._build_lora_request(),
            )

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            return self._response_from_result(final_output.outputs[0])
        except Exception:
            logging.exception("Error processing request")
            return self._error_response()


class LLMActor(BaseVLLMActor):
    """
    A class that encapsulates a HuggingFace model and tokenizer for chat-based inference.
    """

    def __init__(self, args: LLMEngineArgs):
        logging.info(f"Initializing LLMActor with model {args.model_id} on GPU {torch.cuda.current_device()}")
        self._validate_adapter_config(args)
        llm_kwargs = {
            "model": args.model_id,
            "tokenizer": args.tokenizer_id,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "enable_prefix_caching": args.enable_prefix_caching,
            "dtype": args.dtype,
            "enable_lora": args.enable_lora,
            "max_lora_rank": args.max_lora_rank,
            "enforce_eager": args.enforce_eager,
            "max_logprobs": args.max_logprobs,
        }
        _add_optional_vllm_limits(llm_kwargs, args)
        self.engine = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.enable_thinking = args.enable_thinking
        self._init_adapter_state(args)

    async def _generate_from_prompt(self, prompt: Any, sampling_params_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sampling_params = SamplingParams(**sampling_params_dict)
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self.engine.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=self._build_lora_request(),
                ),
            )
            return self._response_from_result(outputs[0].outputs[0])
        except Exception:
            logging.exception("Error processing request")
            return self._error_response()
