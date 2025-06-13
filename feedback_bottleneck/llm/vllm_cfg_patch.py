import asyncio
import random
import uuid
from typing import Any, Dict, List, Optional

from vllm import SamplingParams

from feedback_bottleneck.llm.guidance import apply_top_p, combine_candidates_with_cfg, merge_candidate_text, sample_from_scores


def _extract_candidates(result: Any) -> List[Dict[str, Any]]:
    return [
        {
            "token_id": token_id,
            "text": logprob.decoded_token,
            "logprob": float(logprob.logprob),
        }
        for token_id, logprob in result.logprobs[0].items()
    ]


def _check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: List[str],
    include_in_output: bool,
) -> Optional[tuple[str, int]]:
    if not new_char_count or not stop:
        return None

    for stop_str in stop:
        stop_string_len = len(stop_str)
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        if include_in_output:
            stop_index += stop_string_len
            if stop_index >= len(output_text):
                return stop_str, -1

        return stop_str, stop_index
    return None


async def _generate_cfg_core(
    *,
    conditional_prompt_token_ids: List[int],
    sampling_params: SamplingParams,
    guidance_scale: float,
    eos_token_id: Optional[int],
    seed: int,
    step_fn,
) -> Dict[str, Any]:
    generated_text = ""
    generated_token_ids: List[int] = []
    generated_logprob = 0.0
    finish_reason = "length"
    stop_reason: Optional[str | int] = None

    step_sampling_params = sampling_params.clone()
    step_sampling_params.max_tokens = 1
    max_tokens = sampling_params.max_tokens
    temperature = sampling_params.temperature
    top_p = sampling_params.top_p
    rng = random.Random(seed)
    stop_strings = sampling_params.stop
    include_stop_str_in_output = sampling_params.include_stop_str_in_output
    ignore_eos = sampling_params.ignore_eos
    min_tokens = sampling_params.min_tokens
    active_stop_token_ids = sampling_params.stop_token_ids

    for _ in range(max_tokens):
        conditional_result, unconditional_result = await step_fn(generated_token_ids, step_sampling_params)
        conditional_output = conditional_result.outputs[0]
        unconditional_output = unconditional_result.outputs[0]

        conditional_candidates = _extract_candidates(conditional_output)
        unconditional_candidates = _extract_candidates(unconditional_output)
        guided_scores = combine_candidates_with_cfg(
            conditional=conditional_candidates,
            unconditional=unconditional_candidates,
            guidance_scale=guidance_scale,
        )
        guided_scores = apply_top_p(guided_scores, top_p)

        token_text = merge_candidate_text(
            conditional=conditional_candidates,
            unconditional=unconditional_candidates,
        )
        chosen_token_id = sample_from_scores(guided_scores, temperature, rng)
        chosen_piece = token_text[chosen_token_id]
        generated_token_ids.append(chosen_token_id)
        generated_logprob += guided_scores[chosen_token_id]
        can_stop = len(generated_token_ids) > min_tokens

        token_caused_stop = False
        if can_stop and not ignore_eos and eos_token_id is not None and chosen_token_id == eos_token_id:
            finish_reason = "stop"
            token_caused_stop = True
        elif can_stop and chosen_token_id in active_stop_token_ids:
            finish_reason = "stop"
            stop_reason = chosen_token_id
            token_caused_stop = True

        if not token_caused_stop or include_stop_str_in_output:
            generated_text += chosen_piece

        if not token_caused_stop and can_stop and stop_strings:
            stop_match = _check_stop_strings(
                output_text=generated_text,
                new_char_count=len(chosen_piece),
                stop=stop_strings,
                include_in_output=include_stop_str_in_output,
            )
            if stop_match is not None:
                finish_reason = "stop"
                stop_reason, truncate_to = stop_match
                if truncate_to != -1:
                    generated_text = generated_text[:truncate_to]
                token_caused_stop = True

        if token_caused_stop:
            break

        print(chosen_piece, end="", flush=True)

    print()

    return {
        "text": generated_text,
        "token_ids": generated_token_ids,
        "logprobs": generated_logprob,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "step_logprobs": [],
        "guided": True,
        "prompt_token_ids": list(conditional_prompt_token_ids),
    }


async def _async_generate_cfg(
    engine: Any,
    *,
    conditional_prompt_token_ids: List[int],
    unconditional_prompt_token_ids: List[int],
    sampling_params: Any,
    guidance_scale: float,
    eos_token_id: Optional[int],
    seed: int,
    lora_request: Any = None,
) -> Dict[str, Any]:
    async def step_fn(generated_token_ids: List[int], step_sampling_params: Any):
        async def run_one(prompt_token_ids: List[int]) -> Any:
            request_id = str(uuid.uuid4())
            result_generator = engine.generate(
                {"prompt_token_ids": prompt_token_ids},
                step_sampling_params,
                request_id,
                lora_request=lora_request,
            )
            final_output = None
            async for request_output in result_generator:
                final_output = request_output
            return final_output

        conditional_ids = list(conditional_prompt_token_ids) + list(generated_token_ids)
        unconditional_ids = list(unconditional_prompt_token_ids) + list(generated_token_ids)
        return await asyncio.gather(run_one(conditional_ids), run_one(unconditional_ids))

    return await _generate_cfg_core(
        conditional_prompt_token_ids=conditional_prompt_token_ids,
        sampling_params=sampling_params,
        guidance_scale=guidance_scale,
        eos_token_id=eos_token_id,
        seed=seed,
        step_fn=step_fn,
    )


def _sync_generate_cfg(
    llm: Any,
    *,
    conditional_prompt_token_ids: List[int],
    unconditional_prompt_token_ids: List[int],
    sampling_params: Any,
    guidance_scale: float,
    eos_token_id: Optional[int],
    seed: int,
    use_tqdm: bool = False,
    lora_request: Any = None,
) -> Dict[str, Any]:
    async def step_fn(generated_token_ids: List[int], step_sampling_params: Any):
        prompts = [
            {"prompt_token_ids": list(conditional_prompt_token_ids) + list(generated_token_ids)},
            {"prompt_token_ids": list(unconditional_prompt_token_ids) + list(generated_token_ids)},
        ]
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=step_sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )
        return outputs[0], outputs[1]

    return asyncio.run(
        _generate_cfg_core(
            conditional_prompt_token_ids=conditional_prompt_token_ids,
            sampling_params=sampling_params,
            guidance_scale=guidance_scale,
            eos_token_id=eos_token_id,
            seed=seed,
            step_fn=step_fn,
        )
    )


def ensure_vllm_cfg_patched() -> bool:
    try:
        from vllm import LLM, AsyncLLMEngine
    except ImportError:
        return False

    if getattr(LLM, "_feedback_bottleneck_cfg_patched", False):
        return True

    def llm_generate_cfg(
        self,
        *,
        conditional_prompt_token_ids: List[int],
        unconditional_prompt_token_ids: List[int],
        sampling_params: Any,
        guidance_scale: float,
        eos_token_id: Optional[int],
        seed: int,
        use_tqdm: bool = False,
        lora_request: Any = None,
    ) -> Dict[str, Any]:
        return _sync_generate_cfg(
            self,
            conditional_prompt_token_ids=conditional_prompt_token_ids,
            unconditional_prompt_token_ids=unconditional_prompt_token_ids,
            sampling_params=sampling_params,
            guidance_scale=guidance_scale,
            eos_token_id=eos_token_id,
            seed=seed,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    async def async_engine_generate_cfg(
        self,
        *,
        conditional_prompt_token_ids: List[int],
        unconditional_prompt_token_ids: List[int],
        sampling_params: Any,
        guidance_scale: float,
        eos_token_id: Optional[int],
        seed: int,
        lora_request: Any = None,
    ) -> Dict[str, Any]:
        return await _async_generate_cfg(
            self,
            conditional_prompt_token_ids=conditional_prompt_token_ids,
            unconditional_prompt_token_ids=unconditional_prompt_token_ids,
            sampling_params=sampling_params,
            guidance_scale=guidance_scale,
            eos_token_id=eos_token_id,
            seed=seed,
            lora_request=lora_request,
        )

    LLM.generate_cfg = llm_generate_cfg
    LLM._feedback_bottleneck_cfg_patched = True
    AsyncLLMEngine.generate_cfg = async_engine_generate_cfg
    AsyncLLMEngine._feedback_bottleneck_cfg_patched = True
    return True
