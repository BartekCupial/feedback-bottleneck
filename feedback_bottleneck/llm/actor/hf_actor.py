import asyncio
import logging
from typing import Any, Dict

import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_bottleneck.config.args import LLMEngineArgs
from feedback_bottleneck.llm.actor.llm_actor import BaseVLLMActor


def _torch_dtype(dtype: str):
    if dtype in ("auto", None):
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported HF dtype: {dtype}")


class _ChunkedFP32Linear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, output_chunk_size: int = 128):
        super().__init__()
        self.linear = linear
        self.output_chunk_size = output_chunk_size
        self._logged_shape = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        original_shape = input_.shape[:-1]
        flat_input = input_.reshape(-1, input_.shape[-1]).float().contiguous()
        outputs = []
        weight = self.linear.weight
        bias = self.linear.bias
        if not self._logged_shape:
            print(
                "gemma4_projection_workaround",
                f"input={tuple(input_.shape)}",
                f"flat={tuple(flat_input.shape)}",
                f"weight={tuple(weight.shape)}",
                f"chunk={self.output_chunk_size}",
                flush=True,
            )
            self._logged_shape = True

        for start in range(0, weight.shape[0], self.output_chunk_size):
            end = min(start + self.output_chunk_size, weight.shape[0])
            weight_chunk = weight[start:end].float().contiguous()
            bias_chunk = bias[start:end].float() if bias is not None else None
            outputs.append(F.linear(flat_input.float(), weight_chunk, bias_chunk))

        output = torch.cat(outputs, dim=-1).to(dtype=input_.dtype)
        return output.reshape(*original_shape, output.shape[-1])


def _patch_gemma4_projection_for_gh200(model: torch.nn.Module) -> None:
    patched = 0
    for module in model.modules():
        projection = getattr(module, "per_layer_model_projection", None)
        if isinstance(projection, torch.nn.Linear):
            module.per_layer_model_projection = _ChunkedFP32Linear(projection)
            patched += 1
    if patched:
        logging.info("Patched %d Gemma4 per-layer projection module(s) with chunked FP32 linear", patched)


class HFTransformersActor(BaseVLLMActor):
    """Small local Transformers fallback for models that are not stable in vLLM."""

    def __init__(self, args: LLMEngineArgs):
        logging.info(f"Initializing HFTransformersActor with model {args.model_id}")
        self._validate_adapter_config(args)
        if args.adapter_path is not None:
            raise ValueError("HFTransformersActor does not support adapter_path")

        tokenizer_id = args.get_tokenizer_id()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": _torch_dtype(args.dtype),
            "device_map": "auto",
        }
        self.model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        model_id_lower = args.model_id.lower()
        if "gemma-4" in model_id_lower or "gemma4" in model_id_lower:
            _patch_gemma4_projection_for_gh200(self.model)
        self.model.eval()
        self._init_adapter_state(args)

    async def _get_tokenizer(self):
        return self.tokenizer

    def _generate_sync(self, prompt: str, sampling_params_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        temperature = float(sampling_params_dict.get("temperature", 1.0))
        top_p = float(sampling_params_dict.get("top_p", 1.0))
        max_new_tokens = int(sampling_params_dict.get("max_tokens", 2048))
        min_new_tokens = int(sampling_params_dict.get("min_tokens", 0))
        stop_token_ids = sampling_params_dict.get("stop_token_ids") or []
        stop_strings = sampling_params_dict.get("stop") or []
        ignore_eos = bool(sampling_params_dict.get("ignore_eos", False))
        include_stop = bool(sampling_params_dict.get("include_stop_str_in_output", True))

        eos_token_id = None if ignore_eos else self.tokenizer.eos_token_id
        if stop_token_ids:
            eos_values = [eos_token_id] if eos_token_id is not None else []
            eos_values.extend(stop_token_ids)
            eos_token_id = eos_values

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)

        generated_ids = output_ids[0, input_len:].detach().cpu().tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        for stop in stop_strings:
            stop_idx = text.find(stop)
            if stop_idx >= 0:
                text = text[: stop_idx + len(stop) if include_stop else stop_idx]
                break

        finish_reason = "length"
        if generated_ids and self.tokenizer.eos_token_id in generated_ids:
            finish_reason = "stop"
        if stop_strings and any(stop in text for stop in stop_strings):
            finish_reason = "stop"

        return {
            "text": text,
            "logprobs": None,
            "token_ids": generated_ids,
            "step_logprobs": None,
            "finish_reason": finish_reason,
        }

    async def _generate_from_prompt(self, prompt: Any, sampling_params_dict: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate_sync(prompt, sampling_params_dict))


@ray.remote(num_gpus=1)
class RemoteHFTransformersActor(HFTransformersActor):
    def ready(self):
        return True
