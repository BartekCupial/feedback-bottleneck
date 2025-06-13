import inspect

import torch


def model_supports_token_type_ids(model) -> bool:
    if hasattr(model, "module"):
        model = model.module

    return "token_type_ids" in inspect.signature(model.forward).parameters


def maybe_add_text_token_type_ids(model_inputs: dict, use_token_type_ids: bool) -> dict:
    if not use_token_type_ids or "token_type_ids" in model_inputs:
        return model_inputs

    return {
        **model_inputs,
        "token_type_ids": torch.zeros_like(model_inputs["input_ids"]),
    }
