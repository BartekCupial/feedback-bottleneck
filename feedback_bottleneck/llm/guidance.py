import random
from typing import Dict, List

import torch


def apply_top_p(scores: Dict[int, float], top_p: float) -> Dict[int, float]:
    if top_p >= 1.0:
        return scores

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    logits = torch.tensor([score for _, score in ranked], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)
    cumulative = 0.0
    kept: Dict[int, float] = {}
    for (token_id, score), prob in zip(ranked, probs.tolist(), strict=False):
        kept[token_id] = score
        cumulative += prob
        if cumulative >= top_p:
            break
    return kept


def sample_from_scores(
    scores: Dict[int, float],
    temperature: float,
    rng: random.Random,
) -> int:
    if not scores:
        raise ValueError("Cannot sample from an empty score set")

    if temperature <= 0:
        return max(scores.items(), key=lambda item: item[1])[0]

    items = list(scores.items())
    token_ids = [token_id for token_id, _ in items]
    logits = torch.tensor([score / temperature for _, score in items], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)

    # Use the seeded Python RNG to seed torch sampling deterministically per draw.
    torch_gen = torch.Generator()
    torch_gen.manual_seed(rng.randrange(2**63 - 1))
    sampled_idx = torch.multinomial(probs, num_samples=1, generator=torch_gen).item()
    return token_ids[sampled_idx]


def combine_candidates_with_cfg(
    conditional: List[Dict],
    unconditional: List[Dict],
    guidance_scale: float,
) -> Dict[int, float]:
    cond_scores = {candidate["token_id"]: candidate["logprob"] for candidate in conditional}
    uncond_scores = {candidate["token_id"]: candidate["logprob"] for candidate in unconditional}

    floor = (
        min(
            list(cond_scores.values()) + list(uncond_scores.values()),
            default=-100.0,
        )
        - 20.0
    )

    guided_scores: Dict[int, float] = {}
    for token_id in cond_scores.keys() | uncond_scores.keys():
        cond = cond_scores.get(token_id, floor)
        uncond = uncond_scores.get(token_id, floor)
        guided_scores[token_id] = uncond + guidance_scale * (cond - uncond)

    return guided_scores


def merge_candidate_text(conditional, unconditional):
    return {
        **{candidate["token_id"]: candidate["text"] for candidate in conditional},
        **{candidate["token_id"]: candidate["text"] for candidate in unconditional},
    }
