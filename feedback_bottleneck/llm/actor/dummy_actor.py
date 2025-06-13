import asyncio
from typing import Any, Dict, List

import ray


class DummyActor:
    """
    A dummy Ray actor that simulates a model for rapid health checks, dry runs, or testing pipelines.
    """

    def __init__(self, args):
        # Accept all arguments for API compatibility (ignores them).
        pass

    async def _process_request(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate minor async delay
        return {
            "text": "This is a dummy response.",
            "logprobs": 0.0,
            "token_ids": [0, 1, 2],
            "prompt_token_ids": [42],
        }

    async def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._process_request(messages, sampling_params)


@ray.remote
class RemoteDummyActor(DummyActor):
    pass
