import asyncio
from types import SimpleNamespace

import ray
from transformers import AutoTokenizer

from feedback_bottleneck.config.args import LLMClientArgs
from feedback_bottleneck.llm.actor.client import LLMClientWrapper, LLMResponse, create_llm_client
from feedback_bottleneck.llm.actor.llm_actor import render_chat_prompt_token_ids


class ClientActor:
    def __init__(self, args: LLMClientArgs):
        self.client: LLMClientWrapper = create_llm_client(args)()
        self.enable_thinking = args.enable_thinking

        # this is hack for getting comparable tokens out of apis which do not return token_ids
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_id,
            use_fast=False,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    async def chat(self, messages, sampling_params):
        # Convert the API format to the format expected by BALROG client wrappers
        formatted_messages = self._convert_messages(messages)
        prompt_token_ids = render_chat_prompt_token_ids(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )

        loop = asyncio.get_event_loop()
        llm_response: LLMResponse = await loop.run_in_executor(
            None,
            lambda: self.client.generate(formatted_messages, sampling_params),
        )
        logprobs = None
        token_ids = self.tokenizer.encode(llm_response.completion, return_tensors="pt").tolist()[0]

        return {
            "text": llm_response.completion,
            "logprobs": logprobs,
            "token_ids": token_ids,
            "prompt_token_ids": prompt_token_ids,
        }

    def _convert_messages(self, messages):
        formatted_messages = []
        for msg in messages:
            text = "".join(content["text"] for content in msg["content"] if content["type"] == "text")
            formatted_messages.append(SimpleNamespace(role=msg["role"], content=text, attachment=None))
        return formatted_messages


@ray.remote
class RemoteClientActor(ClientActor):
    pass
