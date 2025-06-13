import abc
import json
import random
import re
import sys  # for flush
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

from feedback_bottleneck.config.args import Args


class BaseAgent(abc.ABC):
    """Base class for agents using prompt-based interactions."""

    def __init__(self, llm_actor, prompt_formatter, args: Args):
        """Initialize the agent with a client and prompt builder."""
        self.llm_actor = llm_actor
        self.prompt_formatter = prompt_formatter
        self.args = args

        sampling_args = args.llm_actor.sampling_args
        self.sampling_params = vars(sampling_args)
        self.guidance_scale = args.guidance_scale
        self.cfg_enabled = args.llm_actor.actor_type == "vllm"
        self.rng = random.Random(args.seed)

    def generate(self, messages, llm=None, sampling_params=None):
        """Generate a response based on the current prompt."""
        if llm is None:
            llm = self.llm_actor

        if sampling_params is None:
            sampling_params = self.sampling_params

        if self.args.use_distributed:
            return llm.chat.remote(messages, sampling_params)
        else:
            return llm.chat(messages, sampling_params)

    def reset(self):
        self.prompt_formatter.reset()

    async def get_action(self, messages):
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def generate_with_cfg(self, conditional_messages, unconditional_messages, llm=None, sampling_params=None):
        if llm is None:
            llm = self.llm_actor

        if sampling_params is None:
            sampling_params = self.sampling_params

        if self.guidance_scale == 1.0 or not self.cfg_enabled:
            return await self.generate(conditional_messages, llm=llm, sampling_params=sampling_params)

        cfg_seed = self.rng.randrange(2**63 - 1)
        if self.args.use_distributed:
            return await llm.chat_with_cfg.remote(
                conditional_messages,
                unconditional_messages,
                sampling_params,
                self.guidance_scale,
                cfg_seed,
            )
        return await llm.chat_with_cfg(
            conditional_messages,
            unconditional_messages,
            sampling_params,
            self.guidance_scale,
            cfg_seed,
        )

    def parse_tag_content(self, text: str, tags: List[str]) -> List[str]:
        """Consistency: Generic parser for any tag list."""
        results = []
        for tag in tags:
            match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            results.append(match.group(1).strip() if match else "")
        return results


class BaseFormatter(abc.ABC):
    def __init__(self, args: Args):
        self.args = args
        self.max_history = args.max_history
        self.reset()

    def reset(self):
        self.obs_history = deque(maxlen=self.max_history)
        self.act_history = deque(maxlen=self.max_history)

    def append_observation(self, obs: Dict[str, Any]):
        """Adds a new observation to the context window."""
        self.obs_history.append(obs)

    def append_action(self, action: str):
        """Adds a new action to the context window."""
        self.act_history.append(action)

    @abc.abstractmethod
    def get_prompt(self, role: str, **kwargs) -> List[Dict[str, str]]:
        """
        Constructs the messages list (system, user, etc.) for the LLM.

        Args:
            role: The specific agent component requesting the prompt (e.g., 'actor', 'judge', 'planner').
            **kwargs: Additional context if needed.

        Returns:
            A list of message dictionaries compatible with the LLM API.
        """
        pass

    @abc.abstractmethod
    def generate_sft_samples(self, episode):
        """
        Generates supervised fine-tuning samples from a given episode.

        Args:
            episode: A list of dictionaries representing the episode data.
        """
        pass
