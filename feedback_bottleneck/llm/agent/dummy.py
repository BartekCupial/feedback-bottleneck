from feedback_bottleneck.llm.agent.base import BaseAgent


class DummyAgent(BaseAgent):
    def reset(self):
        pass

    async def get_action(self, obs):
        return dict(
            action="Noop",
            logprob=0.0,
            prompt_token_ids=[],
            action_token_ids=[],
        )
