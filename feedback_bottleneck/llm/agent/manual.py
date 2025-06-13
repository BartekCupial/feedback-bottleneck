from feedback_bottleneck.llm.agent.base import BaseAgent


class ManualAgent(BaseAgent):
    def reset(self):
        pass

    async def get_action(self, obs):
        action = input("> ")
        return dict(
            action=action,
            logprob=0.0,
            prompt_token_ids=[],
            action_token_ids=[],
        )
