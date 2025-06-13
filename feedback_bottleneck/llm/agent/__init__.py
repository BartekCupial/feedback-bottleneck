from importlib import import_module

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.llm.agent.base import BaseAgent

ENV_AGENT_MODULES = [
    ("crafter", "Crafter", "feedback_bottleneck.llm.agent.crafter"),
    ("nle", "NLE", "feedback_bottleneck.llm.agent.nle"),
    ("countdown", "Countdown", "feedback_bottleneck.llm.agent.countdown"),
    ("math", "Math", "feedback_bottleneck.llm.agent.math"),
    ("bbeh", "BBeh", "feedback_bottleneck.llm.agent.bbeh"),
    ("sciknow", "SciKnowEval", "feedback_bottleneck.llm.agent.sciknoweval"),
    ("enterprise_ops", "EnterpriseOps", "feedback_bottleneck.llm.agent.enterprise_ops"),
    ("arc_agi", "ArcAgi", "feedback_bottleneck.llm.agent.arc_agi"),
    ("code", "Code", "feedback_bottleneck.llm.agent.code"),
]


def _resolve_agent_module(env_name: str) -> tuple[str, str]:
    env_name_lower = env_name.lower()
    for token, class_stem, module_path in ENV_AGENT_MODULES:
        if token in env_name_lower:
            return class_stem, module_path
    raise ValueError(f"Unknown environment for LLM agent dispatch: {env_name}")


def _load_agent_class(env_name: str, class_name: str):
    _, module_path = _resolve_agent_module(env_name)
    module = import_module(module_path)
    return getattr(module, class_name)


def create_formatter(args: Args):
    if args.llm_agent in {"dummy", "manual"}:
        return None

    class_stem, _ = _resolve_agent_module(args.env_name)

    if args.llm_agent == "naive":
        formatter_name = f"Naive{class_stem}Formatter"
    elif args.llm_agent in {"hierarchical", "hierarchical_separate"}:
        formatter_name = f"Hierarchical{class_stem}Formatter"
    else:
        raise ValueError(f"Unknown agent type: {args.llm_agent}")

    formatter_cls = _load_agent_class(args.env_name, formatter_name)
    return formatter_cls(args)


class AgentFactory:
    def __init__(self, args: Args, llm_actor, llm_judge=None):
        self.llm_actor = llm_actor
        self.llm_judge = llm_judge
        self.args = args

    def create_agent(self) -> BaseAgent:
        formatter = create_formatter(self.args)
        class_stem, _ = _resolve_agent_module(self.args.env_name)

        if self.args.llm_agent == "naive":
            agent_cls = _load_agent_class(self.args.env_name, f"Naive{class_stem}Agent")
            return agent_cls(self.llm_actor, formatter, self.args)

        if self.args.llm_agent == "hierarchical":
            agent_cls = _load_agent_class(self.args.env_name, f"Hierarchical{class_stem}Agent")
            return agent_cls(self.llm_actor, formatter, self.args)

        if self.args.llm_agent == "hierarchical_separate":
            agent_cls = _load_agent_class(self.args.env_name, f"HierarchicalSeparate{class_stem}Agent")
            return agent_cls(self.llm_judge, self.llm_actor, formatter, self.args)

        if self.args.llm_agent == "dummy":
            from feedback_bottleneck.llm.agent.dummy import DummyAgent

            return DummyAgent(self.llm_actor, formatter, self.args)

        if self.args.llm_agent == "manual":
            from feedback_bottleneck.llm.agent.manual import ManualAgent

            return ManualAgent(self.llm_actor, formatter, self.args)

        raise ValueError(f"Unknown agent type: {self.args.llm_agent}")
