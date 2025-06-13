from typing import Literal, Union

import ray

from feedback_bottleneck.config.args import Args, LLMActorArgs


def create_llm_actor_helper(
    args: LLMActorArgs,
    use_distributed: bool = False,
):
    """
    Factory function to create a Ray actor for LLM inference.
    This allows for distributed execution of LLM tasks.
    """
    if args.actor_type == "client":
        from feedback_bottleneck.llm.actor.client_actor import ClientActor, RemoteClientActor

        if use_distributed:
            return RemoteClientActor.remote(args.client_args)
        else:
            return ClientActor(args.client_args)
    elif args.actor_type == "vllm":
        from feedback_bottleneck.llm.actor.llm_actor import LLMActor, RemoteLLMActor

        if use_distributed:
            num_gpus = args.engine_args.tensor_parallel_size * args.engine_args.pipeline_parallel_size
            actor = RemoteLLMActor.options(num_gpus=num_gpus).remote(args.engine_args)
            ray.get(actor.ready.remote())
            return actor
        else:
            return LLMActor(args.engine_args)
    elif args.actor_type == "hf":
        from feedback_bottleneck.llm.actor.hf_actor import HFTransformersActor, RemoteHFTransformersActor

        if use_distributed:
            actor = RemoteHFTransformersActor.remote(args.engine_args)
            ray.get(actor.ready.remote())
            return actor
        else:
            return HFTransformersActor(args.engine_args)
    elif args.actor_type == "dummy":
        from feedback_bottleneck.llm.actor.dummy_actor import DummyActor, RemoteDummyActor

        if use_distributed:
            return RemoteDummyActor.remote(args.dummy_args)
        else:
            return DummyActor(args.dummy_args)
    else:
        raise ValueError(f"Unknown actor type: {args.actor_type}. Supported types are 'client', 'vllm', 'hf', and 'dummy'.")


def create_llm_actor(args: Args):
    return create_llm_actor_helper(args.llm_actor, use_distributed=args.use_distributed)


def create_llm_judge(args: Args):
    return create_llm_actor_helper(args.llm_judge, use_distributed=args.use_distributed)


def create_llm_actors(args: Args):
    llm_actor = create_llm_actor(args)

    llm_judge = None
    if args.llm_agent == "hierarchical_separate":
        llm_judge = create_llm_judge(args)

    return llm_actor, llm_judge
