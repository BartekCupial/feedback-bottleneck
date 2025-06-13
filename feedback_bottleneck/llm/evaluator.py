import asyncio
import copy
import io
import json
import pickle
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs import make_env
from feedback_bottleneck.llm.agent import AgentFactory
from feedback_bottleneck.utils.utils import NumpyEncoder


def set_seed(env_name, seed, idx):
    if env_name in {"math", "bbeh", "code", "sciknoweval", "enterprise_ops"}:
        return seed + idx
    else:
        return seed * 9999 + idx


class EnvironmentRunner:
    def __init__(self, agent_factory: AgentFactory, args: Args):
        self.args = args
        self.env = None
        self.agent = agent_factory.create_agent()
        self.agent_factory = agent_factory

        self._directory = Path(args.output_dir).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._trace_dir = self._directory / "live_traces"
        self._trace_dir.mkdir(exist_ok=True)
        self._file = (self._directory / "stats.jsonl").open("a")
        self.current_trace_file = None

    def reset(self, seed=None):
        if self.current_trace_file:
            self.current_trace_file.close()
            self.current_trace_file = None

        if self.env is None:
            verifier_llm = self.agent_factory.llm_judge or self.agent_factory.llm_actor
            self.env = make_env(
                self.args.env_name,
                self.args.task,
                self.args,
                render_mode=self.args.render_mode,
                llm_judge=verifier_llm,
            )

        obs, info = self.env.reset(seed=seed)
        self.agent.reset()

        self.current_episode_id = str(uuid.uuid4())
        self.current_trace_path = self._trace_dir / f"run_{self.current_episode_id}.jsonl"
        self.current_trace_file = self.current_trace_path.open("w")

        self._last_obs = obs

        self.episode_return = 0
        self.episode_len = 0
        self.plan_lengths = []
        self.invalid_action_count = 0
        self.plan_completion_events = []
        self.adaptive_replan_events = []
        self.input_token_lengths = []
        self.output_token_lengths = []

        return dict(obs=obs, info=info, episode_id=self.current_episode_id)

    async def get_action(self):
        action_return = await self.agent.get_action(self._last_obs)

        text_action = action_return["action"]
        valid_text_action, feedback = self.env.get_wrapper_attr("check_action_validity")(text_action)
        action = self.env.get_wrapper_attr("language_action_space").map(valid_text_action)

        return dict(
            action=action,
            text_action=text_action,
            valid_text_action=valid_text_action,
            feedback=feedback,
            prompt_token_ids=action_return["prompt_token_ids"],
            action_token_ids=action_return["action_token_ids"],
            logprob=action_return["logprob"],
            plan=action_return.get("plan", ""),
            plan_completed=action_return.get("plan_completed", None),
            adaptive_replan=action_return.get("adaptive_replan", None),
            raw_output=action_return.get("raw_output", ""),
            judge_feedback=action_return.get("judge_feedback", ""),
        )

    async def step(self):
        action_return = await self.get_action()
        input_token_length = len(action_return["prompt_token_ids"])
        output_token_length = len(action_return["action_token_ids"])
        self.input_token_lengths.append(input_token_length)
        self.output_token_lengths.append(output_token_length)

        if action_return["plan"]:
            self.plan_lengths.append(len(action_return["plan"]))

        if action_return["valid_text_action"] != action_return["text_action"]:
            self.invalid_action_count += 1

        if action_return.get("plan_completed") is not None:
            self.plan_completion_events.append(action_return["plan_completed"])

        if action_return.get("adaptive_replan") is not None:
            self.adaptive_replan_events.append(action_return["adaptive_replan"])

        obs, reward, term, trun, info = self.env.step(action_return["valid_text_action"])

        post_step_hook = getattr(getattr(self.env, "unwrapped", self.env), "apply_post_step_verification", None)
        if post_step_hook is not None:
            obs, reward, term, trun, info = await self.env.get_wrapper_attr("apply_post_step_verification")(
                action=action_return["valid_text_action"],
                obs=obs,
                reward=reward,
                terminated=term,
                truncated=trun,
                info=info,
            )

        trace_entry = {
            "step": self.episode_len,
            "action": action_return["action"],
            "text_action": action_return["text_action"],
            "plan": action_return["plan"],
            "judge_feedback": action_return.get("judge_feedback", ""),
            "raw_output": action_return["raw_output"],
            "reward": reward,
            "term": term,
            "trun": trun,
            "end_status": info.get("end_status"),
            "short_term_context": self._last_obs.get("text", {}).get("short_term_context", ""),
            "long_term_context": self._last_obs.get("text", {}).get("long_term_context", ""),
            "logprob": action_return["logprob"],
            "input_token_length": input_token_length,
            "output_token_length": output_token_length,
            "plan_completed": action_return.get("plan_completed", None),
            "adaptive_replan": action_return.get("adaptive_replan", None),
        }
        self.current_trace_file.write(json.dumps(trace_entry, cls=NumpyEncoder) + "\n")
        self.current_trace_file.flush()

        self._last_obs = obs
        self.episode_return += reward
        self.episode_len += 1

        if term or trun:
            episode_extra_stats = info.get("episode_extra_stats", {})

            episode_extra_stats["reward"] = self.episode_return
            episode_extra_stats["len"] = self.episode_len

            episode_extra_stats["planning_frequency"] = len(self.plan_lengths) / self.episode_len
            episode_extra_stats["plan_length"] = (
                sum(self.plan_lengths) / len(self.plan_lengths) if self.plan_lengths else 0
            )
            episode_extra_stats["plan_length_std"] = np.std(self.plan_lengths) if self.plan_lengths else 0
            episode_extra_stats["total_plans"] = len(self.plan_lengths)

            if self.plan_completion_events:
                episode_extra_stats["plan_completion_rate"] = sum(self.plan_completion_events) / len(
                    self.plan_completion_events
                )
            else:
                episode_extra_stats["plan_completion_rate"] = 0.0

            if self.adaptive_replan_events:
                episode_extra_stats["adaptive_replan_rate"] = sum(self.adaptive_replan_events) / len(
                    self.adaptive_replan_events
                )
            else:
                episode_extra_stats["adaptive_replan_rate"] = 0.0

            episode_extra_stats["total_plan_updates"] = len(self.plan_completion_events)

            episode_extra_stats["invalid_action_frequency"] = self.invalid_action_count / self.episode_len

            episode_extra_stats["input_tokens"] = int(np.sum(self.input_token_lengths))
            episode_extra_stats["output_tokens"] = int(np.sum(self.output_token_lengths))

            episode_extra_stats.update(self.env.get_wrapper_attr("get_stats")())
            episode_extra_stats["episode_id"] = self.current_episode_id

            info["episode_extra_stats"] = episode_extra_stats

            if self.args.record_stats:
                self._save(info["episode_extra_stats"])

        return dict(
            obs=obs,
            reward=reward,
            term=term,
            trun=trun,
            info=info,
            **action_return,
        )

    def _save(self, stats):
        self._file.write(json.dumps(stats, cls=NumpyEncoder) + "\n")
        self._file.flush()

    def get_binary_state(self) -> bytes:
        # dump the full env state to a temp file and return its raw bytes
        data = b""
        with tempfile.NamedTemporaryFile() as tmp_dir:
            tmp_path = f"{tmp_dir}/state.joblib"

            try:
                base_env = getattr(self.env, "unwrapped", self.env)
                if hasattr(base_env, "save_state"):
                    base_env.save_state(tmp_path)

                    with open(tmp_path, "rb") as f:
                        data = f.read()
            except Exception:
                pass

        return data

    def close(self):
        if self.env:
            self.env.close()
        if self._file:
            self._file.close()
        if self.current_trace_file:
            self.current_trace_file.close()


class EpisodeCollector:
    def __init__(self, agent_factory, args):
        self.args = args
        self.runner = EnvironmentRunner(agent_factory=agent_factory, args=args)

    async def evaluate_episode(self, seed=None):
        step_return = self.runner.reset(seed=seed)

        while True:
            step_return = await self.runner.step()

            if step_return["term"] or step_return["trun"]:
                break

        episode_log = step_return["info"].get("episode_extra_stats", dict())

        return episode_log

    async def collect_episode(self, seed=None):
        step_return = self.runner.reset(seed=seed)

        # Episode-level ID and output dir
        episode_id = step_return["episode_id"]

        observations = []
        image_paths = []
        state_dumps = []
        actions = []
        text_actions = []
        raw_text_actions = []
        judge_feedbacks = []
        plans = []
        plan_completed = []
        adaptive_replan = []
        raw_outputs = []
        input_token_lengths = []
        output_token_lengths = []
        rewards = []
        terms = []
        truncs = []

        step = 0
        while True:
            observations.append(copy.deepcopy(step_return["obs"]))
            state_dumps.append(self.runner.get_binary_state())

            image = step_return["obs"].get("image", None)
            if image is not None:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                image_paths.append(img_byte_arr.getvalue())
            else:
                image_paths.append(None)

            step_return = await self.runner.step()

            text_actions.append(step_return["valid_text_action"])
            raw_text_actions.append(step_return["text_action"])
            judge_feedbacks.append(step_return.get("judge_feedback", ""))
            actions.append(step_return["action"])
            plans.append(step_return["plan"])
            plan_completed.append(step_return["plan_completed"])
            adaptive_replan.append(step_return["adaptive_replan"])
            raw_outputs.append(step_return["raw_output"])
            input_token_lengths.append(len(step_return["prompt_token_ids"]))
            output_token_lengths.append(len(step_return["action_token_ids"]))
            rewards.append(step_return["reward"])
            terms.append(step_return["term"])
            truncs.append(step_return["trun"])

            if step_return["term"] or step_return["trun"]:
                break

            step += 1

        episode_log = step_return["info"].get("episode_extra_stats", dict())

        raw_inner_obs = [o["obs"] for o in observations]
        obs_keys = raw_inner_obs[0].keys() if raw_inner_obs else []
        flattened_obs = {f"obs_{k}": [step[k] for step in raw_inner_obs] for k in obs_keys}

        episode_df = pd.DataFrame(
            {
                "episode_id": episode_id,
                "timestep": list(range(step + 1)),
                "actions": actions,
                "text_actions": text_actions,
                "raw_text_actions": raw_text_actions,
                "judge_feedbacks": judge_feedbacks,
                "plans": plans,
                "plan_completed": plan_completed,
                "adaptive_replan": adaptive_replan,
                "raw_outputs": raw_outputs,
                "input_token_length": input_token_lengths,
                "output_token_length": output_token_lengths,
                "rewards": rewards,
                "terms": terms,
                "truncs": truncs,
                "image_paths": image_paths,
                "state_dumps": state_dumps,
                "short_term_context": [obs["text"]["short_term_context"] for obs in observations],
                "long_term_context": [obs["text"]["long_term_context"] for obs in observations],
                **flattened_obs,
            }
        )

        return episode_df, episode_log

    def close(self):
        """Close resources including agent file handles."""
        self.runner.close()


class RolloutCollector:
    def __init__(self, agent_factory, args):
        self.args = args
        self.runner = EnvironmentRunner(agent_factory=agent_factory, args=args)
        self._initialized = False

    def reset(self, seed=None):
        self._initialized = True
        return self.runner.reset(seed=seed)

    async def collect_rollout(self, steps_to_collect: int, break_on_done: bool = False, seed=None):
        """
        Collect fixed-count transition steps for RL-style rollouts.
        Returns:
            collected: List[Tuple[prompt_token_ids, action_token_ids, reward, truncated, done, logprob]]
            episodic_stats: List[Dict] with per-episode stats whenever an episode ends.
        """
        if seed is not None:
            self.reset(seed=seed)
        elif not self._initialized:
            self.reset(seed=seed)

        collected = []
        episodic_stats = []

        for _ in range(steps_to_collect):
            step_return = await self.runner.step()

            rew = step_return["reward"]
            term = step_return["term"]
            trun = step_return["trun"]
            info = step_return["info"]
            prompt_token_ids = step_return["prompt_token_ids"]
            action_token_ids = step_return["action_token_ids"]
            logprob = step_return["logprob"]

            collected.append((prompt_token_ids, action_token_ids, rew, trun, term or trun, logprob))

            last_prompt_token_ids = prompt_token_ids
            last_term = term
            last_trun = trun

            if term or trun:
                episodic_stats.append(info.get("episode_extra_stats", dict()))
                self.reset()
                if break_on_done:
                    break

        # Append a bootstrap transition placeholder for value computation at t+1.
        collected.append((last_prompt_token_ids, [0], -100, last_trun, last_term or last_trun, -100))

        return collected, episodic_stats

    def close(self):
        self.runner.close()


@ray.remote(max_concurrency=1)
class RemoteEpisodeCollector(EpisodeCollector):
    pass


@ray.remote(max_concurrency=1)
class RemoteRolloutCollector(RolloutCollector):
    pass


class Evaluator:
    def __init__(self, agent_factory, args: Args):
        self.args = args
        self.num_eval_workers = args.num_eval_workers
        self.collectors = (
            [RemoteEpisodeCollector.remote(agent_factory, args) for _ in range(args.num_eval_workers)]
            if args.use_distributed
            else [EpisodeCollector(agent_factory, args)]
        )
        self.rollout_workers = (
            [RemoteRolloutCollector.remote(agent_factory, args) for _ in range(args.num_eval_workers)]
            if args.use_distributed
            else [RolloutCollector(agent_factory, args)]
        )

    def _run_distributed(self, method_name, num_tasks):
        pending = []
        ref_to_actor = {}
        results = []
        idx = 0

        # Launch up to as many tasks as workers
        for i in range(min(self.num_eval_workers, num_tasks)):
            seed = set_seed(self.args.env_name, self.args.seed, idx)
            ref = getattr(self.collectors[i], method_name).remote(seed=seed)
            pending.append(ref)
            ref_to_actor[ref] = self.collectors[i]
            idx += 1

        pbar = tqdm(total=num_tasks, desc="Collecting episodes remotely", unit="task")

        # For the remainder, each time a worker finishes, submit a new task
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=None)
            ref = done[0]
            results.append(ray.get(ref))
            pbar.update(1)

            if idx < num_tasks:
                actor = ref_to_actor.pop(ref)
                seed = set_seed(self.args.env_name, self.args.seed, idx)
                ref = getattr(actor, method_name).remote(seed=seed)
                pending.append(ref)
                ref_to_actor[ref] = actor
                idx += 1

        pbar.close()
        return results

    def _run_local(self, method_name, num_tasks):
        results = []
        with tqdm(total=num_tasks, desc="Collecting episodes locally", unit="task") as pbar:
            for i in range(num_tasks):
                # Use the single collector for sequential execution
                seed = set_seed(self.args.env_name, self.args.seed, i)
                result = asyncio.run(getattr(self.collectors[0], method_name)(seed=seed))
                results.append(result)
                pbar.update(1)
        return results

    def _seed_for_rollout(self, rollout_idx, seed_sequence=None):
        if seed_sequence is not None:
            return seed_sequence[rollout_idx]
        return set_seed(self.args.env_name, self.args.seed, rollout_idx)

    def _run_distributed_collect_rollouts(self, n_rollouts, steps_to_collect, break_on_done=False, seed_sequence=None):
        if seed_sequence is not None and len(seed_sequence) != n_rollouts:
            raise ValueError(f"seed_sequence length {len(seed_sequence)} does not match n_rollouts={n_rollouts}")

        pending = []
        ref_to_actor = {}
        collected = [None] * n_rollouts
        episodic_stats = [None] * n_rollouts
        idx = 0

        # Launch up to as many rollout tasks as workers.
        for i in range(min(self.num_eval_workers, n_rollouts)):
            rollout_idx = idx
            seed = self._seed_for_rollout(rollout_idx, seed_sequence=seed_sequence)
            ref = self.rollout_workers[i].collect_rollout.remote(
                steps_to_collect=steps_to_collect,
                break_on_done=break_on_done,
                seed=seed,
            )
            pending.append(ref)
            ref_to_actor[ref] = (self.rollout_workers[i], rollout_idx)
            idx += 1

        # As each worker finishes, schedule the next rollout task on that worker.
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=None)
            ref = done[0]
            worker_collected, worker_stats = ray.get(ref)
            actor, rollout_idx = ref_to_actor.pop(ref)
            collected[rollout_idx] = worker_collected
            episodic_stats[rollout_idx] = worker_stats

            if idx < n_rollouts:
                rollout_idx = idx
                seed = self._seed_for_rollout(rollout_idx, seed_sequence=seed_sequence)
                ref = actor.collect_rollout.remote(
                    steps_to_collect=steps_to_collect,
                    break_on_done=break_on_done,
                    seed=seed,
                )
                pending.append(ref)
                ref_to_actor[ref] = (actor, rollout_idx)
                idx += 1

        return collected, episodic_stats

    def _run_local_collect_rollouts(self, n_rollouts, steps_to_collect, break_on_done=False, seed_sequence=None):
        if seed_sequence is not None and len(seed_sequence) != n_rollouts:
            raise ValueError(f"seed_sequence length {len(seed_sequence)} does not match n_rollouts={n_rollouts}")

        collected = []
        episodic_stats = []

        for i in range(n_rollouts):
            seed = self._seed_for_rollout(i, seed_sequence=seed_sequence)
            worker_collected, worker_stats = asyncio.run(
                self.rollout_workers[0].collect_rollout(
                    steps_to_collect=steps_to_collect,
                    break_on_done=break_on_done,
                    seed=seed,
                )
            )
            collected.append(worker_collected)
            episodic_stats.append(worker_stats)

        return collected, episodic_stats

    def collect_episodes(self, num_tasks):
        if self.args.use_distributed:
            return self._run_distributed("collect_episode", num_tasks)
        else:
            return self._run_local("collect_episode", num_tasks)

    def evaluate_episodes(self, num_tasks):
        if self.args.use_distributed:
            return self._run_distributed("evaluate_episode", num_tasks)
        else:
            return self._run_local("evaluate_episode", num_tasks)

    def collect_rollouts(self, n_rollouts, steps_to_collect, break_on_done=False, seed_sequence=None):
        if self.args.use_distributed:
            return self._run_distributed_collect_rollouts(
                n_rollouts=n_rollouts,
                steps_to_collect=steps_to_collect,
                break_on_done=break_on_done,
                seed_sequence=seed_sequence,
            )
        else:
            return self._run_local_collect_rollouts(
                n_rollouts=n_rollouts,
                steps_to_collect=steps_to_collect,
                break_on_done=break_on_done,
                seed_sequence=seed_sequence,
            )

    def shutdown(self):
        print("Shutting down evaluator...")
        # Close all collectors to clean up agent file handles
        for collector in self.collectors:
            try:
                if self.args.use_distributed:
                    ray.get(collector.close.remote())
                else:
                    collector.close()
            except Exception as e:
                print(f"Warning: Error closing collector: {e}")

        for worker in self.rollout_workers:
            try:
                if self.args.use_distributed:
                    ray.get(worker.close.remote())
                else:
                    worker.close()
            except Exception as e:
                print(f"Warning: Error closing rollout worker: {e}")
