from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from feedback_bottleneck.config.args import Args, RefinementPromptArgs
from feedback_bottleneck.llm.agent.base import BaseAgent, BaseFormatter

Observation = Dict[str, Any]
Message = Dict[str, Any]
StepObservationBuilder = Callable[[Dict[str, Any]], Observation]
CorrectnessFn = Callable[[Dict[str, Any]], bool]
ActionNormalizer = Callable[[Optional[str]], str]

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a careful tutor. Read the student's latest attempt, identify the most important mistake, "
    "and give targeted feedback without revealing the final answer."
)


@dataclass(frozen=True)
class RefinementPromptSpec:
    system_prompt: str
    student_initial_prompt: Callable[[Observation], str]
    student_refinement_prompt: Callable[[Observation, str], str]
    teacher_initial_prompt: Callable[[Observation, str], str]
    teacher_refinement_prompt: Callable[[Observation, str], str]
    teacher_system_prompt: Optional[str] = None


def _join_prompt_sections(*sections: str) -> str:
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def render_task_block(task_name: Optional[str]) -> str:
    if not task_name:
        return ""
    return f"Task:\n{task_name}"


def render_student_response_instruction(final_output_instruction: str) -> str:
    return f"You may reason in <think>...</think>. {final_output_instruction}"


def render_ground_truth(
    prompt_args: RefinementPromptArgs,
    *,
    solution_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    none_message: str = "You do not have access to the ground truth solution or final answer.",
) -> str:
    if prompt_args.teacher_reference_mode == "none":
        return none_message
    if prompt_args.teacher_reference_mode == "solution":
        if solution_text is not None:
            return solution_text
        raise ValueError("teacher_reference_mode='solution' requires a solution_text")
    if prompt_args.teacher_reference_mode == "answer":
        if answer_text is not None:
            return answer_text
        raise ValueError("teacher_reference_mode='answer' requires an answer_text block")
    raise ValueError(f"Unsupported teacher_reference_mode: {prompt_args.teacher_reference_mode}")


def render_teacher_output_instruction(
    prompt_args: RefinementPromptArgs,
    *,
    extra_rules: Optional[List[str]] = None,
) -> str:
    rules = list(extra_rules or [])
    if prompt_args.teacher_feedback_word_limit > 0:
        rules.append(f"Keep the final feedback under {prompt_args.teacher_feedback_word_limit} words.")

    if prompt_args.teacher_prompt_style == "plain":
        return "\n".join(rules)
    if prompt_args.teacher_prompt_style == "feedback_tag":
        return "\n".join(
            [
                *rules,
                "Return only:",
                "<feedback>",
                "your feedback",
                "</feedback>",
            ]
        )
    if prompt_args.teacher_prompt_style == "reasoning_feedback_tag":
        return "\n".join(
            [
                *rules,
                "You may reason about the mistake first inside <think>...</think> tags.",
                "Your final feedback message must appear exactly once inside <feedback>...</feedback> tags.",
            ]
        )
    raise ValueError(f"Unsupported teacher_prompt_style: {prompt_args.teacher_prompt_style}")


def get_teacher_system_prompt(prompt_spec: RefinementPromptSpec) -> str:
    return prompt_spec.teacher_system_prompt or prompt_spec.system_prompt


def _text_message(role: str, text: str) -> Message:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def _trim_history_for_user_turn(history: deque, max_turns: int) -> None:
    assert max_turns >= 0
    max_messages = 1 + 2 * max_turns
    if not history:
        return

    max_recent_messages = max_messages - 1
    if len(history) <= max_messages:
        return

    first_user = history[0]
    recent_messages = list(history)[-max_recent_messages:] if max_recent_messages else []
    history.clear()
    history.append(first_user)
    history.extend(recent_messages)


def _default_is_step_correct(step_data: Dict[str, Any]) -> bool:
    return bool(float(step_data.get("rewards", 0.0)) == 1.0)


def _default_normalize_action(action: Optional[str]) -> str:
    return (action or "").strip()


class NaiveRefinementFormatter(BaseFormatter):
    def __init__(
        self,
        args: Args,
        *,
        prompt_spec: RefinementPromptSpec,
        step_observation_builder: StepObservationBuilder,
    ):
        self.prompt_spec = prompt_spec
        self.step_observation_builder = step_observation_builder
        super().__init__(args)

    def reset(self):
        super().reset()
        self.chat_history = deque()

    def build_observation_from_step(self, step_data: Dict[str, Any]) -> Observation:
        return self.step_observation_builder(step_data)

    def append_user_turn(self, obs: Observation):
        self.chat_history.append(_text_message("user", self.prompt_spec.student_initial_prompt(obs)))
        _trim_history_for_user_turn(self.chat_history, self.max_history)

    def append_assistant_turn(self, answer: str):
        self.chat_history.append(_text_message("assistant", answer))

    def get_prompt(self, role="actor", **kwargs):
        del role, kwargs
        return [_text_message("system", self.prompt_spec.system_prompt), *self.chat_history]

    def generate_sft_samples(self, episode_iterator):
        self.reset()
        episode_steps = list(episode_iterator)

        for step_data in episode_steps:
            observation = self.build_observation_from_step(step_data)
            self.append_observation(observation)
            self.append_user_turn(observation)

            messages = self.get_prompt(role="actor")
            target = step_data["text_actions"]
            yield messages, target

            self.append_action(target)
            self.append_assistant_turn(target)


class HierarchicalRefinementFormatter(BaseFormatter):
    def __init__(
        self,
        args: Args,
        *,
        prompt_spec: RefinementPromptSpec,
        step_observation_builder: StepObservationBuilder,
        is_step_correct: CorrectnessFn = _default_is_step_correct,
        normalize_action_for_change: ActionNormalizer = _default_normalize_action,
    ):
        self.prompt_spec = prompt_spec
        self.step_observation_builder = step_observation_builder
        self.is_step_correct = is_step_correct
        self.normalize_action_for_change = normalize_action_for_change
        super().__init__(args)

    def reset(self):
        super().reset()
        self.student_chat_history = deque()
        self.teacher_chat_history = deque()

    def build_observation_from_step(self, step_data: Dict[str, Any]) -> Observation:
        return self.step_observation_builder(step_data)

    def append_student_prompt(self, obs: Observation, prev_solution: Optional[str], feedback: Optional[str]):
        del prev_solution
        if feedback:
            prompt_text = self.prompt_spec.student_refinement_prompt(obs, feedback)
        else:
            prompt_text = self.prompt_spec.student_initial_prompt(obs)

        self.student_chat_history.append(_text_message("user", prompt_text))
        _trim_history_for_user_turn(self.student_chat_history, self.max_history)

    def append_teacher_prompt(self, obs: Observation, prev_solution: str):
        if len(self.teacher_chat_history) == 0:
            prompt_text = self.prompt_spec.teacher_initial_prompt(obs, prev_solution)
        else:
            prompt_text = self.prompt_spec.teacher_refinement_prompt(obs, prev_solution)

        self.teacher_chat_history.append(_text_message("user", prompt_text))
        _trim_history_for_user_turn(self.teacher_chat_history, self.max_history)

    def append_teacher_feedback(self, feedback: str):
        self.teacher_chat_history.append(_text_message("assistant", feedback))

    def append_student_solution(self, solution: str):
        self.student_chat_history.append(_text_message("assistant", solution))

    def get_prompt(self, role: str, **kwargs):
        del kwargs
        if role == "teacher":
            return [_text_message("system", get_teacher_system_prompt(self.prompt_spec)), *self.teacher_chat_history]
        if role == "student":
            return [_text_message("system", self.prompt_spec.system_prompt), *self.student_chat_history]
        raise ValueError(f"Unsupported role: {role}")

    def build_student_query_messages(
        self,
        observation: Observation,
        prev_solution: Optional[str],
        feedback: Optional[str],
    ) -> List[Message]:
        messages = [
            _text_message("system", self.prompt_spec.system_prompt),
            _text_message("user", self.prompt_spec.student_initial_prompt(observation)),
        ]

        if prev_solution:
            messages.append(_text_message("assistant", prev_solution))

        if feedback:
            messages.append(_text_message("user", self.prompt_spec.student_refinement_prompt(observation, feedback)))

        return messages

    def build_student_target_messages(self, observation: Observation, corrected_solution: str) -> List[Message]:
        return [
            _text_message("system", self.prompt_spec.system_prompt),
            _text_message("user", self.prompt_spec.student_initial_prompt(observation)),
            _text_message("assistant", corrected_solution),
        ]

    def generate_sft_samples(self, episode_iterator):
        self.reset()
        episode_steps = list(episode_iterator)

        for step_data in episode_steps:
            observation = self.build_observation_from_step(step_data)
            self.append_observation(observation)

            prev_solution = self.act_history[-1] if self.act_history else None
            feedback = step_data["judge_feedbacks"] if prev_solution else None

            messages = self.build_student_query_messages(observation, prev_solution, feedback)
            target = step_data["text_actions"]
            yield messages, target

            self.append_student_prompt(observation, prev_solution, feedback)
            self.append_action(target)
            self.append_student_solution(target)

    def generate_crsft_samples(self, episode_iterator):
        self.reset()
        episode_steps = list(episode_iterator)
        if not episode_steps:
            return

        episode_solved = int(self.is_step_correct(episode_steps[-1]))
        num_turns_in_episode = len(episode_steps)

        for idx, step_data in enumerate(episode_steps):
            observation = self.build_observation_from_step(step_data)
            self.append_observation(observation)

            prev_solution = self.act_history[-1] if self.act_history else None
            feedback = step_data["judge_feedbacks"] if prev_solution else None
            messages = self.build_student_query_messages(observation, prev_solution, feedback)
            target = step_data["text_actions"]

            if feedback:
                prev_norm = self.normalize_action_for_change(episode_steps[idx - 1]["text_actions"])
                curr_norm = self.normalize_action_for_change(target)
                changed_answer = prev_norm != curr_norm
            else:
                changed_answer = False

            solved = self.is_step_correct(step_data)
            target_messages = self.build_student_target_messages(observation, target)
            metadata = {
                "episode_id": step_data["episode_id"],
                "problem": observation["obs"]["problem"],
                "problem_id": observation["obs"]["problem_id"],
                "source": observation["obs"]["source"],
                "turn_index": idx,
                "changed_answer": int(changed_answer),
                "solved": int(solved),
                "episode_solved": episode_solved,
                "is_last_turn": int(idx == num_turns_in_episode - 1),
                "num_turns_in_episode": num_turns_in_episode,
            }

            yield messages, target, messages, target_messages, metadata

            self.append_student_prompt(observation, prev_solution, feedback)
            self.append_action(target)
            self.append_student_solution(target)


class NaiveRefinementAgent(BaseAgent):
    def __init__(
        self,
        llm_actor,
        prompt_formatter,
        args: Args,
        *,
        answer_tag: Optional[str] = None,
    ):
        super().__init__(llm_actor, prompt_formatter, args)
        self.answer_tag = answer_tag

    def _extract_action(self, solution: str) -> str:
        if self.answer_tag is None:
            return solution
        answer = self.parse_tag_content(solution, [self.answer_tag])[0]
        return answer or solution

    async def get_action(self, obs: Observation):
        self.prompt_formatter.append_observation(obs)
        self.prompt_formatter.append_user_turn(obs)

        messages = self.prompt_formatter.get_prompt(role="actor")
        output = await self.generate(messages)
        solution = output["text"].strip()

        self.prompt_formatter.append_action(solution)
        self.prompt_formatter.append_assistant_turn(solution)

        return {
            "action": self._extract_action(solution),
            "plan": "",
            "judge_feedback": "",
            "logprob": output["logprobs"],
            "prompt_token_ids": output["prompt_token_ids"],
            "action_token_ids": output["token_ids"],
            "raw_output": output["text"],
        }


class HierarchicalRefinementAgent(BaseAgent):
    def __init__(
        self,
        llm_actor,
        prompt_formatter,
        args: Args,
        *,
        answer_tag: Optional[str] = None,
    ):
        super().__init__(llm_actor, prompt_formatter, args)
        self.answer_tag = answer_tag
        self.current_feedback = ""
        self.teacher_parse_feedback_tags = args.refinement_prompt.teacher_parse_feedback_tags

    def reset(self):
        super().reset()
        self.current_feedback = ""

    def _extract_action(self, solution: str) -> str:
        if self.answer_tag is None:
            return solution
        answer = self.parse_tag_content(solution, [self.answer_tag])[0]
        return answer or solution

    def parse_teacher_feedback(self, raw_teacher_text: str) -> str:
        raw_teacher_text = str(raw_teacher_text).strip()
        if not self.teacher_parse_feedback_tags:
            return raw_teacher_text or "No feedback provided"
        tagged_feedback = self.parse_tag_content(raw_teacher_text, ["feedback"])[-1]
        return tagged_feedback or raw_teacher_text or "No feedback provided"

    async def generate_teacher_feedback(self, teacher_messages: List[Message]):
        return await self.generate(teacher_messages)

    async def get_action(self, obs: Observation):
        self.prompt_formatter.append_observation(obs)

        teacher_feedback = ""
        prev_attempt = self.prompt_formatter.act_history[-1] if self.prompt_formatter.act_history else None

        if prev_attempt:
            self.prompt_formatter.append_teacher_prompt(obs, prev_attempt)
            teacher_messages = self.prompt_formatter.get_prompt("teacher")
            teacher_res = await self.generate_teacher_feedback(teacher_messages)
            raw_teacher_text = teacher_res["text"].strip()
            teacher_feedback = self.parse_teacher_feedback(raw_teacher_text)
            self.current_feedback = teacher_feedback
            self.prompt_formatter.append_teacher_feedback(teacher_feedback)
        else:
            teacher_res = {"text": "", "logprobs": None, "token_ids": []}

        self.prompt_formatter.append_student_prompt(obs, prev_attempt, self.current_feedback)
        student_messages = self.prompt_formatter.get_prompt("student")
        student_res = await self.generate(student_messages)
        solution = student_res["text"].strip()
        self.prompt_formatter.append_student_solution(solution)
        self.prompt_formatter.append_action(solution)

        return dict(
            action=self._extract_action(solution),
            plan=teacher_feedback,
            judge_feedback=teacher_feedback,
            logprob=student_res["logprobs"],
            prompt_token_ids=student_res["prompt_token_ids"],
            action_token_ids=student_res["token_ids"],
            raw_output=f"<teacher>\n{teacher_res['text']}\n</teacher>\n<student>\n{student_res['text']}\n</student>",
        )


class HierarchicalSeparateRefinementAgent(HierarchicalRefinementAgent):
    def __init__(
        self,
        llm_judge,
        llm_actor,
        prompt_formatter,
        args: Args,
        *,
        answer_tag: Optional[str] = None,
    ):
        super().__init__(llm_actor, prompt_formatter, args, answer_tag=answer_tag)
        self.llm_judge = llm_judge
        self.sampling_params_judge = vars(args.llm_judge.sampling_args)

    async def generate_teacher_feedback(self, teacher_messages: List[Message]):
        return await self.generate(teacher_messages, self.llm_judge, self.sampling_params_judge)
