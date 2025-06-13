import logging
import re
from dataclasses import dataclass
from typing import Any

from feedback_bottleneck.config.args import Args

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict math answer verifier.

You are given:
- a math problem
- the reference final answer
- optionally a reference solution
- a candidate answer from a student

Decide whether the candidate answer should count as correct.

Rules:
- Judge mathematical equivalence, not wording.
- Use the reference answer as the main ground truth.
- The candidate may include reasoning; focus on the final claimed answer.
- Be conservative. If the candidate answer is ambiguous or unsupported, mark it incorrect.
- Reply with the exact XML tags requested below."""


def build_math_verifier_messages(problem: str, reference_answer: str, reference_solution: str, candidate_answer: str):
    user_prompt = f"""Problem:
{problem}

Reference final answer:
{reference_answer}

Reference solution:
{reference_solution or "N/A"}

Candidate answer:
{candidate_answer}

Respond with exactly this format:
<verdict>correct|incorrect</verdict>
<reason>short justification</reason>"""

    return [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]


def parse_math_verifier_output(text: str) -> tuple[bool, str]:
    verdict_match = re.search(r"<verdict>(.*?)</verdict>", text, re.DOTALL | re.IGNORECASE)
    reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL | re.IGNORECASE)

    if verdict_match is None:
        reason = f"Verifier output missing <verdict>: {text!r}"
        logger.warning(reason)
        return False, reason

    verdict = verdict_match.group(1).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        reason = f"Unsupported verifier verdict {verdict!r} in output: {text!r}"
        logger.warning(reason)
        return False, reason

    reason = reason_match.group(1).strip() if reason_match is not None else ""
    return verdict == "correct", reason


@dataclass
class MathLLMVerification:
    correct: bool
    feedback: str
    raw_output: str


class MathLLMVerifier:
    def __init__(self, args: Args, llm_judge):
        self.args = args
        self.llm_judge = llm_judge
        self.sampling_params = vars(args.math_args.verifier_sampling_args).copy()

    async def verify(self, observation: dict[str, Any], candidate_answer: str) -> MathLLMVerification:
        messages = build_math_verifier_messages(
            problem=observation["problem"],
            reference_answer=observation["answer"],
            reference_solution=observation.get("solution", ""),
            candidate_answer=candidate_answer,
        )

        if self.args.use_distributed:
            response = await self.llm_judge.chat.remote(messages, self.sampling_params)
        else:
            response = await self.llm_judge.chat(messages, self.sampling_params)

        correct, reason = parse_math_verifier_output(response["text"])
        feedback = "Correct." if correct else f"Incorrect. {reason}".strip()
        return MathLLMVerification(
            correct=correct,
            feedback=feedback,
            raw_output=response["text"],
        )
