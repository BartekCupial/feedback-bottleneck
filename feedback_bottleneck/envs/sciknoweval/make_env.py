import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.dataset_env import DatasetBackedTextEnv
from feedback_bottleneck.envs.dataset_utils import DatasetExample, load_dataset_examples
from feedback_bottleneck.envs.env_wrapper import EnvWrapper
from feedback_bottleneck.envs.sciknoweval.evaluate import evaluate_supported_record, render_reference_answer


@dataclass(frozen=True)
class SciKnowTaskSpec:
    name: str
    scorer_kind: str
    domain: Optional[str] = None
    task: Optional[str] = None
    subtask: Optional[str] = None
    level: Optional[str] = None


SUPPORTED_SCIKNOWEVAL_TASKS = (
    SciKnowTaskSpec("chemical_literature_QA", "cls", domain="Chemistry", task="literature_multi_choice_question"),
    SciKnowTaskSpec("reaction_mechanism_inference", "cls", subtask="reaction_mechanism_inference"),
    SciKnowTaskSpec("chemical_detailed_understanding", "cls", domain="Chemistry", subtask="detailed_understanding"),
    SciKnowTaskSpec("chemical_hypothesis_verification", "cls", domain="Chemistry", subtask="hypothesis_verification"),
    SciKnowTaskSpec("molar_weight_calculation", "cls", task="molar_weight_calculation"),
    SciKnowTaskSpec("molecular_property_calculation", "cls", task="molecular_property_prediction", level="L3"),
    SciKnowTaskSpec("molecule_structure_prediction", "cls", task="molecule_structure_prediction"),
    SciKnowTaskSpec("reaction_prediction", "reaction", task="reaction_prediction"),
    SciKnowTaskSpec("retrosynthesis", "reaction", task="retrosynthesis"),
    SciKnowTaskSpec("balancing_chemical_equation", "filling", task="balancing_chemical_equation"),
    SciKnowTaskSpec("mol_toxicity_prediction", "cls", task="mol_toxicity_prediction"),
    SciKnowTaskSpec("chemical_laboratory_safety_test", "cls", domain="Chemistry", task="laboratory_safety_test"),
    SciKnowTaskSpec("biology_literature_QA", "cls", domain="Biology", task="literature_multi_choice_question"),
    SciKnowTaskSpec("biomedical_judgment_and_interpretation", "cls", subtask="biomedical_judgment_and_interpretation"),
    SciKnowTaskSpec("biological_detailed_understanding", "cls", domain="Biology", subtask="detailed_understanding"),
    SciKnowTaskSpec("biological_hypothesis_verification", "cls", domain="Biology", subtask="hypothesis_verification"),
    SciKnowTaskSpec("solubility_prediction", "cls", subtask="solubility_prediction"),
    SciKnowTaskSpec("beta_lactamase_activity_prediction", "cls", subtask="beta_lactamase_activity_prediction"),
    SciKnowTaskSpec("fluorescence_prediction", "cls", subtask="fluorescence_prediction"),
    SciKnowTaskSpec("GB1_ftness_prediction", "cls", subtask="GB1_ftness_prediction"),
    SciKnowTaskSpec("stability_prediction", "cls", subtask="stability_prediction"),
    SciKnowTaskSpec("Protein_Protein_Interaction", "cls", subtask="Protein_Protein_Interaction"),
    SciKnowTaskSpec("proteotoxicity_prediction", "cls", task="proteotoxicity_prediction"),
    SciKnowTaskSpec("biological_laboratory_safety_test", "cls", domain="Biology", task="laboratory_safety_test"),
    SciKnowTaskSpec("material_literature_QA", "cls", task="material_literature_QA"),
    SciKnowTaskSpec("material_hypothesis_verification", "cls", subtask="material_hypothesis_verification"),
    SciKnowTaskSpec("material_data_extraction", "cls", subtask="material_data_extraction"),
    SciKnowTaskSpec("material_detailed_understanding", "cls", subtask="material_detailed_understanding"),
    SciKnowTaskSpec(
        "valence_electron_difference_calculation",
        "cls",
        task="valence_electron_difference_calculation",
    ),
    SciKnowTaskSpec("lattice_volume_calculation", "cls", task="lattice_volume_calculation"),
    SciKnowTaskSpec("perovskite_stability_prediction", "cls", task="perovskite_stability_prediction"),
    SciKnowTaskSpec("diffusion_rate_analysis", "cls", task="diffusion_rate_analysis"),
    SciKnowTaskSpec("material_safety_QA", "cls", task="material_safety_QA"),
    SciKnowTaskSpec("material_toxicity_prediction", "cls", task="material_toxicity_prediction"),
    SciKnowTaskSpec("physics_literature_QA", "cls", task="physics_literature_QA"),
    SciKnowTaskSpec("physics_hypothesis_verification", "cls", subtask="physics_hypothesis_verification"),
    SciKnowTaskSpec("physics_detailed_understanding", "cls", subtask="physics_detailed_understanding"),
    SciKnowTaskSpec("general_physics_calculation", "cls", task="general_physics_calculation"),
    SciKnowTaskSpec("physics_safety_QA", "cls", task="physics_safety_QA"),
    SciKnowTaskSpec("physics_laboratory_safety_test", "cls", task="physics_laboratory_safety_test"),
)
SUPPORTED_TASK_NAMES = {spec.name for spec in SUPPORTED_SCIKNOWEVAL_TASKS}
SUPPORTED_TASK_NAMES_BY_TOKEN = {name.lower(): name for name in SUPPORTED_TASK_NAMES}


@dataclass
class SciKnowEvalExample(DatasetExample):
    task_name: str = ""
    domain: str = ""
    level: str = ""
    subtask: str = ""
    scorer_kind: str = ""


def _normalize_task_token(task: str) -> str:
    return str(task).strip().replace("-", "_").replace(" ", "_").lower()


def _normalize_question_type(value) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("SciKnowEval row has an empty `type` list.")
        value = value[0]
    return str(value).strip()


def _get_details(record: dict) -> dict:
    details = record.get("details") or {}
    if not isinstance(details, dict):
        raise ValueError("SciKnowEval row has a non-dict `details` field.")
    return details


def _get_detail_value(record: dict, key: str) -> Optional[str]:
    details = _get_details(record)
    value = details.get(key, record.get(key))
    if value is None:
        return None
    return str(value).strip()


def _record_matches_spec(record: dict, spec: SciKnowTaskSpec) -> bool:
    domain = record.get("domain")
    task = _get_detail_value(record, "task")
    subtask = _get_detail_value(record, "subtask")
    level = _get_detail_value(record, "level")

    if spec.domain is not None and str(domain).strip() != spec.domain:
        return False
    if spec.task is not None and task != spec.task:
        return False
    if spec.subtask is not None and subtask != spec.subtask:
        return False
    if spec.level is not None and level != spec.level:
        return False
    return True


def _resolve_supported_task_spec(record: dict) -> Optional[SciKnowTaskSpec]:
    for spec in SUPPORTED_SCIKNOWEVAL_TASKS:
        if _record_matches_spec(record, spec):
            return spec
    return None


def _format_problem(record: dict) -> str:
    question = record.get("question")
    if question is None:
        raise ValueError("SciKnowEval row is missing `question`.")

    parts = [str(question).strip()]
    choices = record.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if labels and texts:
        choice_lines = [f"{str(label).strip()}. {str(text).strip()}" for label, text in zip(labels, texts)]
        parts.extend(["", "Choices:", *choice_lines])

    return "\n".join(parts)


def _make_problem_id(task_name: str, problem: str, answer: str) -> str:
    digest = hashlib.sha1(f"{task_name}\n{problem}\n{answer}".encode("utf-8")).hexdigest()[:16]
    return f"{task_name}:{digest}"


def _normalize_record(record: dict, idx: int) -> SciKnowEvalExample:
    del idx
    spec = _resolve_supported_task_spec(record)
    if spec is None:
        raise ValueError(
            "Unsupported SciKnowEval row. "
            "This env only supports objective tasks with local scorers in the first pass."
        )

    domain = record.get("domain")
    if domain is None:
        raise ValueError("SciKnowEval row is missing `domain`.")

    level = _get_detail_value(record, "level")
    if level is None:
        raise ValueError("SciKnowEval row is missing `details.level`.")
    subtask = _get_detail_value(record, "subtask") or spec.subtask or spec.task or spec.name

    problem = _format_problem(record)
    reference_answer = render_reference_answer(record)
    source = str(record.get("source") or "sciknoweval_hf")

    return SciKnowEvalExample(
        problem=problem,
        answer=reference_answer,
        task_name=spec.name,
        domain=str(domain).strip(),
        level=level,
        subtask=subtask,
        scorer_kind=spec.scorer_kind,
        source=source,
        problem_id=_make_problem_id(spec.name, problem, reference_answer),
        raw_record=dict(record),
    )


def _dataset_row_matches_task(row: dict, canonical_task: str) -> bool:
    spec = _resolve_supported_task_spec(dict(row))
    return spec is not None and spec.name == canonical_task


def _apply_sciknoweval_dataset_filters(dataset, config: Args):
    domains = tuple(str(domain).strip() for domain in config.sciknoweval_args.domains if str(domain).strip())
    levels = tuple(str(level).strip() for level in config.sciknoweval_args.levels if str(level).strip())
    question_types = tuple(
        str(question_type).strip()
        for question_type in config.sciknoweval_args.question_types
        if str(question_type).strip()
    )

    if domains:
        dataset = dataset.filter(lambda row: str(row["domain"]).strip() in domains)

    if levels:
        dataset = dataset.filter(
            lambda row: str((row.get("details") or {}).get("level", row.get("level", ""))).strip() in levels
        )

    if question_types:
        dataset = dataset.filter(lambda row: _normalize_question_type(row["type"]) in question_types)

    return dataset


def _filter_sciknoweval_dataset(dataset, config: Args, task: str):
    dataset = _apply_sciknoweval_dataset_filters(dataset, config)
    normalized_task = _normalize_task_token(task)

    if normalized_task in {"default", "all", "all_objective"}:
        return dataset.filter(lambda row: _resolve_supported_task_spec(dict(row)) is not None)

    canonical_task = SUPPORTED_TASK_NAMES_BY_TOKEN.get(normalized_task)
    if canonical_task is None:
        supported = ", ".join(sorted(SUPPORTED_TASK_NAMES))
        raise ValueError(f"SciKnowEval task '{task}' is not supported by this env. " f"Supported tasks: {supported}")

    return dataset.filter(lambda row: _dataset_row_matches_task(row, canonical_task))


def load_sciknoweval_examples(config: Args, task: str) -> list[SciKnowEvalExample]:
    return load_dataset_examples(
        config.sciknoweval_args.dataset_path,
        config.sciknoweval_args.dataset_config,
        config.sciknoweval_args.dataset_split,
        normalize_record=_normalize_record,
        problem_split=config.sciknoweval_args.problem_split,
        problem_split_seed=config.sciknoweval_args.problem_split_seed,
        problem_split_test_size=config.sciknoweval_args.problem_split_test_size,
        split_label="sciknoweval",
        dataset_filter=lambda dataset: _filter_sciknoweval_dataset(dataset, config, task),
        empty_error_message=(
            f"No supported SciKnowEval examples found in {config.sciknoweval_args.dataset_path} for task '{task}'"
        ),
    )


class SciKnowEvalEnv(DatasetBackedTextEnv[SciKnowEvalExample]):
    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
        llm_judge=None,
    ):
        del render_mode, llm_judge
        super().__init__(
            env_name=env_name,
            task=task,
            config=config,
            examples=load_sciknoweval_examples(config, task),
            max_steps=int(config.sciknoweval_args.max_turns),
            max_action_length=int(config.sciknoweval_args.max_action_length),
        )

    def get_example_info(self, example: SciKnowEvalExample) -> dict[str, Any]:
        return {
            "task_name": example.task_name,
            "domain": example.domain,
            "level": example.level,
            "subtask": example.subtask,
            "scorer_kind": example.scorer_kind,
        }

    def evaluate_answer(self, answer: str, example: SciKnowEvalExample) -> tuple[bool, str]:
        if example.raw_record is None:
            raise ValueError("SciKnowEval example is missing `raw_record`, cannot evaluate answer.")

        correct = evaluate_supported_record(
            response=answer,
            record=example.raw_record,
            scorer_kind=example.scorer_kind,
        )
        return correct, "Correct." if correct else "Incorrect."

    def _build_observation(self) -> dict[str, Any]:
        example = self._require_current_example()
        attempts_text, verdict_text = self._format_attempt_history()
        remaining_turns = self.max_steps - len(self._attempts)

        return {
            "obs": {
                "problem": example.problem,
                "problem_id": example.problem_id,
                "source": example.source,
                "task_name": example.task_name,
                "domain": example.domain,
                "level": example.level,
                "subtask": example.subtask,
                "answer": example.answer,
                "attempts": list(self._attempts),
                "attempts_remaining": remaining_turns,
                "num_attempts": len(self._attempts),
                "last_feedback": verdict_text,
            },
            "text": {
                "short_term_context": verdict_text,
                "long_term_context": (
                    f"Domain: {example.domain}\n"
                    f"Level: {example.level}\n"
                    f"Task: {example.task_name}\n"
                    f"Subtask: {example.subtask}\n"
                    f"Problem:\n{example.problem}\n"
                    f"Attempts remaining: {remaining_turns}\n"
                    f"Previous attempts:\n{attempts_text}"
                ),
            },
        }


def make_sciknoweval_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = SciKnowEvalEnv(
        env_name,
        task,
        config,
        render_mode=render_mode,
        llm_judge=llm_judge,
    )
    return EnvWrapper(env, env_name, task, args=config)
