def _normalize_type(value) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("SciKnowEval row has an empty `type` list.")
        value = value[0]
    return str(value).strip()


def _get_choice_pairs(record: dict) -> list[tuple[str, str]]:
    choices = record.get("choices") or {}
    labels = [str(label).strip() for label in choices.get("label") or []]
    texts = [str(text).strip() for text in choices.get("text") or []]

    if labels and not texts:
        raise ValueError("SciKnowEval row exposes choice labels but no choice texts.")
    if texts and not labels:
        raise ValueError("SciKnowEval row exposes choice texts but no choice labels.")

    return list(zip(labels, texts))


def render_reference_answer(record: dict) -> str:
    answer_key = record.get("answerKey")
    choice_pairs = _get_choice_pairs(record)

    if answer_key is not None and choice_pairs:
        answer_key = str(answer_key).strip()
        for label, text in choice_pairs:
            if answer_key == label:
                return f"{label}. {text}"

    answer = record.get("answer")
    if answer is not None:
        return str(answer).strip()
    if answer_key is not None:
        return str(answer_key).strip()

    raise ValueError("SciKnowEval row does not expose `answer` or `answerKey`.")


def _get_single_score_tf(record: dict, response: str) -> bool:
    pred = response.strip()
    answer = str(record["answer"]).strip()

    if answer.lower() in pred.lower():
        return True
    if "true" in pred.lower():
        return answer == "Yes"
    if "false" in pred.lower():
        return answer == "No"
    return False


def _get_single_score_mcq(record: dict, response: str) -> bool:
    pred = response.strip()
    answer = str(record["answerKey"]).strip()
    choice_pairs = _get_choice_pairs(record)
    labels = [label for label, _ in choice_pairs]
    texts = [text for _, text in choice_pairs]

    if answer not in labels:
        raise ValueError(f"SciKnowEval answerKey {answer!r} not found among choice labels {labels!r}.")

    correct_option = texts[labels.index(answer)]
    wrong_options = [text for label, text in zip(labels, texts) if label != answer]

    if len(pred) == 0:
        return False
    if len(pred) == 1:
        return pred == answer
    if correct_option in pred and sum(int(option in pred) for option in wrong_options) == 0:
        return True

    for key in ["D", "C", "B", "A"]:
        if (
            f"{key}." in pred
            or f"{key}\n" in pred
            or f"{key})" in pred
            or f"{key} " in pred
            or f'"{key}' in pred
            or f'{key}"' in pred
            or pred[0] == key
        ):
            return key == answer

    return False


def evaluate_supported_record(*, response: str, record: dict, scorer_kind: str) -> bool:
    question_type = _normalize_type(record["type"])

    if scorer_kind == "cls":
        if "mcq" in question_type:
            return _get_single_score_mcq(record, response)
        if "true" in question_type:
            return _get_single_score_tf(record, response)
        raise ValueError(f"Unsupported SciKnowEval classification type: {question_type}")

    if scorer_kind == "reaction":
        if "mcq" in question_type:
            return _get_single_score_mcq(record, response)
        if question_type == "filling":
            candidates = [candidate.strip() for candidate in response.strip().split(".") if candidate.strip()]
            answer = str(record["answer"]).strip()
            return any(candidate in answer for candidate in candidates)
        raise ValueError(f"Unsupported SciKnowEval reaction type: {question_type}")

    if scorer_kind == "filling":
        answer = str(record["answer"]).strip()
        return answer in response.strip()

    raise ValueError(f"Unsupported SciKnowEval scorer kind: {scorer_kind}")
