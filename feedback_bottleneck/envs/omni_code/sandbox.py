from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tasks import CodeTask, CodeTestSpec


def _normalize_text_output(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines()).strip()


def _normalize_structured(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_structured(item) for item in value]
    if isinstance(value, list):
        return [_normalize_structured(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_structured(val) for key, val in value.items()}
    if isinstance(value, str):
        return value.strip()
    return value


def _json_default(value: Any):
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Unsupported object for JSON serialization: {type(value)}")


@dataclass
class CodeCaseResult:
    index: int
    passed: bool
    input_preview: str = ""
    expected_preview: str = ""
    actual_preview: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error_type: str = ""


@dataclass
class CodeExecutionResult:
    passed: bool
    pass_rate: float
    num_passed: int
    num_total: int
    timeout: bool
    cases: list[CodeCaseResult] = field(default_factory=list)

    def feedback(self, max_chars: int = 2000) -> str:
        if self.passed:
            return f"All tests passed ({self.num_passed}/{self.num_total})."

        if not self.cases:
            return "No test results were produced."

        first_failure = next((case for case in self.cases if not case.passed), self.cases[-1])
        lines = [f"Passed {self.num_passed}/{self.num_total} tests."]
        if first_failure.error_type:
            lines.append(f"Failure type: {first_failure.error_type}")
        if first_failure.input_preview:
            lines.append(f"Input: {first_failure.input_preview}")
        if first_failure.expected_preview:
            lines.append(f"Expected: {first_failure.expected_preview}")
        if first_failure.actual_preview:
            lines.append(f"Actual: {first_failure.actual_preview}")
        if first_failure.stderr:
            lines.append(f"Stderr: {first_failure.stderr}")
        if first_failure.stdout and first_failure.stdout != first_failure.actual_preview:
            lines.append(f"Stdout: {first_failure.stdout}")

        text = "\n".join(lines).strip()
        return text[:max_chars]


class SubprocessCodeSandbox:
    """
    Best-effort local sandbox for Python solutions.

    This isolates execution in a temporary directory, clears most environment
    variables and enforces timeouts, but it is not a secure boundary against
    adversarial code. For stronger guarantees use a container or VM backend.
    """

    def __init__(self, python_executable: str = sys.executable, default_timeout_s: float = 2.0):
        self.python_executable = python_executable
        self.default_timeout_s = default_timeout_s

    def run(
        self,
        task: CodeTask,
        candidate_code: str,
        timeout_s: float | None = None,
        task_timeout_s: float | None = None,
    ) -> CodeExecutionResult:
        spec = task.tests
        effective_timeout = timeout_s or self.default_timeout_s
        deadline = time.perf_counter() + task_timeout_s if task_timeout_s is not None else None

        if spec.kind == "asserts":
            return self._run_assert_cases(candidate_code, spec, effective_timeout, deadline)
        if spec.kind == "call_based":
            return self._run_call_cases(candidate_code, spec, effective_timeout, deadline)
        if spec.kind == "stdin_stdout":
            return self._run_stdio_cases(candidate_code, spec, effective_timeout, deadline)
        raise ValueError(f"Unsupported code test kind: {spec.kind}")

    def _base_env(self) -> dict[str, str]:
        keep_keys = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TMP", "TEMP"}
        base = {key: value for key, value in os.environ.items() if key.upper() in keep_keys}
        base["PYTHONNOUSERSITE"] = "1"
        base["PYTHONDONTWRITEBYTECODE"] = "1"
        return base

    def _finalize(self, cases: list[CodeCaseResult]) -> CodeExecutionResult:
        num_total = len(cases)
        num_passed = sum(1 for case in cases if case.passed)
        timeout = any(case.error_type in {"timeout", "task_timeout"} for case in cases)
        pass_rate = float(num_passed) / float(num_total) if num_total else 0.0
        return CodeExecutionResult(
            passed=num_total > 0 and num_passed == num_total,
            pass_rate=pass_rate,
            num_passed=num_passed,
            num_total=num_total,
            timeout=timeout,
            cases=cases,
        )

    def _remaining_timeout(self, deadline: float | None, case_timeout_s: float) -> float | None:
        if deadline is None:
            return case_timeout_s
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0:
            return None
        return min(case_timeout_s, remaining_s)

    def _append_task_timeout_cases(
        self,
        cases: list[CodeCaseResult],
        *,
        start_idx: int,
        spec: CodeTestSpec,
    ):
        if spec.kind == "asserts":
            for idx in range(start_idx, len(spec.asserts)):
                cases.append(
                    CodeCaseResult(
                        index=idx,
                        passed=False,
                        input_preview=str(spec.asserts[idx])[:300],
                        error_type="task_timeout",
                    )
                )
            return

        for idx in range(start_idx, min(len(spec.inputs), len(spec.outputs))):
            input_preview = (
                _normalize_text_output(spec.inputs[idx])[:300]
                if spec.kind == "stdin_stdout"
                else json.dumps(spec.inputs[idx], default=_json_default)[:300]
            )
            expected_preview = (
                _normalize_text_output(spec.outputs[idx])[:300]
                if spec.kind == "stdin_stdout"
                else json.dumps(spec.outputs[idx], default=_json_default)[:300]
            )
            cases.append(
                CodeCaseResult(
                    index=idx,
                    passed=False,
                    input_preview=input_preview,
                    expected_preview=expected_preview,
                    error_type="task_timeout",
                )
            )

    def _run_stdio_cases(
        self,
        candidate_code: str,
        spec: CodeTestSpec,
        timeout_s: float,
        deadline: float | None,
    ) -> CodeExecutionResult:
        cases = []
        with tempfile.TemporaryDirectory(prefix="feedback_bottleneck_code_") as tmpdir:
            code_path = Path(tmpdir) / "candidate.py"
            code_path.write_text(candidate_code, encoding="utf-8")

            for idx, (case_input, expected_output) in enumerate(zip(spec.inputs, spec.outputs)):
                effective_case_timeout = self._remaining_timeout(deadline, timeout_s)
                if effective_case_timeout is None:
                    self._append_task_timeout_cases(cases, start_idx=idx, spec=spec)
                    break
                started = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [self.python_executable, "-I", "-B", str(code_path)],
                        input=str(case_input),
                        text=True,
                        capture_output=True,
                        cwd=tmpdir,
                        env=self._base_env(),
                        timeout=effective_case_timeout,
                    )
                    actual = _normalize_text_output(proc.stdout)
                    expected = _normalize_text_output(expected_output)
                    passed = proc.returncode == 0 and actual == expected
                    stderr = _normalize_text_output(proc.stderr)
                    error_type = ""
                    if proc.returncode != 0:
                        error_type = f"exit_code_{proc.returncode}"
                    elif not passed:
                        error_type = "wrong_answer"
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=passed,
                            input_preview=_normalize_text_output(case_input)[:300],
                            expected_preview=expected[:300],
                            actual_preview=actual[:300],
                            stdout=actual[:500],
                            stderr=stderr[:500],
                            duration_s=time.perf_counter() - started,
                            error_type=error_type,
                        )
                    )
                except subprocess.TimeoutExpired:
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=False,
                            input_preview=_normalize_text_output(case_input)[:300],
                            expected_preview=_normalize_text_output(expected_output)[:300],
                            duration_s=time.perf_counter() - started,
                            error_type="timeout",
                        )
                    )
        return self._finalize(cases)

    def _run_call_cases(
        self,
        candidate_code: str,
        spec: CodeTestSpec,
        timeout_s: float,
        deadline: float | None,
    ) -> CodeExecutionResult:
        cases = []
        with tempfile.TemporaryDirectory(prefix="feedback_bottleneck_code_") as tmpdir:
            code_path = Path(tmpdir) / "candidate.py"
            harness_path = Path(tmpdir) / "harness.py"
            code_path.write_text(candidate_code, encoding="utf-8")
            harness_path.write_text(
                textwrap.dedent(
                    """
                    import importlib.util
                    import json

                    with open("payload.json", "r", encoding="utf-8") as f:
                        payload = json.load(f)

                    spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    fn = getattr(module, payload["fn_name"])
                    args = payload["args"]
                    if isinstance(args, dict):
                        result = fn(**args)
                    elif isinstance(args, list):
                        result = fn(*args)
                    else:
                        result = fn(args)

                    print(json.dumps(result))
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            for idx, (case_input, expected_output) in enumerate(zip(spec.inputs, spec.outputs)):
                effective_case_timeout = self._remaining_timeout(deadline, timeout_s)
                if effective_case_timeout is None:
                    self._append_task_timeout_cases(cases, start_idx=idx, spec=spec)
                    break
                started = time.perf_counter()
                payload_path = Path(tmpdir) / "payload.json"
                payload_path.write_text(
                    json.dumps({"fn_name": spec.fn_name, "args": case_input}, default=_json_default),
                    encoding="utf-8",
                )
                try:
                    proc = subprocess.run(
                        [self.python_executable, "-I", "-B", str(harness_path)],
                        text=True,
                        capture_output=True,
                        cwd=tmpdir,
                        env=self._base_env(),
                        timeout=effective_case_timeout,
                    )
                    stderr = _normalize_text_output(proc.stderr)
                    stdout = _normalize_text_output(proc.stdout)
                    error_type = ""
                    actual_value = stdout
                    actual_normalized = stdout
                    if proc.returncode == 0:
                        try:
                            actual_value = json.loads(proc.stdout)
                            actual_normalized = _normalize_structured(actual_value)
                        except Exception:
                            actual_normalized = stdout
                    expected_normalized = _normalize_structured(expected_output)
                    passed = proc.returncode == 0 and actual_normalized == expected_normalized
                    if proc.returncode != 0:
                        error_type = f"exit_code_{proc.returncode}"
                    elif not passed:
                        error_type = "wrong_answer"
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=passed,
                            input_preview=json.dumps(case_input, default=_json_default)[:300],
                            expected_preview=json.dumps(expected_output, default=_json_default)[:300],
                            actual_preview=json.dumps(actual_value, default=_json_default)[:300]
                            if proc.returncode == 0
                            else stdout[:300],
                            stdout=stdout[:500],
                            stderr=stderr[:500],
                            duration_s=time.perf_counter() - started,
                            error_type=error_type,
                        )
                    )
                except subprocess.TimeoutExpired:
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=False,
                            input_preview=json.dumps(case_input, default=_json_default)[:300],
                            expected_preview=json.dumps(expected_output, default=_json_default)[:300],
                            duration_s=time.perf_counter() - started,
                            error_type="timeout",
                        )
                    )
        return self._finalize(cases)

    def _run_assert_cases(
        self,
        candidate_code: str,
        spec: CodeTestSpec,
        timeout_s: float,
        deadline: float | None,
    ) -> CodeExecutionResult:
        cases = []
        with tempfile.TemporaryDirectory(prefix="feedback_bottleneck_code_") as tmpdir:
            code_path = Path(tmpdir) / "candidate.py"
            harness_path = Path(tmpdir) / "harness.py"
            code_path.write_text(candidate_code, encoding="utf-8")

            for idx, assert_code in enumerate(spec.asserts):
                effective_case_timeout = self._remaining_timeout(deadline, timeout_s)
                if effective_case_timeout is None:
                    self._append_task_timeout_cases(cases, start_idx=idx, spec=spec)
                    break
                started = time.perf_counter()
                harness_path.write_text(
                    textwrap.dedent(
                        f"""
                        import importlib.util

                        spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        globals().update(module.__dict__)

                        {assert_code}
                        print("OK")
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                try:
                    proc = subprocess.run(
                        [self.python_executable, "-I", "-B", str(harness_path)],
                        text=True,
                        capture_output=True,
                        cwd=tmpdir,
                        env=self._base_env(),
                        timeout=effective_case_timeout,
                    )
                    stdout = _normalize_text_output(proc.stdout)
                    stderr = _normalize_text_output(proc.stderr)
                    passed = proc.returncode == 0
                    error_type = ""
                    if proc.returncode != 0:
                        error_type = f"exit_code_{proc.returncode}"
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=passed,
                            input_preview=assert_code[:300],
                            stdout=stdout[:500],
                            stderr=stderr[:500],
                            duration_s=time.perf_counter() - started,
                            error_type=error_type,
                        )
                    )
                except subprocess.TimeoutExpired:
                    cases.append(
                        CodeCaseResult(
                            index=idx,
                            passed=False,
                            input_preview=assert_code[:300],
                            duration_s=time.perf_counter() - started,
                            error_type="timeout",
                        )
                    )
        return self._finalize(cases)
