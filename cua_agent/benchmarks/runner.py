"""Infrastructure-agnostic benchmark runner.

Drives a ``run_fn(prompt) -> summary`` over a list of tasks, evaluates each with
a per-task ``success`` predicate, and aggregates pass rate + metrics (steps,
tokens, duration when the summary exposes them).

Real suites need environment setup this runner deliberately does NOT own:

* OSWorld-Verified — Ubuntu VM per task, state-based checkers (xlang-ai/OSWorld).
* Windows Agent Arena — Windows 11 VM on Azure (microsoft/WindowsAgentArena).
* MacArena / macOSWorld — macOS app tasks.

To wire a suite: build ``BenchmarkTask`` objects from its task list (prompt +
success checker), pass a ``run_fn`` that runs the agent inside that suite's
sandbox, and feed the result to ``run_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# A task's success predicate receives the run summary and returns pass/fail.
SuccessFn = Callable[[dict[str, Any]], bool]
# Runs one task by prompt and returns a summary dict (e.g. Orchestrator.run_task).
RunFn = Callable[[str], dict[str, Any]]


@dataclass
class BenchmarkTask:
    id: str
    prompt: str
    success: Optional[SuccessFn] = None


@dataclass
class TaskResult:
    id: str
    passed: bool
    error: str = ""
    steps: Optional[int] = None
    tokens: Optional[int] = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    results: list[TaskResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "results": [
                {
                    "id": r.id,
                    "passed": r.passed,
                    "error": r.error,
                    "steps": r.steps,
                    "tokens": r.tokens,
                }
                for r in self.results
            ],
        }


def _metric(summary: dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def run_benchmark(
    run_fn: RunFn,
    tasks: Sequence[BenchmarkTask],
    *,
    on_result: Optional[Callable[[TaskResult], None]] = None,
) -> BenchmarkReport:
    """Run each task via ``run_fn`` and aggregate a report."""
    report = BenchmarkReport()
    for task in tasks:
        try:
            summary = run_fn(task.prompt) or {}
            passed = bool(task.success(summary)) if task.success else bool(summary.get("success"))
            result = TaskResult(
                id=task.id,
                passed=passed,
                steps=_metric(summary, "steps", "step_count"),
                tokens=_metric(summary, "tokens", "tokens_used", "total_tokens"),
                summary=summary,
            )
        except Exception as exc:  # a failing task must not abort the suite
            result = TaskResult(id=task.id, passed=False, error=str(exc))
        report.results.append(result)
        if on_result is not None:
            on_result(result)
    return report
