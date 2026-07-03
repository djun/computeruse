"""Benchmark harness for evaluating the agent on task suites.

The runner here is infrastructure-agnostic: it drives a callable that runs one
task and reports pass/fail + metrics. Wiring real suites (OSWorld-Verified,
Windows Agent Arena, MacArena) requires their task definitions and a
VM/sandbox — see ``runner.py`` docstring.
"""

from cua_agent.benchmarks.runner import (
    BenchmarkReport,
    BenchmarkTask,
    TaskResult,
    run_benchmark,
)

__all__ = ["BenchmarkReport", "BenchmarkTask", "TaskResult", "run_benchmark"]
