from cua_agent.benchmarks import BenchmarkTask, run_benchmark


def test_run_benchmark_aggregates_pass_rate_and_metrics() -> None:
    summaries = {
        "t1": {"success": True, "steps": 5, "tokens_used": 1200},
        "t2": {"success": False, "steps": 8, "tokens_used": 3000},
        "t3": {"steps": 3, "total_tokens": 500},  # success decided by predicate below
    }

    def run_fn(prompt: str) -> dict:
        return summaries[prompt]

    tasks = [
        BenchmarkTask(id="t1", prompt="t1"),
        BenchmarkTask(id="t2", prompt="t2"),
        BenchmarkTask(id="t3", prompt="t3", success=lambda s: s.get("steps", 0) <= 3),
    ]

    report = run_benchmark(run_fn, tasks)

    assert report.total == 3
    assert report.passed == 2  # t1 (success) + t3 (predicate)
    assert report.pass_rate == 2 / 3
    assert report.to_dict()["pass_rate"] == 0.6667
    by_id = {r.id: r for r in report.results}
    assert by_id["t1"].tokens == 1200
    assert by_id["t3"].tokens == 500
    assert by_id["t2"].passed is False


def test_run_benchmark_isolates_task_failures() -> None:
    def run_fn(prompt: str) -> dict:
        raise RuntimeError("boom")

    report = run_benchmark(run_fn, [BenchmarkTask(id="x", prompt="x")])

    assert report.total == 1
    assert report.passed == 0
    assert "boom" in report.results[0].error
