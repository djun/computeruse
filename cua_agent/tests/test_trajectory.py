from cua_agent.agent.state_manager import ActionResult
from cua_agent.observability.trajectory import (
    TrajectoryRecorder,
    load_trajectory,
    replay_trajectory,
)


class _FakeComputer:
    def __init__(self) -> None:
        self.executed: list[dict] = []

    def execute(self, action: dict) -> ActionResult:
        self.executed.append(action)
        return ActionResult(success=True, reason="ok")


def test_recorder_writes_and_loads_and_strips_heavy_fields(tmp_path) -> None:
    path = tmp_path / "traj.jsonl"
    rec = TrajectoryRecorder(path)
    rec.record({"type": "left_click", "x": 10, "y": 20, "zoom_image": "BIGB64"}, success=True, frame_hash="h1")
    rec.record({"type": "type", "text": "hi"}, success=False, frame_hash="h2", reason="miss")

    records = load_trajectory(path)
    assert len(records) == 2
    assert records[0]["turn"] == 0
    assert records[0]["action"] == {"type": "left_click", "x": 10, "y": 20}  # heavy field stripped
    assert records[1]["success"] is False


def test_replay_reexecutes_actions_and_reports_match(tmp_path) -> None:
    path = tmp_path / "traj.jsonl"
    rec = TrajectoryRecorder(path)
    rec.record({"type": "left_click", "x": 1, "y": 2}, success=True)
    rec.record({"type": "noop"}, success=True)  # skipped on replay
    rec.record({"type": "type", "text": "x"}, success=True)

    computer = _FakeComputer()
    outcomes = replay_trajectory(computer, load_trajectory(path))

    # noop is skipped; the two real actions are executed in order.
    assert [a["type"] for a in computer.executed] == ["left_click", "type"]
    assert outcomes[1]["skipped"] is True
    assert outcomes[0]["replay_success"] is True
    assert outcomes[0]["matched"] is True


def test_load_missing_file_returns_empty(tmp_path) -> None:
    assert load_trajectory(tmp_path / "nope.jsonl") == []
