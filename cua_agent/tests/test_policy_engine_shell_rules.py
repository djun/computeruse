from unittest.mock import patch
import tempfile
from pathlib import Path

from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.config import Settings


def _policy(*, shell_allowed_commands: str = "") -> PolicyEngine:
    settings = Settings(use_openrouter=False, shell_allowed_commands=shell_allowed_commands)
    return PolicyEngine("nonexistent_rules.yaml", settings=settings)


def test_legacy_shell_command_alias_blocks_sandbox_shell() -> None:
    policy = _policy()
    policy.rules["blocked_actions"] = ["shell_command"]

    decision = policy.evaluate({"type": "sandbox_shell", "cmd": "ls"})

    assert decision.allowed is False
    assert "action blocked" in decision.reason


@patch("cua_agent.policies.policy_engine.shutil.which", return_value="/usr/bin/python")
def test_shell_allowed_commands_env_allows_trusted_basename(_mock_which) -> None:
    policy = _policy(shell_allowed_commands="python")
    policy.rules["blocked_actions"] = []

    decision = policy.evaluate({"type": "sandbox_shell", "cmd": "python -V"})

    assert decision.allowed is True


@patch(
    "cua_agent.policies.policy_engine.os.path.realpath",
    return_value="/opt/homebrew/Cellar/python@3.12/3.12.1/bin/python3.12",
)
@patch("cua_agent.policies.policy_engine.shutil.which", return_value="/opt/homebrew/bin/python")
def test_shell_allowed_commands_env_allows_trusted_symlink_path(
    _mock_which, _mock_realpath
) -> None:
    policy = _policy(shell_allowed_commands="python")
    policy.rules["blocked_actions"] = []

    decision = policy.evaluate({"type": "sandbox_shell", "cmd": "python -V"})

    assert decision.allowed is True


@patch("cua_agent.policies.policy_engine.shutil.which", return_value="/tmp/python")
def test_shell_allowed_commands_rejects_untrusted_resolution(_mock_which) -> None:
    policy = _policy(shell_allowed_commands="python")
    policy.rules["blocked_actions"] = []

    decision = policy.evaluate({"type": "sandbox_shell", "cmd": "python -V"})

    assert decision.allowed is False
    assert "allowlisted" in decision.reason


@patch("cua_agent.policies.policy_engine.shutil.which", return_value="/usr/bin/python")
def test_blocked_shell_basename_takes_precedence(_mock_which) -> None:
    policy = _policy(shell_allowed_commands="python")
    policy.rules["blocked_actions"] = []
    policy.rules["blocked_shell_basenames"] = ["python"]

    decision = policy.evaluate({"type": "sandbox_shell", "cmd": "python -V"})

    assert decision.allowed is False
    assert decision.reason == "command blocked: python"


def test_script_path_must_be_relative_to_workspace() -> None:
    policy = _policy()
    policy.rules["blocked_actions"] = []

    decision = policy.evaluate(
        {"type": "script_op", "operation": "write", "path": "../escape.py", "content": "print('x')"}
    )

    assert decision.allowed is False
    assert "relative to workspace" in decision.reason


def test_script_write_with_risky_import_requires_hitl() -> None:
    policy = _policy()
    policy.rules["blocked_actions"] = []

    decision = policy.evaluate(
        {
            "type": "script_op",
            "operation": "write",
            "path": "tools/check.py",
            "content": "import socket\nprint('ok')\n",
        }
    )

    assert decision.allowed is True
    assert decision.hitl_required is True
    assert "risky pattern" in decision.reason


def test_script_run_blocks_disallowed_extension() -> None:
    policy = _policy()
    policy.rules["blocked_actions"] = []
    policy.rules["allowed_script_extensions"] = [".py"]

    decision = policy.evaluate(
        {
            "type": "script_op",
            "operation": "run",
            "path": "tools/deploy.sh",
        }
    )

    assert decision.allowed is False
    assert "extension not allowed" in decision.reason


def test_script_run_with_risky_file_requires_hitl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        script_path = workspace / "tools" / "scan.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("import subprocess\nprint('ok')\n", encoding="utf-8")

        settings = Settings(
            use_openrouter=False,
            shell_workspace_root=str(workspace),
            script_allowed_extensions=".py",
        )
        policy = PolicyEngine("nonexistent_rules.yaml", settings=settings)
        policy.rules["blocked_actions"] = []

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/scan.py",
            }
        )

        assert decision.allowed is True
        assert decision.hitl_required is True
        assert "risky pattern" in decision.reason


def test_script_run_blocks_blocked_shell_command_in_script() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        script_path = workspace / "tools" / "download.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("curl https://example.com\n", encoding="utf-8")

        settings = Settings(
            use_openrouter=False,
            shell_workspace_root=str(workspace),
            script_allowed_extensions=".sh",
        )
        policy = PolicyEngine("nonexistent_rules.yaml", settings=settings)
        policy.rules["blocked_actions"] = []
        policy.rules["blocked_shell_basenames"] = ["curl"]

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/download.sh",
            }
        )

        assert decision.allowed is False
        assert "blocked command" in decision.reason


def test_script_run_with_destructive_shell_pattern_requires_hitl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        script_path = workspace / "tools" / "wipe.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("rm -rf ./build\n", encoding="utf-8")

        settings = Settings(
            use_openrouter=False,
            shell_workspace_root=str(workspace),
            script_allowed_extensions=".sh",
        )
        policy = PolicyEngine("nonexistent_rules.yaml", settings=settings)
        policy.rules["blocked_actions"] = []

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/wipe.sh",
            }
        )

        assert decision.allowed is True
        assert decision.hitl_required is True
        assert "dangerous shell pattern" in decision.reason


def test_script_run_uses_cwd_for_policy_target_resolution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        root_script = workspace / "run.py"
        root_script.write_text("print('safe')\n", encoding="utf-8")

        sub_script = workspace / "subdir" / "run.py"
        sub_script.parent.mkdir(parents=True, exist_ok=True)
        sub_script.write_text("import subprocess\nprint('risky')\n", encoding="utf-8")

        settings = Settings(
            use_openrouter=False,
            shell_workspace_root=str(workspace),
            script_allowed_extensions=".py",
        )
        policy = PolicyEngine("nonexistent_rules.yaml", settings=settings)
        policy.rules["blocked_actions"] = []

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "cwd": "subdir",
                "path": "run.py",
            }
        )

        assert decision.allowed is True
        assert decision.hitl_required is True
        assert "risky pattern" in decision.reason


def test_default_script_settings_do_not_override_yaml_extension_rules() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rules_path = Path(tmp) / "rules.yaml"
        rules_path.write_text("allowed_script_extensions:\n  - .py\n", encoding="utf-8")

        settings = Settings(
            use_openrouter=False,
            script_allowed_extensions=".py,.sh,.js,.ps1,.bat,.cmd",
        )
        policy = PolicyEngine(str(rules_path), settings=settings)
        policy.rules["blocked_actions"] = []

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/deploy.sh",
            }
        )

        assert decision.allowed is False
        assert "extension not allowed" in decision.reason


def test_explicit_script_settings_can_override_yaml_extension_rules() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rules_path = Path(tmp) / "rules.yaml"
        rules_path.write_text("allowed_script_extensions:\n  - .py\n", encoding="utf-8")

        settings = Settings(use_openrouter=False, script_allowed_extensions=".py,.sh")
        policy = PolicyEngine(str(rules_path), settings=settings)
        policy.rules["blocked_actions"] = []

        decision = policy.evaluate(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/deploy.sh",
            }
        )

        assert decision.allowed is True
