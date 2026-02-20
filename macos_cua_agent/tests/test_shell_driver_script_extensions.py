from pathlib import Path
import tempfile

from cua_agent.utils.config import Settings
from macos_cua_agent.drivers.shell_driver import ShellDriver


def test_macos_shell_driver_filters_non_runnable_script_extensions() -> None:
    settings = Settings(
        script_allowed_extensions=".py,.sh,.js,.ps1,.bat,.cmd",
    )
    driver = ShellDriver(settings)

    assert driver.allowed_script_extensions == {".py", ".sh", ".js", ".ps1"}


def test_macos_shell_driver_rejects_bat_run_as_disallowed_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        script_path = workspace / "tools" / "deploy.bat"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("@echo off\necho hi\n", encoding="utf-8")

        settings = Settings(
            execution_profile="remote_cli",
            enable_shell=False,
            shell_workspace_root=str(workspace),
            script_allowed_extensions=".py,.sh,.js,.ps1,.bat,.cmd",
        )
        driver = ShellDriver(settings)

        result = driver.execute(
            {
                "type": "script_op",
                "operation": "run",
                "path": "tools/deploy.bat",
            }
        )

        assert result.success is False
        assert "extension not allowed" in result.reason
