from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MACOS_ADAPTER_ROOT = REPO_ROOT / "macos_cua_agent"
WINDOWS_ADAPTER_ROOT = REPO_ROOT / "windows_cua_agent"


def _python_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.rglob("*.py"))


def test_macos_adapter_does_not_duplicate_core_layers() -> None:
    duplicated_layers = ("agent", "memory", "orchestrator", "policies")
    duplicated_files: list[str] = []

    for layer in duplicated_layers:
        for py_file in _python_files(MACOS_ADAPTER_ROOT / layer):
            duplicated_files.append(py_file.relative_to(REPO_ROOT).as_posix())

    assert not duplicated_files, (
        "macOS adapter must not duplicate core layers from cua_agent/: "
        + ", ".join(duplicated_files)
    )


def test_macos_utils_are_platform_specific_only() -> None:
    utils_root = MACOS_ADAPTER_ROOT / "utils"
    found = {path.name for path in _python_files(utils_root)}
    expected = {"__init__.py", "health.py", "macos_integration.py"}
    assert found == expected


def test_windows_adapter_does_not_duplicate_core_layers() -> None:
    duplicated_layers = ("agent", "memory", "orchestrator")
    duplicated_files: list[str] = []

    for layer in duplicated_layers:
        for py_file in _python_files(WINDOWS_ADAPTER_ROOT / layer):
            duplicated_files.append(py_file.relative_to(REPO_ROOT).as_posix())

    assert not duplicated_files, (
        "Windows adapter must not duplicate core layers from cua_agent/: "
        + ", ".join(duplicated_files)
    )
