from __future__ import annotations

import importlib
import os
import platform
from typing import Callable

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.utils.config import Settings


def _default_adapter_module() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos_cua_agent"
    if system == "Windows":
        return "windows_cua_agent"
    raise RuntimeError(
        f"Unsupported platform {system!r}. Set CUA_ADAPTER to an installed adapter package."
    )


def load_computer(settings: Settings, adapter_module: str | None = None) -> ComputerAdapter:
    """
    Load a concrete `ComputerAdapter`.

    Resolution order:
    1) explicit `adapter_module`
    2) env var `CUA_ADAPTER`
    3) default per `platform.system()`
    """
    module_name = adapter_module or os.getenv("CUA_ADAPTER") or _default_adapter_module()
    try:
        module = importlib.import_module(f"{module_name}.computer")
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError(
            f"Failed to import adapter '{module_name}.computer'. "
            "Ensure the adapter package is installed and importable."
        ) from exc

    factory = getattr(module, "create_computer", None)
    if not callable(factory):
        raise RuntimeError(
            f"Adapter '{module_name}.computer' must expose create_computer(settings: Settings)."
        )

    create_fn = factory  # type: Callable[[Settings], ComputerAdapter]
    return create_fn(settings)

