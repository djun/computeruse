"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.5")
    planner_api_key: str | None = None
    planner_base_url: str | None = None
    planner_model: str = os.getenv("PLANNER_MODEL", "anthropic/claude-3.5-sonnet")
    reflector_api_key: str | None = None
    reflector_base_url: str | None = None
    reflector_model: str = os.getenv("REFLECTOR_MODEL", "openai/gpt-5.1")
    enable_reflection: bool = _get_bool("ENABLE_REFLECTION", True)
    strict_step_completion: bool = _get_bool("STRICT_STEP_COMPLETION", True)
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None # Changed to None

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    encode_format: str = os.getenv("ENCODE_FORMAT", "JPEG")
    verify_delay_ms: int = int(os.getenv("VERIFY_DELAY_MS", "200"))
    settle_delay_ms: int = int(os.getenv("SETTLE_DELAY_MS", "500"))
    ssim_change_threshold: float = float(os.getenv("SSIM_CHANGE_THRESHOLD", "0.985"))
    max_steps: int = int(os.getenv("MAX_STEPS", "50"))
    max_failures: int = int(os.getenv("MAX_FAILURES", "5"))
    max_wall_clock_seconds: int | None = (
        int(os.getenv("MAX_WALL_CLOCK_SECONDS", "0")) or None
    )

    enable_hid: bool = _get_bool("ENABLE_HID", False)
    enable_semantic: bool = _get_bool("ENABLE_SEMANTIC", True)
    enable_shell: bool = _get_bool("ENABLE_SHELL", False)
    windows_auto_elevate: bool = _get_bool("WINDOWS_AUTO_ELEVATE", True)
    use_openrouter: bool = _get_bool("USE_OPENROUTER", True)
    planner_use_openrouter: bool = use_openrouter
    enable_embeddings: bool = _get_bool("ENABLE_EMBEDDINGS", False)
    memory_root: str | None = os.getenv("MEMORY_ROOT")

    shell_workspace_root: str = os.getenv("SHELL_WORKSPACE_ROOT", ".agent_shell")
    shell_max_runtime_s: int = int(os.getenv("SHELL_MAX_RUNTIME_S", "10"))
    shell_max_output_bytes: int = int(os.getenv("SHELL_MAX_OUTPUT_BYTES", "65536"))
    shell_allowed_commands: str = os.getenv("SHELL_ALLOWED_COMMANDS", "")

    # Browser/AppleScript safety
    browser_script_timeout_s: float = float(os.getenv("BROWSER_SCRIPT_TIMEOUT_S", "8"))
    browser_navigation_timeout_s: float = float(os.getenv("BROWSER_NAVIGATION_TIMEOUT_S", "12"))

    # Reasoning Tokens Configuration
    reasoning_effort: str | None = os.getenv("REASONING_EFFORT")  # high, medium, low
    reasoning_max_tokens: int | None = (
        int(os.getenv("REASONING_MAX_TOKENS", "0")) or None
    )

    def __post_init__(self):
        # Dynamically load OpenRouter settings
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_base_url is None:
            self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Dynamically load Planner settings
        if self.planner_api_key is None:
            self.planner_api_key = os.getenv("PLANNER_API_KEY") or self.openrouter_api_key
        if self.planner_base_url is None:
            self.planner_base_url = os.getenv("PLANNER_BASE_URL", self.openrouter_base_url)

        # Dynamically load Reflector settings
        if self.reflector_api_key is None:
            self.reflector_api_key = os.getenv("REFLECTOR_API_KEY") or self.openrouter_api_key
        if self.reflector_base_url is None:
            self.reflector_base_url = os.getenv("REFLECTOR_BASE_URL", self.openrouter_base_url)

        # Handle embedding_api_key
        if self.embedding_api_key is None:
            self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        if self.embedding_api_key is None:
            self.embedding_api_key = os.getenv("OPENAI_API_KEY")
        if self.embedding_api_key is None and self.openrouter_api_key:
            self.embedding_api_key = self.openrouter_api_key

        # Handle embedding_base_url
        if self.embedding_base_url is None:
            self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        if self.embedding_base_url is None and self.openrouter_api_key:
            self.embedding_base_url = self.openrouter_base_url
        elif self.embedding_base_url is None: # Default to OpenAI if nothing else is set
            self.embedding_base_url = "https://api.openai.com/v1"

        # Handle embedding_model
        if self.embedding_model is None:
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
