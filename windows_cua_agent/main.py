from __future__ import annotations

from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import configure_logging, get_logger
from windows_cua_agent.computer import create_computer


def _read_prompt() -> str | None:
    try:
        return input("\nEnter a prompt (blank to quit): ").strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__, level=settings.log_level)
    computer = create_computer(settings)

    try:
        computer.run_health_checks(settings, logger=logger)
    except Exception as exc:
        logger.error("%s", exc)
        return

    orchestrator = Orchestrator(settings, computer)

    while True:
        user_prompt = _read_prompt()
        if not user_prompt:
            logger.info("No prompt provided; exiting.")
            return
        logger.info("Starting session for prompt: %s", user_prompt)
        summary = orchestrator.run_task(user_prompt)
        logger.info("Task complete. Summary: %s", summary)


if __name__ == "__main__":
    run()
