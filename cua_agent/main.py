from __future__ import annotations

import argparse

from cua_agent.computer.loader import load_computer
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import configure_logging, get_logger


def _read_prompt() -> str | None:
    try:
        return input("\nEnter a prompt (blank to quit): ").strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def run(adapter: str | None = None) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__, level=settings.log_level)

    try:
        computer = load_computer(settings, adapter_module=adapter)
    except Exception as exc:
        logger.error("%s", exc)
        return

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cua_agent")
    parser.add_argument(
        "--adapter",
        dest="adapter",
        default=None,
        help="Adapter package to use (e.g. macos_cua_agent, windows_cua_agent). Overrides CUA_ADAPTER.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(adapter=args.adapter)


if __name__ == "__main__":
    main()
