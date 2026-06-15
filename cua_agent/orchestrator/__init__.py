"""Planning and orchestration core."""

__all__ = ["Orchestrator"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from cua_agent.orchestrator.orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(name)
