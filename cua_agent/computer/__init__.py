"""Computer tool abstractions and shared types."""

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.computer.loader import load_computer
from cua_agent.computer.types import DisplayInfo

__all__ = ["ComputerAdapter", "DisplayInfo", "load_computer"]
