"""SIMULATION_MODE / sends_real_input() semantics."""

from cua_agent.utils.config import Settings


def test_default_is_dry_run() -> None:
    # ENABLE_HID defaults False -> no real input.
    assert Settings(enable_hid=False, simulation_mode=False).sends_real_input() is False


def test_enable_hid_sends_real_input() -> None:
    assert Settings(enable_hid=True, simulation_mode=False).sends_real_input() is True


def test_simulation_mode_forces_dry_run_even_with_hid() -> None:
    # Explicit simulation switch overrides ENABLE_HID.
    assert Settings(enable_hid=True, simulation_mode=True).sends_real_input() is False


def test_simulation_mode_and_planner_reflector_default_to_core_model() -> None:
    settings = Settings()
    # Blank planner/reflector consolidate on the core model.
    assert settings.planner_model == settings.openrouter_model
    assert settings.reflector_model == settings.openrouter_model
