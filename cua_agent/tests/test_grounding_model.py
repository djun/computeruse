from types import SimpleNamespace
from unittest.mock import MagicMock

from cua_agent.computer.types import DisplayInfo
from cua_agent.grounding.grounding_model import GroundingModelClient
from cua_agent.utils.config import Settings

_DISPLAY = DisplayInfo(
    logical_width=1280,
    logical_height=720,
    physical_width=2560,
    physical_height=1440,
    scale_factor=2.0,
)


def _client() -> GroundingModelClient:
    return GroundingModelClient(Settings(enable_uitars_grounder=False))


def test_parse_absolute_pixel_pair() -> None:
    assert _client()._parse_coordinates("(640, 360)", 1280, 720) == (640.0, 360.0)


def test_parse_uitars_action_string() -> None:
    text = "Thought: click the button\nAction: click(start_box='(100,200)')"
    assert _client()._parse_coordinates(text, 1280, 720) == (100.0, 200.0)


def test_parse_in_frame_pair_is_absolute() -> None:
    # In-frame values are treated as absolute pixels (UI-TARS-1.5 emits absolute).
    assert _client()._parse_coordinates("(500, 500)", 1280, 720) == (500.0, 500.0)


def test_parse_out_of_frame_small_pair_is_scaled_from_1000() -> None:
    # <=1000 but outside the frame => older 0-1000 normalized space; scale to pixels.
    x, y = _client()._parse_coordinates("(800, 900)", 1280, 720)
    assert (round(x), round(y)) == (1024, 648)


def test_parse_clamps_out_of_bounds() -> None:
    x, y = _client()._parse_coordinates("(5000, 5000)", 1280, 720)
    assert (x, y) == (1280.0, 720.0)


def test_parse_returns_none_without_numbers() -> None:
    assert _client()._parse_coordinates("no coordinates here", 1280, 720) is None


def test_locate_disabled_returns_none() -> None:
    client = _client()
    assert client.available is False
    assert client.locate("frame", "Submit button", _DISPLAY) is None


def test_locate_parses_model_output_and_counts_tokens() -> None:
    client = _client()
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="(320, 180)"))],
        usage=SimpleNamespace(total_tokens=15),
    )
    coords = client.locate("frame-b64", "Submit button", _DISPLAY)
    assert coords == (320.0, 180.0)
    assert client.tokens_used == 15
