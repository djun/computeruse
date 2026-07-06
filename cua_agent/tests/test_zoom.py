import base64
import io

from unittest.mock import MagicMock, patch

from PIL import Image

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import COMPUTER_ACTION_SPACE, DisplayInfo
from cua_agent.utils.config import Settings

_DISPLAY = DisplayInfo(
    logical_width=200,
    logical_height=100,
    physical_width=200,
    physical_height=100,
    scale_factor=1.0,
)


class _DummyComputer:
    platform_name = "test"
    system_info = "test"
    display = _DISPLAY


def _b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _pipeline():
    from macos_cua_agent.drivers.vision_pipeline import VisionPipeline

    with patch("macos_cua_agent.drivers.vision_pipeline.get_display_info", return_value=_DISPLAY):
        return VisionPipeline(Settings(encode_format="PNG"))


def test_normalize_box_clamps_and_orders() -> None:
    from cua_agent.computer.vision_pipeline_base import SharedVisionPipeline

    assert SharedVisionPipeline._normalize_box([50, 25, 150, 75], 200, 100) == (50, 25, 150, 75)
    # Reversed + out of bounds -> ordered and clamped.
    assert SharedVisionPipeline._normalize_box([300, -10, 40, 200], 200, 100) == (40, 0, 200, 100)
    # Dict form.
    assert SharedVisionPipeline._normalize_box({"x": 10, "y": 20, "w": 30, "h": 40}, 200, 100) == (10, 20, 40, 60)


def test_crop_region_upscales() -> None:
    pipe = _pipeline()
    src = _b64(Image.new("RGB", (200, 100), (255, 0, 0)))
    out = pipe.crop_region(src, [50, 25, 150, 75], upscale=2.0)
    # 100x50 region upscaled 2x -> 200x100.
    assert _decode(out).size == (200, 100)


def test_capture_zoom_crops_grabbed_frame() -> None:
    pipe = _pipeline()
    with patch.object(pipe, "_grab_frame", return_value=Image.new("RGB", (200, 100), (0, 128, 0))):
        out = pipe.capture_zoom([0, 0, 100, 100], upscale=1.0)
    assert _decode(out).size == (100, 100)


def test_zoom_in_action_space() -> None:
    assert "zoom" in COMPUTER_ACTION_SPACE


def test_cognitive_core_maps_zoom_action() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    mapped = core._map_single_computer_action({"action": "zoom", "region": [10, 20, 110, 90]})
    assert mapped["type"] == "zoom"
    assert mapped["region"] == [10.0, 20.0, 110.0, 90.0]


def test_cognitive_core_zoom_requires_region() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    mapped = core._map_single_computer_action({"action": "zoom"})
    assert mapped["type"] == "invalid_action"
