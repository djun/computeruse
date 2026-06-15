import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.config import Settings
from cua_agent.utils.image_mime import image_data_uri, image_mime_from_base64


PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode("ascii")


class _DummyComputer:
    platform_name = "macOS"
    system_info = "test"
    display = DisplayInfo(
        logical_width=1280,
        logical_height=720,
        physical_width=1280,
        physical_height=720,
        scale_factor=1.0,
    )


def test_image_data_uri_uses_sniffed_png_over_jpeg_fallback() -> None:
    assert image_mime_from_base64(PNG_B64, fallback="image/jpeg") == "image/png"
    assert image_data_uri(PNG_B64, fallback="image/jpeg").startswith("data:image/png;base64,")


def test_cognitive_core_sends_sniffed_png_mime_for_overlay_even_when_encode_format_is_jpeg() -> None:
    core = CognitiveCore(Settings(use_openrouter=False, encode_format="JPEG"), _DummyComputer())
    core.client = MagicMock()
    core.client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    core._call_openrouter(
        PNG_B64,
        history=[],
        include_visual_context=True,
        user_prompt="pesquise efeito ferranti",
        repeat_info=None,
        plan=None,
        current_step=None,
        loop_state=None,
        ax_tree=None,
        som_tags=[],
        relevant_skills=[],
    )

    messages = core.client.chat.completions.create.call_args.kwargs["messages"]
    image_url = messages[1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
