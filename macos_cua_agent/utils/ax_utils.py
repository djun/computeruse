from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from macos_cua_agent.utils.coordinates import clamp_point, point_to_px
from macos_cua_agent.utils.macos_integration import DisplayInfo


def flatten_nodes_with_frames(tree: Dict[str, Any], max_nodes: int = 40) -> List[Dict[str, Any]]:
    """
    Flatten an accessibility tree into a list of nodes that have usable frames.
    Each item: {"id": int, "frame": {"x","y","w","h"}, "role": str, "label": str}
    """
    nodes: List[Dict[str, Any]] = []

    def _label_for(node: Dict[str, Any]) -> str:
        title = (node.get("title") or "").strip()
        value = (node.get("value") or "").strip()
        role = (node.get("role") or "").strip()
        if title:
            return title
        if value:
            return value
        return role or "element"

    def _walk(node: Dict[str, Any]) -> None:
        if len(nodes) >= max_nodes:
            return
        frame = node.get("frame") or {}
        if frame and frame.get("w", 0) > 0 and frame.get("h", 0) > 0:
            nodes.append(
                {
                    "frame": {
                        "x": float(frame.get("x", 0)),
                        "y": float(frame.get("y", 0)),
                        "w": float(frame.get("w", 0)),
                        "h": float(frame.get("h", 0)),
                    },
                    "role": (node.get("role") or "").strip(),
                    "label": _label_for(node),
                }
            )
        for child in node.get("children") or []:
            if len(nodes) >= max_nodes:
                return
            _walk(child)

    if tree:
        _walk(tree)

    # Attach stable IDs after traversal
    for idx, n in enumerate(nodes, start=1):
        n["id"] = idx
    return nodes


def draw_som_overlay(
    base_image_b64: str, nodes: List[Dict[str, Any]], display: DisplayInfo
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Draw bounding boxes for nodes onto the screenshot.
    Returns (overlay_base64, manifest) where manifest contains id, frame (logical), role, label.
    """
    if not nodes:
        return base_image_b64, []

    try:
        img_bytes = base64.b64decode(base_image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        # If decode fails, return original frame without overlay
        return base_image_b64, []

    draw = ImageDraw.Draw(image)
    palette = [
        (255, 99, 71),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
    ]
    manifest: List[Dict[str, Any]] = []

    # Derive scale factor for drawing (image may already be logical-sized)
    scale = 1.0
    if display.logical_width and image.width:
        scale = image.width / float(display.logical_width)

    for node in nodes:
        frame = node["frame"]
        lx, ly = frame["x"], frame["y"]
        lw, lh = frame["w"], frame["h"]
        # Convert logical to image pixel coords, clamp to bounds
        px0, py0 = point_to_px(lx, ly, scale)
        px1, py1 = point_to_px(lx + lw, ly + lh, scale)
        px0, py0 = clamp_point(px0, py0, image.width, image.height)
        px1, py1 = clamp_point(px1, py1, image.width, image.height)

        color = palette[(node["id"] - 1) % len(palette)]
        draw.rectangle([px0, py0, px1, py1], outline=color, width=2)
        label = f"#{node['id']}"
        text_bg = [px0, py0 - 14, px0 + 20, py0]
        draw.rectangle(text_bg, fill=color)
        draw.text((px0 + 2, py0 - 13), label, fill=(0, 0, 0))

        manifest.append(
            {
                "id": node["id"],
                "role": node.get("role", ""),
                "label": node.get("label", ""),
                "frame": frame,
            }
        )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return overlay_b64, manifest
