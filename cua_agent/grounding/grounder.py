"""Grounding service that observes the desktop and builds fused UI candidates."""

from __future__ import annotations

from typing import Any

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.grounding.fusion import GroundingFusion
from cua_agent.orchestrator.react_types import GroundedNode, GroundingBundle
from cua_agent.utils.ax_pruning import prune_ax_tree_for_prompt
from cua_agent.utils.ax_utils import draw_som_overlay, flatten_nodes_with_frames
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class Grounder:
    """Collect semantic and visual candidates and expose fused grounding state."""

    def __init__(self, settings: Settings, computer: ComputerAdapter) -> None:
        self.settings = settings
        self.computer = computer
        self.display = computer.display
        self.fusion = GroundingFusion()
        self.logger = get_logger(__name__, level=settings.log_level)

    def observe(
        self,
        *,
        previous: GroundingBundle | None = None,
        force_vision: bool = False,
        include_semantic: bool = True,
        include_visual: bool = True,
        max_nodes: int = 80,
    ) -> GroundingBundle:
        frame, frame_hash = self.computer.capture_with_hash()
        ax_tree: dict[str, Any] | None = None
        semantic_nodes: list[GroundedNode] = []
        visual_nodes: list[GroundedNode] = []

        if include_semantic and self.settings.enable_semantic:
            ax_res = self.computer.get_active_window_tree(max_depth=4)
            if ax_res.success:
                raw_tree = (ax_res.metadata or {}).get("tree")
                ax_tree = prune_ax_tree_for_prompt(raw_tree) if raw_tree else None
                if ax_tree:
                    semantic_nodes = self._semantic_nodes(ax_tree, max_nodes=max_nodes)
            else:
                self.logger.debug("semantic grounding unavailable: %s", ax_res.reason)

        should_detect_visual = include_visual or force_vision or not semantic_nodes
        if should_detect_visual:
            try:
                visual_raw = self.computer.detect_ui_elements(frame)
                visual_nodes = self._visual_nodes(visual_raw, max_nodes=max_nodes)
            except Exception as exc:
                self.logger.debug("visual grounding failed: %s", exc)

        fused_nodes = self.fusion.fuse(semantic_nodes, visual_nodes)
        prompt_tree = ax_tree or self._visual_tree(visual_nodes)
        overlay_nodes = self._overlay_nodes(fused_nodes or semantic_nodes or visual_nodes, max_nodes=40)
        overlay_b64, som_tags = draw_som_overlay(frame, overlay_nodes, self.display) if overlay_nodes else (frame, [])

        previous_hash = previous.frame_hash if previous else None
        avg_confidence = (
            sum(node.confidence for node in fused_nodes) / len(fused_nodes)
            if fused_nodes
            else 0.0
        )
        quality = {
            "has_semantic": bool(semantic_nodes),
            "has_visual": bool(visual_nodes),
            "semantic_count": len(semantic_nodes),
            "visual_count": len(visual_nodes),
            "fused_count": len(fused_nodes),
            "avg_confidence": round(avg_confidence, 4),
            "stale_hash": bool(previous_hash and previous_hash == frame_hash),
            "force_vision": bool(force_vision),
        }

        return GroundingBundle(
            screenshot_b64=frame,
            frame_hash=frame_hash,
            semantic_nodes=semantic_nodes,
            visual_nodes=visual_nodes,
            fused_nodes=fused_nodes,
            som_tags=som_tags,
            quality=quality,
            overlay_b64=overlay_b64,
            ax_tree=prompt_tree,
        )

    def _semantic_nodes(self, tree: dict[str, Any], *, max_nodes: int) -> list[GroundedNode]:
        flattened = flatten_nodes_with_frames(tree, max_nodes=max_nodes)
        nodes: list[GroundedNode] = []
        for idx, raw in enumerate(flattened):
            coerced = self.fusion.coerce_node(raw, source="semantic", index=idx)
            coerced.confidence = max(coerced.confidence, 0.7)
            nodes.append(coerced)
        return nodes

    def _visual_nodes(self, nodes: list[dict[str, Any]], *, max_nodes: int) -> list[GroundedNode]:
        out: list[GroundedNode] = []
        for idx, raw in enumerate((nodes or [])[:max_nodes]):
            coerced = self.fusion.coerce_node(raw, source="visual", index=idx)
            out.append(coerced)
        return out

    def _visual_tree(self, nodes: list[GroundedNode]) -> dict[str, Any] | None:
        if not nodes:
            return None
        return {
            "role": "AXWindow",
            "title": "Visual Fallback",
            "frame": {
                "x": 0,
                "y": 0,
                "w": self.display.logical_width,
                "h": self.display.logical_height,
            },
            "children": [
                {
                    "role": node.role,
                    "title": node.label,
                    "frame": dict(node.frame),
                    "path": node.path,
                    "source": node.source,
                    "confidence": node.confidence,
                    "gid": node.gid,
                }
                for node in nodes
            ],
        }

    def _overlay_nodes(self, nodes: list[GroundedNode], *, max_nodes: int) -> list[dict[str, Any]]:
        overlay: list[dict[str, Any]] = []
        for idx, node in enumerate(nodes[:max_nodes], start=1):
            overlay.append(
                {
                    "id": idx,
                    "gid": node.gid,
                    "role": node.role,
                    "label": node.label,
                    "path": node.path,
                    "frame": dict(node.frame),
                    "source": node.source,
                    "confidence": node.confidence,
                }
            )
        return overlay
