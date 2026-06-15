"""Typed ReAct loop records shared by the orchestrator services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SensorName = Literal["none", "os_telemetry", "a11y_tree", "pixel_diff", "vision_full"]
GroundingSource = Literal["semantic", "visual", "fused"]


@dataclass
class GroundedNode:
    gid: str
    role: str
    label: str
    path: str
    frame: dict[str, float]
    source: GroundingSource
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingBundle:
    screenshot_b64: str
    frame_hash: str
    semantic_nodes: list[GroundedNode] = field(default_factory=list)
    visual_nodes: list[GroundedNode] = field(default_factory=list)
    fused_nodes: list[GroundedNode] = field(default_factory=list)
    som_tags: list[dict[str, Any]] = field(default_factory=list)
    active_window_title: str = ""
    active_app: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    overlay_b64: str = ""
    ax_tree: dict[str, Any] | None = None

    def prompt_nodes(self, limit: int = 50) -> list[dict[str, Any]]:
        nodes = self.fused_nodes or self.semantic_nodes or self.visual_nodes
        return [node.to_dict() for node in nodes[:limit]]

    def to_compact_dict(self, *, include_image: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_hash": self.frame_hash,
            "semantic_count": len(self.semantic_nodes),
            "visual_count": len(self.visual_nodes),
            "fused_count": len(self.fused_nodes),
            "quality": dict(self.quality),
            "active_window_title": self.active_window_title,
            "active_app": self.active_app,
            "candidates": self.prompt_nodes(),
        }
        if include_image:
            payload["screenshot_b64"] = self.screenshot_b64
            payload["overlay_b64"] = self.overlay_b64
        return payload


@dataclass
class ActionEnvelope:
    observation_summary: str = ""
    state_assessment: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] | None = None
    fallback_if_failed: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_fresh_grounding: bool = False
    raw_response_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationOutcome:
    passed: bool
    reason: str
    sensor: str
    changed: bool = False
    next_frame: str = ""
    next_hash: str = ""
    hash_distance: int = 0
    ssim_score: float | None = None
    ax_tree_after: dict[str, Any] | None = None
    ax_changed: bool = False
    note: str = ""
    force_vision_next_turn: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryDecision:
    stop: bool = False
    advance_step: bool = False
    replan: bool = False
    refresh_grounding: bool = False
    force_vision_next_turn: bool = False
    reason: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReActTurn:
    turn_id: int
    step_id: int | None
    observation_summary: str
    grounding_quality: dict[str, Any]
    selected_target_gid: str | None
    action: dict[str, Any] | None
    verification: dict[str, Any] | None
    result: dict[str, Any] | None
    reflection: dict[str, Any] | None
    recovery_decision: dict[str, Any] | None
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
