"""Fuse accessibility nodes and visual detections into grounded UI candidates."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from cua_agent.orchestrator.react_types import GroundedNode


ROLE_ALIASES = {
    "axbutton": "button",
    "button": "button",
    "controltype.button": "button",
    "axtextfield": "textbox",
    "axtextarea": "textbox",
    "edit": "textbox",
    "controltype.edit": "textbox",
    "textbox": "textbox",
    "searchbox": "textbox",
    "axlink": "link",
    "hyperlink": "link",
    "controltype.hyperlink": "link",
    "axcheckbox": "checkbox",
    "checkbox": "checkbox",
    "controltype.checkbox": "checkbox",
    "axcombobox": "combobox",
    "combobox": "combobox",
    "controltype.combobox": "combobox",
    "axmenuitem": "menuitem",
    "menuitem": "menuitem",
    "text": "static_text",
    "statictext": "static_text",
    "axstatictext": "static_text",
    "controltype.text": "static_text",
}


class GroundingFusion:
    """Create stable grounded candidates from semantic and visual sources."""

    def fuse(
        self,
        semantic_nodes: Iterable[GroundedNode | dict[str, Any]],
        visual_nodes: Iterable[GroundedNode | dict[str, Any]],
    ) -> list[GroundedNode]:
        semantic = [self.coerce_node(node, source="semantic", index=i) for i, node in enumerate(semantic_nodes)]
        visual = [self.coerce_node(node, source="visual", index=i) for i, node in enumerate(visual_nodes)]
        semantic = [node for node in semantic if self._valid_frame(node.frame)]
        visual = [node for node in visual if self._valid_frame(node.frame)]

        fused: list[GroundedNode] = []
        used_visual: set[int] = set()

        for sem in semantic:
            best_idx = -1
            best_score = 0.0
            best_iou = 0.0
            for idx, vis in enumerate(visual):
                if idx in used_visual:
                    continue
                iou = self.iou(sem.frame, vis.frame)
                text_score = self._text_similarity(sem.label, vis.label)
                role_score = 1.0 if sem.role and sem.role == vis.role else 0.0
                score = (iou * 0.65) + (text_score * 0.25) + (role_score * 0.10)
                if score > best_score:
                    best_idx = idx
                    best_score = score
                    best_iou = iou

            if best_idx >= 0 and (best_iou > 0.65 or best_score > 0.72):
                vis = visual[best_idx]
                used_visual.add(best_idx)
                frame = sem.frame if self._frame_quality(sem.frame) >= self._frame_quality(vis.frame) else vis.frame
                label = sem.label or vis.label
                role = sem.role or vis.role
                confidence = min(1.0, max(sem.confidence, vis.confidence, 0.75) + best_score * 0.2)
                fused.append(
                    GroundedNode(
                        gid=self._gid("fused", role, label, len(fused)),
                        role=role,
                        label=label,
                        path=sem.path or vis.path,
                        frame=dict(frame),
                        source="fused",
                        confidence=confidence,
                        metadata={
                            "semantic_gid": sem.gid,
                            "visual_gid": vis.gid,
                            "iou": best_iou,
                            "match_score": best_score,
                        },
                    )
                )
            else:
                sem.gid = sem.gid or self._gid("sem", sem.role, sem.label, len(fused))
                fused.append(sem)

        for idx, vis in enumerate(visual):
            if idx in used_visual:
                continue
            vis.gid = vis.gid or self._gid("vis", vis.role, vis.label, len(fused))
            fused.append(vis)

        fused.sort(key=lambda node: (node.confidence, self._frame_area(node.frame)), reverse=True)
        return fused

    def coerce_node(self, node: GroundedNode | dict[str, Any], *, source: str, index: int) -> GroundedNode:
        if isinstance(node, GroundedNode):
            node.role = self.normalize_role(node.role)
            node.frame = self._clean_frame(node.frame)
            if not node.gid:
                node.gid = self._gid("sem" if source == "semantic" else "vis", node.role, node.label, index)
            return node

        frame = self._clean_frame(node.get("frame") or {})
        role = self.normalize_role(str(node.get("role") or node.get("type") or ""))
        label = str(node.get("label") or node.get("title") or node.get("name") or node.get("value") or "").strip()
        path = str(node.get("path") or "").strip()
        confidence = self._as_float(node.get("confidence"), 0.65 if source == "semantic" else 0.45)
        prefix = "sem" if source == "semantic" else "vis"
        gid = str(node.get("gid") or self._gid(prefix, role, label or path, index))
        return GroundedNode(
            gid=gid,
            role=role,
            label=label,
            path=path,
            frame=frame,
            source="semantic" if source == "semantic" else "visual",
            confidence=confidence,
            metadata={k: v for k, v in node.items() if k not in {"frame", "role", "label", "title", "path"}},
        )

    @classmethod
    def normalize_role(cls, role: str) -> str:
        token = str(role or "").strip().lower()
        token = token.replace(" ", "").replace("_", "")
        return ROLE_ALIASES.get(token, token or "element")

    @staticmethod
    def iou(a: dict[str, float], b: dict[str, float]) -> float:
        ax0, ay0 = float(a.get("x", 0)), float(a.get("y", 0))
        bx0, by0 = float(b.get("x", 0)), float(b.get("y", 0))
        ax1, ay1 = ax0 + float(a.get("w", 0)), ay0 + float(a.get("h", 0))
        bx1, by1 = bx0 + float(b.get("w", 0)), by0 + float(b.get("h", 0))
        inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
        inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _gid(self, prefix: str, role: str, label: str, index: int) -> str:
        role_token = self._slug(role or "element")[:24] or "element"
        label_token = self._slug(label or "unlabelled")[:32] or "unlabelled"
        digest = hashlib.sha1(f"{role}|{label}|{index}".encode("utf-8")).hexdigest()[:6]
        return f"{prefix}:{role_token}:{label_token}:{index + 1}:{digest}"

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

    @staticmethod
    def _clean_frame(frame: dict[str, Any]) -> dict[str, float]:
        def _f(name: str) -> float:
            try:
                return float(frame.get(name, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        return {"x": _f("x"), "y": _f("y"), "w": _f("w"), "h": _f("h")}

    @staticmethod
    def _valid_frame(frame: dict[str, float]) -> bool:
        return float(frame.get("w", 0)) > 0 and float(frame.get("h", 0)) > 0

    @staticmethod
    def _frame_area(frame: dict[str, float]) -> float:
        return max(0.0, float(frame.get("w", 0))) * max(0.0, float(frame.get("h", 0)))

    def _frame_quality(self, frame: dict[str, float]) -> float:
        area = self._frame_area(frame)
        if area <= 0:
            return 0.0
        return min(1.0, area / 8000.0)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_norm = re.sub(r"\s+", " ", str(a or "").strip().lower())
        b_norm = re.sub(r"\s+", " ", str(b or "").strip().lower())
        if not a_norm or not b_norm:
            return 0.0
        if a_norm == b_norm:
            return 1.0
        if a_norm in b_norm or b_norm in a_norm:
            return 0.82
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
