from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import platform
import re

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger

# Attempt imports for Accessibility API
try:
    import ApplicationServices
    from ApplicationServices import (
        AXUIElementCreateSystemWide,
        AXUIElementCopyAttributeValue,
        AXUIElementCopyAttributeNames,
        AXUIElementCopyElementAtPosition,
        AXUIElementPerformAction,
        AXUIElementCopyActionNames,
        kAXFocusedApplicationAttribute,
        kAXFocusedUIElementAttribute,
        kAXFocusedWindowAttribute,
        kAXChildrenAttribute,
        kAXParentAttribute,
        kAXRoleAttribute,
        kAXTitleAttribute,
        kAXValueAttribute,
        kAXPositionAttribute,
        kAXSizeAttribute,
        kAXSubroleAttribute,
        kAXRoleDescriptionAttribute,
        AXUIElementSetAttributeValue,
        AXValueGetType,
        AXValueGetValue,
        kAXValueCGPointType,
        kAXValueCGSizeType,
    )
    from Quartz import CGPoint, CGSize
    from CoreFoundation import CFArrayGetCount, CFArrayGetValueAtIndex
    HAS_AX = True
except ImportError:
    HAS_AX = False

_AX_POINT_RE = re.compile(r"x:(-?\d+(?:\.\d+)?)\s+y:(-?\d+(?:\.\d+)?)")
_AX_SIZE_RE = re.compile(r"w:(-?\d+(?:\.\d+)?)\s+h:(-?\d+(?:\.\d+)?)")


class AccessibilityDriver:
    """
    Provides semantic access to the macOS UI via the Accessibility API (AXUIElement).
    Allows the agent to 'see' the structure of windows and controls, not just pixels.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        if not HAS_AX:
            self.logger.warning("pyobjc ApplicationServices not found; AccessibilityDriver disabled.")

    def set_focused_element_value(self, value: str) -> ActionResult:
        """
        Sets the value of the currently focused element (if it supports kAXValueAttribute).
        Useful when we know focus is correct but don't want to move the mouse.
        """
        if not HAS_AX:
            return ActionResult(success=False, reason="Accessibility API unavailable")

        try:
            system_wide = AXUIElementCreateSystemWide()
            err, element = AXUIElementCopyAttributeValue(system_wide, kAXFocusedUIElementAttribute, None)
            if err != 0 or not element:
                return ActionResult(success=False, reason="No focused element to set")

            res = self._set_value_on_element_or_parents(element, value, max_ancestors=3)
            if res:
                return res
            return ActionResult(success=False, reason="Focused element does not accept text")
        except Exception as exc:
            return ActionResult(success=False, reason=f"Focused set failed: {exc}")

    def set_text_element_value(self, x: float, y: float, value: str) -> ActionResult:
        """
        Sets the value of an editable text element at (x, y) using AXValue.
        Phantom mode for typing.
        """
        if not HAS_AX:
            return ActionResult(success=False, reason="Accessibility API unavailable")

        try:
            system_wide = AXUIElementCreateSystemWide()
            err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
            if err != 0 or not element:
                return ActionResult(success=False, reason=f"No element found at ({x}, {y})")

            res = self._set_value_on_element_or_parents(element, value, max_ancestors=3)
            if res:
                return res
            return ActionResult(success=False, reason="Failed to set AXValue on target or parents")
        except Exception as e:
            return ActionResult(success=False, reason=f"Phantom type failed: {e}")

    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult:
        """
        Captures the UI tree of the currently focused window.
        Returns a JSON-serializable dict structure.
        """
        if not HAS_AX:
            return ActionResult(success=False, reason="Accessibility API unavailable (missing pyobjc)")

        try:
            system_wide = AXUIElementCreateSystemWide()
            
            # Get focused app
            err, app = AXUIElementCopyAttributeValue(system_wide, kAXFocusedApplicationAttribute, None)
            if err != 0:
                return ActionResult(success=False, reason="No focused application found")

            # Get focused window
            err, window = AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute, None)
            if err != 0:
                return ActionResult(success=False, reason="No focused window found")

            # Build tree
            # print("Start window print")
            # self._debug_dump_ax_element(window)
            # print("End window print")
            tree = self._build_tree(window, depth=0, max_depth=max_depth)
            if not self._tree_has_frame(tree):
                reason = (
                    "Accessibility tree returned without usable frames (w/h <= 0). "
                    "This can be caused by missing Accessibility permissions or failing to decode AXPosition/AXSize; "
                    "see log for captured raw frames."
                )
                frames = self._collect_frames_for_logging(tree)
                frames_to_log = frames[:20]
                self.logger.error(
                    "%s Frames captured (count=%d, showing up to %d): %s",
                    reason,
                    len(frames),
                    len(frames_to_log),
                    frames_to_log if frames_to_log else "none",
                )
                return ActionResult(success=False, reason=reason)
            return ActionResult(success=True, reason="captured", metadata={"tree": tree})

        except Exception as e:
            self.logger.exception("Failed to capture accessibility tree")
            return ActionResult(success=False, reason=f"AX error: {str(e)}")

    def get_focused_app_name(self) -> str | None:
        """Return the focused application's title (best-effort)."""
        if not HAS_AX:
            return None
        try:
            system_wide = AXUIElementCreateSystemWide()
            err, app = AXUIElementCopyAttributeValue(system_wide, kAXFocusedApplicationAttribute, None)
            if err != 0 or not app:
                return None
            title = self._get_attr(app, kAXTitleAttribute)
            if title:
                return str(title)
        except Exception:
            return None
        return None

    def probe_element(self, x: float, y: float, radius: float = 0.0) -> ActionResult:
        """
        Returns the AX structure of the element at the specific coordinates.
        Useful for targeted inspection (probe_ui).
        """
        if not HAS_AX:
            return ActionResult(success=False, reason="Accessibility API unavailable")

        try:
            center_tree = self._probe_single(x, y)
            if not center_tree:
                return ActionResult(success=False, reason=f"No element found at ({x}, {y})")

            neighbors: List[Dict[str, Any]] = []
            if radius and radius > 0:
                # Sample a small cross around the target to disambiguate tightly packed elements.
                offsets = [(-radius, 0), (radius, 0), (0, -radius), (0, radius)]
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    tree = self._probe_single(nx, ny)
                    if tree:
                        neighbors.append({"x": nx, "y": ny, "tree": tree})

            metadata = {"tree": center_tree}
            if neighbors:
                metadata["neighbors"] = neighbors
            return ActionResult(success=True, reason="probed", metadata=metadata)
        except Exception as e:
            self.logger.exception("Probe UI failed")
            return ActionResult(success=False, reason=str(e))

    def _probe_single(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        system_wide = AXUIElementCreateSystemWide()
        err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
        if err != 0 or element is None:
            return None
        return self._build_tree(element, depth=0, max_depth=2)

    def perform_action_at(self, x: float, y: float, action: str = "AXPress") -> ActionResult:
        """
        Attempts to perform an AX action (default AXPress) on the element at (x, y).
        This is the 'Phantom Mode' implementation.
        """
        if not HAS_AX:
             return ActionResult(success=False, reason="Accessibility API unavailable")

        try:
            system_wide = AXUIElementCreateSystemWide()
            err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
            if err != 0 or not element:
                return ActionResult(success=False, reason=f"No element found at ({x}, {y}) for phantom action")

            # Walk up parent chain to find an element that supports the action
            # (e.g. clicking a label inside a button -> walk up to button)
            curr = element
            attempts = 0
            while curr and attempts < 5:
                err, names = AXUIElementCopyActionNames(curr, None)
                if err == 0 and names and action in names:
                    # Found it!
                    err = AXUIElementPerformAction(curr, action)
                    if err == 0:
                        return ActionResult(success=True, reason=f"Phantom {action} executed on {self._get_role(curr)}")
                    return ActionResult(success=False, reason=f"Failed to execute {action} (err={err})")
                
                # Try parent
                err, parent = AXUIElementCopyAttributeValue(curr, kAXParentAttribute, None)
                if err == 0 and parent:
                    curr = parent
                    attempts += 1
                else:
                    break

            return ActionResult(success=False, reason=f"Action {action} not supported by element at ({x},{y}) or its ancestors")

        except Exception as e:
            return ActionResult(success=False, reason=f"Phantom action failed: {e}")

    def _get_role(self, element: Any) -> str:
        return self._get_attr(element, kAXRoleAttribute) or "Unknown"

    def _build_tree(self, element: Any, depth: int, max_depth: int) -> Dict[str, Any]:
        if depth > max_depth:
            return {"role": "max_depth_reached"}

        node = {}
        
        # Basic attributes
        node["role"] = self._get_attr(element, kAXRoleAttribute)
        node["title"] = self._get_attr(element, kAXTitleAttribute)
        node["value"] = self._get_attr(element, kAXValueAttribute)
        
        # Construct frame from position and size
        pos = self._get_attr(element, kAXPositionAttribute)
        size = self._get_attr(element, kAXSizeAttribute)
        
        frame = None
        raw_frame = None
        if pos is not None and size is not None:
            x, y = self._decode_point(pos)
            w, h = self._decode_size(size)

            raw_frame = {'x': x, 'y': y, 'w': w, 'h': h}
            # Validation: Ensure reasonable values (e.g., not 0x0 unless it's truly hidden/empty)
            # But some elements might be 0x0. However, for the agent, 0x0 is useless.
            if w > 0 and h > 0:
                frame = raw_frame
            else:
                self.logger.debug(
                    "Parsed unusable frame (%s) from pos=%r size=%r for role=%s title=%s",
                    raw_frame,
                    pos,
                    size,
                    node.get("role"),
                    node.get("title"),
                )
        
        node["frame"] = frame
        if raw_frame and frame is None:
            node["raw_frame"] = raw_frame
        
        # Pruning: Skip uninteresting elements if they have no useful content
        # (This is a heuristic to save tokens)
        if not node["title"] and not node["value"] and node["role"] in ["AXGroup", "AXUnknown"]:
             pass # We still check children
        
        # Recursion
        children = []
        err, ax_children = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute, None)
        
        if err == 0 and ax_children:
            # ax_children is a CFArray or list depending on bridge
            # Python bridge usually converts CFArray to list
            if isinstance(ax_children, (list, tuple)):
                for child in ax_children:
                    child_node = self._build_tree(child, depth + 1, max_depth)
                    if self._is_useful_node(child_node):
                        children.append(child_node)
            else:
                 # Handle raw CFArray if needed (rare with modern pyobjc)
                 pass
        
        if children:
            node["children"] = children
        elif not self._is_useful_node(node):
            # If it's a leaf node and useless, we might discard it, but 
            # let's be careful. For now, we return it if it has a frame.
            pass

        return node

    def _decode_point(self, value: Any) -> Tuple[float, float]:
        """Decode AX position into a numeric (x, y). Handles AXValue, structs, tuples, and repr fallback."""
        try:
            if value is None:
                return 0.0, 0.0

            # Tuple/list or CGPoint-like object
            if hasattr(value, "x") and hasattr(value, "y"):
                try:
                    return float(value.x), float(value.y)
                except Exception:
                    pass
            if hasattr(value, "__getitem__"):
                try:
                    return float(value[0]), float(value[1])
                except Exception:
                    pass

            # AXValue wrapper
            try:
                ax_type = AXValueGetType(value)
            except Exception:
                ax_type = None
            if ax_type == kAXValueCGPointType:
                pt = CGPoint()
                try:
                    if AXValueGetValue(value, ax_type, pt):
                        return float(pt.x), float(pt.y)
                except Exception as exc:
                    self.logger.debug("AXValue point decode failed: %s", exc)

            # Last-resort repr parse (handles plain AXValue repr from some PyObjC versions)
            match = _AX_POINT_RE.search(repr(value))
            if match:
                return float(match.group(1)), float(match.group(2))
        except Exception as exc:
            self.logger.debug("Failed to decode point %r: %s", value, exc)
        return 0.0, 0.0

    def _decode_size(self, value: Any) -> Tuple[float, float]:
        """Decode AX size into numeric (w, h). Handles AXValue, structs, tuples, and repr fallback."""
        try:
            if value is None:
                return 0.0, 0.0

            if hasattr(value, "width") and hasattr(value, "height"):
                try:
                    return float(value.width), float(value.height)
                except Exception:
                    pass
            if hasattr(value, "__getitem__"):
                try:
                    return float(value[0]), float(value[1])
                except Exception:
                    pass

            try:
                ax_type = AXValueGetType(value)
            except Exception:
                ax_type = None
            if ax_type == kAXValueCGSizeType:
                sz = CGSize()
                try:
                    if AXValueGetValue(value, ax_type, sz):
                        return float(sz.width), float(sz.height)
                except Exception as exc:
                    self.logger.debug("AXValue size decode failed: %s", exc)

            match = _AX_SIZE_RE.search(repr(value))
            if match:
                return float(match.group(1)), float(match.group(2))
        except Exception as exc:
            self.logger.debug("Failed to decode size %r: %s", value, exc)
        return 0.0, 0.0

    def _get_attr(self, element: Any, attr_name: str) -> Any:
        try:
            err, value = AXUIElementCopyAttributeValue(element, attr_name, None)
            if err == 0:
                # Convert complex types if necessary
                return value
        except Exception:
            pass
        return None

    def _debug_dump_ax_element(self, element: Any) -> None:
        """Print available AX attributes for a raw AXUIElement (debug helper)."""
        try:
            err, attr_names = AXUIElementCopyAttributeNames(element, None)
            if err != 0 or not attr_names:
                print(f"  <failed to fetch attribute names: err={err}>")
                return

            for name in attr_names:
                err, value = AXUIElementCopyAttributeValue(element, name, None)
                if name == kAXChildrenAttribute and err == 0:
                    # Avoid dumping the full child tree; just show the count.
                    summary = f"{len(value)} children" if isinstance(value, (list, tuple)) else str(value)
                    print(f"  {name}: {summary}")
                    continue

                display = value if err == 0 else f"<err {err}>"
                print(f"  {name}: {display}")
        except Exception as exc:
            print(f"  <debug dump failed: {exc}>")

    def _set_value_on_element_or_parents(self, element: Any, value: str, max_ancestors: int = 3) -> ActionResult | None:
        """Try setting kAXValueAttribute on the element, walking up to parents."""
        try:
            err = AXUIElementSetAttributeValue(element, kAXValueAttribute, value)
            if err == 0:
                return ActionResult(success=True, reason="set AXValue successfully")
            curr = element
            attempts = 0
            while curr and attempts < max_ancestors:
                err, parent = AXUIElementCopyAttributeValue(curr, kAXParentAttribute, None)
                if err == 0 and parent:
                    err = AXUIElementSetAttributeValue(parent, kAXValueAttribute, value)
                    if err == 0:
                        return ActionResult(success=True, reason="set AXValue on parent successfully")
                    curr = parent
                    attempts += 1
                else:
                    break
        except Exception:
            return None
        return None

    def _is_useful_node(self, node: Dict[str, Any]) -> bool:
        """Heuristic to determine if a node is worth showing to the LLM."""
        # If it has children, it's useful (structure)
        if "children" in node and node["children"]:
            return True
        # If it has text or value
        if node.get("title") or node.get("value"):
            return True
        # If it is a control
        if node.get("role") in ["AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox"]:
            return True
        return False

    def _tree_has_frame(self, node: Dict[str, Any]) -> bool:
        """Check if any node in the tree has a usable frame."""
        if not node:
            return False
        frame = node.get("frame") or {}
        if frame and frame.get("w", 0) > 0 and frame.get("h", 0) > 0:
            return True
        for child in node.get("children") or []:
            if self._tree_has_frame(child):
                return True
        return False

    def _collect_frames_for_logging(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect frames (including unusable ones) for debug logging."""
        frames: List[Dict[str, Any]] = []
        if not node:
            return frames

        frame = node.get("frame")
        raw_frame = node.get("raw_frame")
        if frame is not None:
            frames.append(
                {"role": node.get("role"), "title": node.get("title"), "frame": frame, "usable": True}
            )
        elif raw_frame is not None:
            frames.append(
                {"role": node.get("role"), "title": node.get("title"), "frame": raw_frame, "usable": False}
            )

        for child in node.get("children") or []:
            frames.extend(self._collect_frames_for_logging(child))
        return frames
