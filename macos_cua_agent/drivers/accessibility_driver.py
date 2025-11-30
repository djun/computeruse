from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import platform

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger

# Attempt imports for Accessibility API
try:
    import ApplicationServices
    from ApplicationServices import (
        AXUIElementCreateSystemWide,
        AXUIElementCopyAttributeValue,
        AXUIElementCopyElementAtPosition,
        AXUIElementPerformAction,
        AXUIElementCopyActionNames,
        kAXFocusedApplicationAttribute,
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
    )
    from CoreFoundation import CFArrayGetCount, CFArrayGetValueAtIndex
    HAS_AX = True
except ImportError:
    HAS_AX = False


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
            tree = self._build_tree(window, depth=0, max_depth=max_depth)
            if not self._tree_has_frame(tree):
                reason = (
                    "Accessibility tree returned without usable frames (w/h <= 0). "
                    "Grant Accessibility permission to this app in System Settings > Privacy & Security > Accessibility, "
                    "then restart and retry."
                )
                self.logger.error(reason)
                return ActionResult(success=False, reason=reason)
            return ActionResult(success=True, reason="captured", metadata={"tree": tree})

        except Exception as e:
            self.logger.exception("Failed to capture accessibility tree")
            return ActionResult(success=False, reason=f"AX error: {str(e)}")

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
        if pos is not None and size is not None:
            try:
                # Handle pyobjc struct wrappers (CGPoint/CGSize) or tuples
                x = getattr(pos, 'x', pos[0] if hasattr(pos, '__getitem__') else 0)
                y = getattr(pos, 'y', pos[1] if hasattr(pos, '__getitem__') else 0)
                w = getattr(size, 'width', size[0] if hasattr(size, '__getitem__') else 0)
                h = getattr(size, 'height', size[1] if hasattr(size, '__getitem__') else 0)
                
                # Validation: Ensure reasonable values (e.g., not 0x0 unless it's truly hidden/empty)
                # But some elements might be 0x0. However, for the agent, 0x0 is useless.
                if w > 0 and h > 0:
                    frame = {'x': x, 'y': y, 'w': w, 'h': h}
            except Exception as e:
                 self.logger.debug("Failed to parse frame for element role %s: %s", node["role"], e)
                 frame = None
        
        node["frame"] = frame
        
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

    def _get_attr(self, element: Any, attr_name: str) -> Any:
        try:
            err, value = AXUIElementCopyAttributeValue(element, attr_name, None)
            if err == 0:
                # Convert complex types if necessary
                return value
        except Exception:
            pass
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
