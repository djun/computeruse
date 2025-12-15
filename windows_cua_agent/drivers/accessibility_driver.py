from __future__ import annotations

import os
from typing import Any, Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.coordinates import clamp_point, point_to_px
from cua_agent.utils.logger import get_logger
from windows_cua_agent.utils.windows_integration import get_display_info, get_foreground_process_image_name


class AccessibilityDriver:
    """
    Windows semantic UI driver.

    This adapter is designed to use Microsoft UI Automation (UIA) via `comtypes` when available.
    In environments where UIA bindings are not installed, it gracefully degrades so the core
    can fall back to visual grounding (OCR/blob detection).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display = get_display_info()
        self._uia_available = self._check_uia_available()
        self._uia: Any | None = None
        self._uia_mod: Any | None = None
        self._walker: Any | None = None
        self._cache_request: Any | None = None
        self._init_error: str | None = None

        if not self._uia_available:
            self.logger.info(
                "UI Automation bindings not available; semantic tree/phantom mode disabled. "
                "Install `comtypes` to enable UIA-based grounding."
            )

    def _check_uia_available(self) -> bool:
        try:
            import comtypes  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

    # --- Public API ---

    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult:
        """
        Capture the semantic UI tree of the currently active window.

        Uses UIA Control View + property caching; returns a JSON-serializable dict.
        """
        if not self._ensure_uia():
            return ActionResult(success=False, reason=self._init_error or "UI Automation unavailable")

        try:
            root = self._get_active_window_element()
            if not root:
                return ActionResult(success=False, reason="No active window element found")

            budget = [800]  # cap traversal to keep latency/tokens bounded
            tree = self._build_tree(root, depth=0, max_depth=max_depth, budget=budget)
            if not self._tree_has_frame(tree):
                return ActionResult(success=False, reason="UIA tree returned without usable frames")
            return ActionResult(success=True, reason="captured", metadata={"tree": tree})
        except Exception as exc:
            self.logger.exception("UIA get_active_window_tree failed")
            return ActionResult(success=False, reason=f"UIA error: {exc}")

    def probe_element(self, x: float, y: float, radius: float = 0.0) -> ActionResult:
        """
        Return a small semantic tree for the element at (x,y) (logical coordinates).
        Optionally samples neighboring points to disambiguate crowded UIs.
        """
        if not self._ensure_uia():
            return ActionResult(success=False, reason=self._init_error or "UI Automation unavailable")

        try:
            center = self._element_from_point(x, y)
            if not center:
                return ActionResult(success=False, reason=f"No element found at ({x}, {y})")

            budget = [120]
            center_tree = self._build_tree(center, depth=0, max_depth=2, budget=budget)

            neighbors: list[dict] = []
            if radius and radius > 0:
                offsets = [(-radius, 0.0), (radius, 0.0), (0.0, -radius), (0.0, radius)]
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    el = self._element_from_point(nx, ny)
                    if not el:
                        continue
                    nb_budget = [60]
                    nb_tree = self._build_tree(el, depth=0, max_depth=1, budget=nb_budget)
                    neighbors.append({"x": nx, "y": ny, "tree": nb_tree})

            metadata: dict = {"tree": center_tree}
            if neighbors:
                metadata["neighbors"] = neighbors
            return ActionResult(success=True, reason="probed", metadata=metadata)
        except Exception as exc:
            self.logger.exception("UIA probe_element failed")
            return ActionResult(success=False, reason=f"UIA probe failed: {exc}")

    def perform_action_at(self, x: float, y: float, action: str) -> ActionResult:
        """
        Phantom-mode execution at the given coordinates.

        Supported actions (macOS AX-compatible names):
        - AXPress: invokes/toggles/selects/expands the element (best-effort)
        - AXShowMenu: expands/clicks the element (best-effort)
        """
        if not self._ensure_uia():
            return ActionResult(success=False, reason=self._init_error or "UI Automation unavailable")

        element = self._element_from_point(x, y)
        if not element:
            return ActionResult(success=False, reason=f"No element found at ({x}, {y})")

        try:
            curr = element
            for _ in range(4):  # climb a few ancestors to find a supported pattern
                ok = self._perform_action_on_element(curr, action)
                if ok:
                    return ActionResult(success=True, reason=f"Phantom {action} executed")
                parent = self._get_parent(curr)
                if not parent:
                    break
                curr = parent
        except Exception as exc:
            return ActionResult(success=False, reason=f"Phantom action failed: {exc}")

        return ActionResult(success=False, reason=f"Action {action} not supported at ({x},{y})")

    def set_text_element_value(self, x: float, y: float, value: str) -> ActionResult:
        """Set a nearby element's value using UIA ValuePattern.SetValue (best-effort)."""
        if not self._ensure_uia():
            return ActionResult(success=False, reason=self._init_error or "UI Automation unavailable")

        element = self._element_from_point(x, y)
        if not element:
            return ActionResult(success=False, reason=f"No element found at ({x}, {y})")

        try:
            curr = element
            for _ in range(4):
                if self._try_set_value(curr, value):
                    return ActionResult(success=True, reason="Phantom type via ValuePattern")
                parent = self._get_parent(curr)
                if not parent:
                    break
                curr = parent
        except Exception as exc:
            return ActionResult(success=False, reason=f"Phantom type failed: {exc}")

        return ActionResult(success=False, reason="Failed to set value on target or parents")

    def set_focused_element_value(self, value: str) -> ActionResult:
        """Set the focused element's value using ValuePattern.SetValue (best-effort)."""
        if not self._ensure_uia():
            return ActionResult(success=False, reason=self._init_error or "UI Automation unavailable")

        try:
            focused = self._uia.GetFocusedElement()  # type: ignore[union-attr]
            if not focused:
                return ActionResult(success=False, reason="No focused element")
            if self._try_set_value(focused, value):
                return ActionResult(success=True, reason="Phantom type on focused element")
            return ActionResult(success=False, reason="Focused element does not support ValuePattern")
        except Exception as exc:
            return ActionResult(success=False, reason=f"Focused phantom type failed: {exc}")

    def get_focused_app_name(self) -> str:
        return get_foreground_process_image_name() or ""

    # --- UIA initialization ---

    def _ensure_uia(self) -> bool:
        if not self._uia_available:
            self._init_error = self._init_error or "UI Automation unavailable"
            return False
        if self._uia is not None:
            return True
        if self._init_error:
            return False

        try:
            import comtypes  # type: ignore
            import comtypes.client  # type: ignore

            try:
                comtypes.CoInitialize()
            except Exception:
                pass

            # Load UIAutomation type library (generates comtypes.gen.UIAutomationClient)
            try:
                comtypes.client.GetModule("UIAutomationCore.dll")
            except Exception:
                windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
                dll_path = os.path.join(windir, "System32", "UIAutomationCore.dll")
                comtypes.client.GetModule(dll_path)

            from comtypes.gen import UIAutomationClient as uia  # type: ignore

            self._uia_mod = uia
            self._uia = comtypes.client.CreateObject(uia.CUIAutomation, interface=uia.IUIAutomation)
            self._walker = self._uia.ControlViewWalker
            self._cache_request = self._build_cache_request()
            return True
        except Exception as exc:
            self._init_error = f"UIA init failed: {exc}"
            self.logger.warning("%s", self._init_error)
            return False

    def _build_cache_request(self) -> Any | None:
        if not self._uia or not self._uia_mod:
            return None
        uia = self._uia_mod
        req = self._uia.CreateCacheRequest()
        try:
            req.TreeScope = uia.TreeScope_Element
        except Exception:
            pass

        prop_ids = [
            getattr(uia, "UIA_NamePropertyId", None),
            getattr(uia, "UIA_ControlTypePropertyId", None),
            getattr(uia, "UIA_BoundingRectanglePropertyId", None),
            getattr(uia, "UIA_IsEnabledPropertyId", None),
            getattr(uia, "UIA_IsOffscreenPropertyId", None),
            getattr(uia, "UIA_ProcessIdPropertyId", None),
            getattr(uia, "UIA_NativeWindowHandlePropertyId", None),
            getattr(uia, "UIA_AutomationIdPropertyId", None),
            getattr(uia, "UIA_ClassNamePropertyId", None),
            getattr(uia, "UIA_ValueValuePropertyId", None),
        ]
        for pid in prop_ids:
            if not pid:
                continue
            try:
                req.AddProperty(int(pid))
            except Exception:
                pass

        pattern_ids = [
            getattr(uia, "UIA_InvokePatternId", None),
            getattr(uia, "UIA_TogglePatternId", None),
            getattr(uia, "UIA_ValuePatternId", None),
            getattr(uia, "UIA_SelectionItemPatternId", None),
            getattr(uia, "UIA_ExpandCollapsePatternId", None),
        ]
        for pat in pattern_ids:
            if not pat:
                continue
            try:
                req.AddPattern(int(pat))
            except Exception:
                pass
        return req

    # --- UIA helpers ---

    def _get_active_window_element(self) -> Any | None:
        if not self._uia:
            return None
        hwnd = self._get_foreground_hwnd()
        if hwnd:
            try:
                return self._uia.ElementFromHandle(hwnd)
            except Exception:
                pass
        try:
            return self._uia.GetFocusedElement()
        except Exception:
            return None

    def _get_foreground_hwnd(self) -> int:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            return int(user32.GetForegroundWindow() or 0)
        except Exception:
            return 0

    def _element_from_point(self, x: float, y: float) -> Any | None:
        if not self._uia or not self._uia_mod:
            return None
        try:
            lx, ly = clamp_point(float(x), float(y), self.display.logical_width, self.display.logical_height)
            px, py = point_to_px(lx, ly, self.display.scale_factor)

            import ctypes

            point_cls = getattr(self._uia_mod, "tagPOINT", None)
            if point_cls is None:
                class _POINT(ctypes.Structure):  # noqa: D401
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                point_cls = _POINT
            pt = point_cls()
            pt.x = int(px)
            pt.y = int(py)
            return self._uia.ElementFromPoint(pt)
        except Exception:
            return None

    def _get_parent(self, element: Any) -> Any | None:
        if not self._walker:
            return None
        try:
            return self._walker.GetParentElement(element)
        except Exception:
            return None

    def _first_child(self, element: Any) -> Any | None:
        if not self._walker:
            return None
        if self._cache_request and hasattr(self._walker, "GetFirstChildElementBuildCache"):
            try:
                return self._walker.GetFirstChildElementBuildCache(element, self._cache_request)
            except Exception:
                pass
        try:
            child = self._walker.GetFirstChildElement(element)
            return self._cache_element(child)
        except Exception:
            return None

    def _next_sibling(self, element: Any) -> Any | None:
        if not self._walker:
            return None
        if self._cache_request and hasattr(self._walker, "GetNextSiblingElementBuildCache"):
            try:
                return self._walker.GetNextSiblingElementBuildCache(element, self._cache_request)
            except Exception:
                pass
        try:
            sib = self._walker.GetNextSiblingElement(element)
            return self._cache_element(sib)
        except Exception:
            return None

    def _cache_element(self, element: Any | None) -> Any | None:
        if not element or not self._cache_request:
            return element
        try:
            if hasattr(element, "BuildUpdatedCache"):
                return element.BuildUpdatedCache(self._cache_request)
        except Exception:
            pass
        return element

    # --- Tree building ---

    _ROLE_MAP = {
        50000: "AXButton",  # UIA_ButtonControlTypeId
        50002: "AXCheckBox",  # UIA_CheckBoxControlTypeId
        50003: "AXComboBox",  # UIA_ComboBoxControlTypeId
        50004: "AXTextField",  # UIA_EditControlTypeId
        50005: "AXLink",  # UIA_HyperlinkControlTypeId
        50007: "AXListItem",  # UIA_ListItemControlTypeId
        50008: "AXList",  # UIA_ListControlTypeId
        50009: "AXMenu",  # UIA_MenuControlTypeId
        50010: "AXMenuBar",  # UIA_MenuBarControlTypeId
        50011: "AXMenuItem",  # UIA_MenuItemControlTypeId
        50018: "AXTabGroup",  # UIA_TabControlTypeId
        50019: "AXTab",  # UIA_TabItemControlTypeId
        50020: "AXStaticText",  # UIA_TextControlTypeId
        50025: "AXTable",  # UIA_TableControlTypeId
        50030: "AXTextArea",  # UIA_DocumentControlTypeId
        50032: "AXWindow",  # UIA_WindowControlTypeId
        50033: "AXGroup",  # UIA_PaneControlTypeId
    }

    _INTERACTIVE_ROLES = {
        "AXButton",
        "AXCheckBox",
        "AXComboBox",
        "AXLink",
        "AXMenuItem",
        "AXTab",
        "AXTextField",
        "AXTextArea",
        "AXListItem",
    }

    def _build_tree(self, element: Any, *, depth: int, max_depth: int, budget: list[int]) -> dict:
        if budget[0] <= 0:
            return {"role": "node_budget_exhausted"}
        if depth > max_depth:
            return {"role": "max_depth_reached"}
        budget[0] -= 1

        el = self._cache_element(element) or element

        name = self._safe_get(el, "CachedName") or self._safe_get(el, "CurrentName") or ""
        control_type = self._safe_get(el, "CachedControlType")
        if control_type is None:
            control_type = self._safe_get(el, "CurrentControlType")
        try:
            ct_int = int(control_type) if control_type is not None else 0
        except Exception:
            ct_int = 0

        role = self._ROLE_MAP.get(ct_int, "AXUnknown")

        rect = self._safe_get(el, "CachedBoundingRectangle")
        if rect is None:
            rect = self._safe_get(el, "CurrentBoundingRectangle")
        frame = self._rect_to_frame(rect)

        value = self._get_value_property(el)

        node: dict = {"role": role}
        if name:
            node["title"] = str(name)
        if value and role in {"AXTextField", "AXTextArea"}:
            node["value"] = value
        if frame:
            node["frame"] = frame

        if depth < max_depth:
            children: list[dict] = []
            child = self._first_child(el)
            while child and budget[0] > 0:
                child_node = self._build_tree(child, depth=depth + 1, max_depth=max_depth, budget=budget)
                if self._is_useful_node(child_node):
                    children.append(child_node)
                child = self._next_sibling(child)
            if children:
                node["children"] = children

        return node

    def _safe_get(self, obj: Any, attr: str) -> Any | None:
        try:
            return getattr(obj, attr)
        except Exception:
            return None

    def _get_value_property(self, element: Any) -> str:
        if not self._uia_mod:
            return ""
        prop_id = getattr(self._uia_mod, "UIA_ValueValuePropertyId", None)
        if not prop_id:
            return ""
        try:
            val = element.GetCachedPropertyValue(int(prop_id))
        except Exception:
            try:
                val = element.GetCurrentPropertyValue(int(prop_id))
            except Exception:
                return ""
        val = getattr(val, "value", val)
        return val if isinstance(val, str) and val else ""

    def _rect_to_frame(self, rect: Any | None) -> dict | None:
        if rect is None:
            return None
        try:
            left = float(rect.left)
            top = float(rect.top)
            right = float(rect.right)
            bottom = float(rect.bottom)
        except Exception:
            try:
                left, top, right, bottom = [float(x) for x in rect]
            except Exception:
                return None

        # Intersect with primary monitor bounds (agent assumes single primary monitor).
        x0 = max(0.0, left)
        y0 = max(0.0, top)
        x1 = min(float(self.display.physical_width), right)
        y1 = min(float(self.display.physical_height), bottom)
        w = x1 - x0
        h = y1 - y0
        if w <= 1.0 or h <= 1.0:
            return None

        scale = float(self.display.scale_factor or 1.0)
        lx = x0 / scale
        ly = y0 / scale
        lw = w / scale
        lh = h / scale

        # Clamp to logical bounds.
        lx = max(0.0, min(lx, float(self.display.logical_width)))
        ly = max(0.0, min(ly, float(self.display.logical_height)))
        lw = max(0.0, min(lw, float(self.display.logical_width) - lx))
        lh = max(0.0, min(lh, float(self.display.logical_height) - ly))
        if lw <= 0.0 or lh <= 0.0:
            return None

        return {"x": lx, "y": ly, "w": lw, "h": lh}

    def _is_useful_node(self, node: dict) -> bool:
        if not node:
            return False
        if node.get("children"):
            return True
        frame = node.get("frame") or {}
        if not frame or float(frame.get("w", 0) or 0) <= 0 or float(frame.get("h", 0) or 0) <= 0:
            return False
        if (node.get("title") or "").strip() or (node.get("value") or "").strip():
            return True
        return (node.get("role") or "") in self._INTERACTIVE_ROLES

    def _tree_has_frame(self, node: dict) -> bool:
        if not node:
            return False
        frame = node.get("frame") or {}
        if frame and float(frame.get("w", 0) or 0) > 0 and float(frame.get("h", 0) or 0) > 0:
            return True
        for child in node.get("children") or []:
            if self._tree_has_frame(child):
                return True
        return False

    # --- Phantom-mode pattern execution ---

    def _perform_action_on_element(self, element: Any, action: str) -> bool:
        action_l = (action or "").strip()
        if not action_l:
            return False
        try:
            if hasattr(element, "SetFocus"):
                try:
                    element.SetFocus()
                except Exception:
                    pass

            if action_l == "AXPress":
                return self._try_invoke(element) or self._try_toggle(element) or self._try_select(element) or self._try_expand(element)
            if action_l == "AXShowMenu":
                # Best-effort: expanding/invoking often reveals menus for combo buttons and menu items.
                return self._try_expand(element) or self._try_invoke(element)
        except Exception:
            return False
        return False

    def _get_pattern(self, element: Any, pattern_id: int, iface: Any) -> Any | None:
        try:
            pat = element.GetCurrentPattern(int(pattern_id))
            if not pat:
                return None
            return pat.QueryInterface(iface)
        except Exception:
            return None

    def _try_invoke(self, element: Any) -> bool:
        if not self._uia_mod:
            return False
        pid = getattr(self._uia_mod, "UIA_InvokePatternId", None)
        iface = getattr(self._uia_mod, "IUIAutomationInvokePattern", None)
        if not pid or not iface:
            return False
        inv = self._get_pattern(element, int(pid), iface)
        if not inv:
            return False
        try:
            inv.Invoke()
            return True
        except Exception:
            return False

    def _try_toggle(self, element: Any) -> bool:
        if not self._uia_mod:
            return False
        pid = getattr(self._uia_mod, "UIA_TogglePatternId", None)
        iface = getattr(self._uia_mod, "IUIAutomationTogglePattern", None)
        if not pid or not iface:
            return False
        pat = self._get_pattern(element, int(pid), iface)
        if not pat:
            return False
        try:
            pat.Toggle()
            return True
        except Exception:
            return False

    def _try_select(self, element: Any) -> bool:
        if not self._uia_mod:
            return False
        pid = getattr(self._uia_mod, "UIA_SelectionItemPatternId", None)
        iface = getattr(self._uia_mod, "IUIAutomationSelectionItemPattern", None)
        if not pid or not iface:
            return False
        pat = self._get_pattern(element, int(pid), iface)
        if not pat:
            return False
        try:
            pat.Select()
            return True
        except Exception:
            return False

    def _try_expand(self, element: Any) -> bool:
        if not self._uia_mod:
            return False
        pid = getattr(self._uia_mod, "UIA_ExpandCollapsePatternId", None)
        iface = getattr(self._uia_mod, "IUIAutomationExpandCollapsePattern", None)
        if not pid or not iface:
            return False
        pat = self._get_pattern(element, int(pid), iface)
        if not pat:
            return False
        try:
            pat.Expand()
            return True
        except Exception:
            return False

    def _try_set_value(self, element: Any, value: str) -> bool:
        if not self._uia_mod:
            return False
        pid = getattr(self._uia_mod, "UIA_ValuePatternId", None)
        iface = getattr(self._uia_mod, "IUIAutomationValuePattern", None)
        if not pid or not iface:
            return False
        pat = self._get_pattern(element, int(pid), iface)
        if not pat:
            return False
        try:
            if hasattr(element, "SetFocus"):
                try:
                    element.SetFocus()
                except Exception:
                    pass
            pat.SetValue(str(value))
            return True
        except Exception:
            return False
