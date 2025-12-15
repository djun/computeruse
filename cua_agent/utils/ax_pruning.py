"""Helpers for pruning accessibility/UI trees for prompting."""

from __future__ import annotations

from typing import Any, Dict, List

INTERACTIVE_ROLES = {"AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox", "AXComboBox", "AXMenuItem"}


def prune_ax_tree_for_prompt(tree: Dict[str, Any], max_nodes: int = 120, max_depth: int = 4) -> Dict[str, Any]:
    """
    Return a pruned tree that keeps only interactive/labelled nodes and drops deep/empty branches.
    """
    if not tree:
        return {}

    kept = 0

    def _keep(node: Dict[str, Any]) -> bool:
        if node.get("role") in INTERACTIVE_ROLES:
            return True
        if node.get("title") or node.get("value"):
            return True
        if node.get("frame") and node["frame"].get("w", 0) > 0 and node["frame"].get("h", 0) > 0:
            return True
        return False

    def _walk(node: Dict[str, Any], depth: int) -> Dict[str, Any] | None:
        nonlocal kept
        if kept >= max_nodes or depth > max_depth:
            return None
        children = []
        for child in node.get("children") or []:
            if kept >= max_nodes:
                break
            pruned = _walk(child, depth + 1)
            if pruned:
                children.append(pruned)

        useful = _keep(node) or bool(children)
        if not useful:
            return None

        kept += 1
        out = {
            "role": node.get("role"),
            "title": node.get("title"),
            "value": node.get("value"),
            "frame": node.get("frame"),
        }
        if children:
            out["children"] = children
        return out

    pruned_root = _walk(tree, 0)
    return pruned_root or {}
