from cua_agent.grounding.fusion import GroundingFusion


def test_fusion_merges_overlapping_semantic_and_visual_nodes() -> None:
    fusion = GroundingFusion()

    fused = fusion.fuse(
        [
            {
                "role": "AXButton",
                "label": "Enviar",
                "path": "window > button",
                "frame": {"x": 10, "y": 10, "w": 100, "h": 30},
                "confidence": 0.8,
            }
        ],
        [
            {
                "role": "button",
                "label": "Enviar",
                "path": "vision.ocr.1",
                "frame": {"x": 12, "y": 11, "w": 98, "h": 28},
                "confidence": 0.7,
            }
        ],
    )

    assert len(fused) == 1
    assert fused[0].source == "fused"
    assert fused[0].role == "button"
    assert fused[0].label == "Enviar"
    assert fused[0].confidence >= 0.8


def test_fusion_keeps_visual_only_candidates() -> None:
    fusion = GroundingFusion()

    fused = fusion.fuse([], [{"role": "text", "label": "Erro", "frame": {"x": 1, "y": 2, "w": 30, "h": 12}}])

    assert len(fused) == 1
    assert fused[0].source == "visual"
    assert fused[0].role == "static_text"
    assert fused[0].gid.startswith("vis:")
