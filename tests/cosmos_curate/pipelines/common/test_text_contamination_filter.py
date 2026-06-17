# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for game text-contamination VLM filtering prompts and parsing."""

from cosmos_curate.pipelines.common.filter_prompts import get_qwen_filter_prompt
from cosmos_curate.pipelines.common.semantic_filter_postprocess import evaluate_semantic_window_results


def test_text_contamination_prompt_defines_ui_and_scene_text() -> None:
    """The dedicated prompt must distinguish UI overlays from world text."""
    prompt = get_qwen_filter_prompt("text-contamination", "post-production text")

    assert "HUD" in prompt
    assert "menus" in prompt
    assert "chat" in prompt
    assert "physically present inside the 3D world" in prompt
    assert '"text_type"' in prompt
    assert '"reason"' in prompt


def test_semantic_eval_preserves_text_contamination_audit_fields() -> None:
    """Semantic parsing should return text_type and reason alongside rejection reasons."""
    result = (
        '{"post-production text": "yes", '
        '"text_type": "hud_ui", '
        '"reason": "Quest text is overlaid on the scene."}'
    )

    _, _, per_window_reasons, _, per_window_audit = evaluate_semantic_window_results(
        [(0, result)],
        filter_criteria=["post-production text"],
        rejection_threshold=0.5,
        score_only=True,
    )

    assert per_window_reasons == {0: {"post-production text": "yes"}}
    assert per_window_audit == {
        0: {
            "text_type": "hud_ui",
            "reason": "Quest text is overlaid on the scene.",
        }
    }
