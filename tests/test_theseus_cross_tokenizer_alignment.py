from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.rebase.methods._shared import _ActivationHook, _align_text_content_features


def test_interpolation_uses_each_tokenizers_content_length() -> None:
    source = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    target = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    source_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    target_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])

    rows = _align_text_content_features(
        source,
        target,
        mode="interpolate",
        source_attention_mask=source_mask,
        target_attention_mask=target_mask,
    )

    assert rows is not None
    source_rows, target_rows = rows
    assert source_rows.shape == (6, 2)
    assert target_rows.shape == (6, 3)
    torch.testing.assert_close(target_rows, torch.cat([target[0, :2], target[1, :4]]))


def test_content_alignment_ignores_left_and_right_padding() -> None:
    source = torch.tensor([[[99.0], [1.0], [3.0], [99.0]]])
    target = torch.tensor([[[2.0], [4.0], [88.0], [88.0]]])
    rows = _align_text_content_features(
        source,
        target,
        mode="mean",
        source_attention_mask=torch.tensor([[0, 1, 1, 0]]),
        target_attention_mask=torch.tensor([[1, 1, 0, 0]]),
    )

    assert rows is not None
    source_rows, target_rows = rows
    torch.testing.assert_close(source_rows, torch.tensor([[2.0]]))
    torch.testing.assert_close(target_rows, torch.tensor([[3.0]]))


def test_activation_hook_registers_only_selected_modules() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    hook = _ActivationHook(model, module_names={"0"})
    try:
        model(torch.ones(1, 2))
        assert set(hook.inputs) == {"0"}
        assert set(hook.outputs) == {"0"}
    finally:
        hook.remove()
