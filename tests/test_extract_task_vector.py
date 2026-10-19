import math

import pytest
import torch

from merge_and_rebase.eval.diagnostics.extract_task_vector import _layer_index, _layer_rows, extract_task_vector


def test_extract_task_vector_and_layer_stats():
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "model.layers.1.mlp.down_proj.weight": torch.ones(2, 3),
        "model.layers.1.rotary.inv_freq": torch.arange(2),
    }
    tuned = {
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0),
        "model.layers.1.mlp.down_proj.weight": torch.full((2, 3), 3.0),
        "model.layers.1.rotary.inv_freq": torch.arange(2),
    }

    delta, rows, summary = extract_task_vector(base, tuned, output_dtype=torch.bfloat16)
    layers = _layer_rows(rows)

    assert set(delta) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }
    assert all(value.dtype == torch.bfloat16 for value in delta.values())
    assert summary["tensor_count"] == 2
    assert summary["nonzero_tensor_count"] == 2
    assert layers[0]["delta_norm_fro"] == pytest.approx(2.0)
    assert layers[1]["delta_norm_fro"] == pytest.approx(math.sqrt(24.0))
    assert _layer_index("transformer.blocks.12.attn.weight") == 12


def test_extract_task_vector_rejects_key_and_shape_mismatches():
    with pytest.raises(KeyError, match="keys differ"):
        extract_task_vector({"a": torch.ones(1)}, {"b": torch.ones(1)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        extract_task_vector({"a": torch.ones(1)}, {"a": torch.ones(2)})
