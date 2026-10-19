from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods._shared import (
    _LayerTransform,
    filter_mapping_by_parameter,
    filter_transforms_by_layer,
)


def _transforms() -> dict[str, _LayerTransform]:
    t = _LayerTransform(kind="weight", t_in=torch.eye(2), t_out=torch.eye(2))
    return {
        "model.layers.0.self_attn.q_proj.weight": t,
        "model.layers.1.self_attn.q_proj.weight": t,
        "model.layers.2.mlp.down_proj.weight": t,
        "model.norm.weight": t,  # not layer-scoped: must survive any filter
    }


def test_no_filter_is_a_no_op() -> None:
    transforms = _transforms()
    assert filter_transforms_by_layer(transforms, None) == transforms


def test_include_keeps_only_named_layers_plus_global_keys() -> None:
    kept = filter_transforms_by_layer(_transforms(), {"include": [1]})
    assert set(kept) == {"model.layers.1.self_attn.q_proj.weight", "model.norm.weight"}


def test_exclude_drops_only_named_layers() -> None:
    kept = filter_transforms_by_layer(_transforms(), {"exclude": [0]})
    assert set(kept) == {
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
        "model.norm.weight",
    }


def test_include_and_exclude_together_is_rejected() -> None:
    with pytest.raises(ValueError, match="include.*or.*exclude"):
        filter_transforms_by_layer(_transforms(), {"include": [0], "exclude": [1]})


def test_empty_filter_is_a_no_op() -> None:
    # An empty mapping is falsy, same as None: no selection requested, nothing is dropped.
    transforms = _transforms()
    assert filter_transforms_by_layer(transforms, {}) == transforms


def test_neither_include_nor_exclude_is_rejected() -> None:
    with pytest.raises(ValueError, match="include.*or.*exclude"):
        filter_transforms_by_layer(_transforms(), {"mode": "typo"})


def test_parameter_filter_keeps_attention_projection_suffixes_only() -> None:
    kept = filter_mapping_by_parameter(
        _transforms(),
        {"include": ["self_attn.q_proj.weight"]},
    )
    assert set(kept) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    }


def test_parameter_filter_supports_globs_and_rejects_bad_config() -> None:
    kept = filter_mapping_by_parameter(_transforms(), {"include": ["*.mlp.down_proj.weight"]})
    assert set(kept) == {"model.layers.2.mlp.down_proj.weight"}
    with pytest.raises(ValueError, match="include.*or.*exclude"):
        filter_mapping_by_parameter(_transforms(), {"include": ["q_proj"], "exclude": ["mlp"]})
    with pytest.raises(ValueError, match="non-empty"):
        filter_mapping_by_parameter(_transforms(), {"include": []})
