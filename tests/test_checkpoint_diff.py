import torch

from merge_and_rebase.eval.diagnostics.checkpoint_diff import compare_states


def test_compare_states_reports_size_normalized_and_energy_metrics() -> None:
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "model.layers.1.mlp.up_proj.weight": torch.full((4, 4), 2.0),
        "model.norm.weight": torch.ones(2),
    }
    tuned = {key: value.clone() for key, value in base.items()}
    tuned["model.layers.0.self_attn.q_proj.weight"] += 1.0
    tuned["model.layers.1.mlp.up_proj.weight"] += 0.5

    parameters, layers, components, summary = compare_states(base, tuned)

    assert len(parameters) == 3
    assert sum(float(row["update_energy_fraction"]) for row in parameters) == 1.0
    assert summary["changed_numel"] == 20
    assert summary["numel"] == 22
    assert summary["delta_norm_fro"] == torch.tensor(8.0).sqrt().item()

    by_layer = {row["layer_idx"]: row for row in layers}
    assert by_layer[0]["relative_norm"] == 1.0
    assert by_layer[1]["relative_norm"] == 0.25
    assert by_layer["outside"]["relative_norm"] == 0.0
    assert sum(float(row["update_energy_fraction"]) for row in layers) == 1.0

    by_component = {row["component"]: row for row in components}
    assert by_component["self_attn.q_proj"]["changed_fraction"] == 1.0
    assert by_component["final_norm"]["changed_fraction"] == 0.0


def test_compare_states_rejects_shape_mismatch() -> None:
    base = {"model.layers.0.mlp.up_proj.weight": torch.ones(2, 2)}
    tuned = {"model.layers.0.mlp.up_proj.weight": torch.ones(3, 2)}

    try:
        compare_states(base, tuned)
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch")
