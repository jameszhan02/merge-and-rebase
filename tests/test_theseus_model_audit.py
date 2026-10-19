from types import SimpleNamespace

from merge_and_rebase.eval.diagnostics.theseus_model_audit import (
    _config_summary,
    _same_architecture,
    _shape_matches,
)


def _config(**overrides):
    values = {
        "model_type": "olmo",
        "architectures": ["OlmoForCausalLM"],
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_summary_and_expected_shape():
    summary = _config_summary(_config(), "source")

    assert summary["checkpoint"] == "source"
    assert _shape_matches(summary, (16, 2048, 8192))
    assert not _shape_matches(summary, (16, 1024, 8192))


def test_same_architecture_checks_attention_layout_too():
    base = _config_summary(_config(), "base")
    tuned = _config_summary(_config(), "tuned")
    changed_kv = _config_summary(_config(num_key_value_heads=8), "changed")

    assert _same_architecture(base, tuned)
    assert not _same_architecture(base, changed_kv)
