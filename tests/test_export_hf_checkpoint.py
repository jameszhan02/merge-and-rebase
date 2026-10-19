from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from merge_and_rebase.eval.diagnostics.export_hf_checkpoint import (
    export_hf_checkpoint,
    load_complete_state,
)
from merge_and_rebase.eval.llm_rebase.artifacts import export_hf_alpha_sweep


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def save_pretrained(self, path: Path, **kwargs) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"fake")
        (path / "model_save_args.json").write_text(json.dumps(kwargs), encoding="utf-8")


class _FakeTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_load_complete_state_rejects_wrapped_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"x": torch.ones(1)}}, path)
    with pytest.raises(TypeError, match="directly to tensors"):
        load_complete_state(path)


def test_export_hf_checkpoint_strict_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    state_path = tmp_path / "merged.pt"
    reference = _FakeModel()
    torch.save(reference.state_dict(), state_path)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )

    output = tmp_path / "hf_export"
    manifest = export_hf_checkpoint(
        base="fake-base",
        state_path=state_path,
        output_dir=output,
        output_dtype=torch.float32,
    )

    assert (output / "model.safetensors").is_file()
    assert (output / "tokenizer.json").is_file()
    assert json.loads((output / "theseus_export.json").read_text())["base"] == "fake-base"
    assert manifest["merged_state"] == str(state_path)
    assert not list(tmp_path.glob(".hf_export.tmp-*"))

    with pytest.raises(FileExistsError, match="already exists"):
        export_hf_checkpoint(base="fake-base", state_path=state_path, output_dir=output)


def test_export_hf_alpha_sweep_materializes_distinct_complete_models(tmp_path: Path) -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    base = {key: value.detach().clone() for key, value in model.state_dict().items()}
    delta = {"proj.weight": torch.ones_like(base["proj.weight"])}

    records = export_hf_alpha_sweep(
        output_root=tmp_path / "sweep",
        alphas=[0.0, 0.25],
        target_model=model,
        target_tokenizer=tokenizer,
        target_base_sd=base,
        merged_delta=delta,
        output_dtype="fp32",
    )

    assert records == [
        {"alpha": 0.0, "path": str(tmp_path / "sweep" / "alpha_0")},
        {"alpha": 0.25, "path": str(tmp_path / "sweep" / "alpha_0.25")},
    ]
    assert (tmp_path / "sweep" / "alpha_0" / "model.safetensors").is_file()
    manifest = json.loads(
        (tmp_path / "sweep" / "alpha_0.25" / "theseus_export.json").read_text()
    )
    assert manifest["alpha"] == 0.25
    assert torch.equal(model.proj.weight.detach(), base["proj.weight"] + 0.25)

    with pytest.raises(FileExistsError, match="already exist"):
        export_hf_alpha_sweep(
            output_root=tmp_path / "sweep",
            alphas=[0.0],
            target_model=model,
            target_tokenizer=tokenizer,
            target_base_sd=base,
            merged_delta=delta,
            output_dtype="fp32",
        )
