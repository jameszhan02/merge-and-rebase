"""Saving rebased states and direct Hugging Face exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import math
import shutil
import tempfile

import torch

from ...merge.runtime import apply_delta, to_cpu_fp32


_EXPORT_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def save_merged_state(
    path: Any,
    merged_delta: dict[str, torch.Tensor],
    best_alpha: float,
    target_base_sd: dict[str, torch.Tensor],
    *,
    message: str,
) -> None:
    scaled = {k: v * best_alpha for k, v in merged_delta.items()}
    best_sd = apply_delta(target_base_sd, scaled)
    outp = Path(str(path))
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_cpu_fp32(best_sd), str(outp))
    print(f"{message} {outp}")


def _alpha_dir_name(alpha: float) -> str:
    return f"alpha_{format(float(alpha), '.12g')}"


def export_hf_alpha_sweep(
    *,
    output_root: Path,
    alphas: list[float],
    target_model: Any,
    target_tokenizer: Any,
    target_base_sd: dict[str, torch.Tensor],
    merged_delta: dict[str, torch.Tensor],
    output_dtype: str = "bf16",
    max_shard_size: str = "2GB",
    manifest_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Materialize one complete HF checkpoint per alpha without an intermediate ``.pt`` file."""
    if output_dtype not in _EXPORT_DTYPES:
        raise ValueError(f"output_dtype must be one of {sorted(_EXPORT_DTYPES)}")
    if not alphas:
        raise ValueError("At least one export alpha is required.")
    normalized = [float(alpha) for alpha in alphas]
    if any(not math.isfinite(alpha) for alpha in normalized):
        raise ValueError("Export alphas must all be finite numbers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Export alphas must be unique: {normalized}")

    outputs = [output_root / _alpha_dir_name(alpha) for alpha in normalized]
    collisions = [path for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "HF export directories already exist; refusing to mix or overwrite checkpoints: "
            + ", ".join(str(path) for path in collisions)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    dtype = _EXPORT_DTYPES[output_dtype]
    target_model.to(device="cpu", dtype=dtype)
    records: list[dict[str, Any]] = []

    for alpha, output_dir in zip(normalized, outputs, strict=True):
        print(f"Materializing complete HF checkpoint at alpha={alpha:.12g}...")
        # Fuse scaling and addition so a sweep does not hold a second full-sized
        # ``scaled_delta`` beside the merged state. This matters for 1B+ models.
        merged = dict(target_base_sd)
        for key, value in merged_delta.items():
            if key in merged:
                merged[key] = merged[key].add(
                    value.to(dtype=merged[key].dtype, device=merged[key].device),
                    alpha=alpha,
                )
            else:
                merged[key] = value * alpha
        missing, unexpected = target_model.load_state_dict(merged, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Strict target load failed at alpha={alpha}: missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        del merged

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_root))
        )
        try:
            target_model.save_pretrained(
                temporary_dir,
                safe_serialization=True,
                max_shard_size=max_shard_size,
            )
            target_tokenizer.save_pretrained(temporary_dir)
            manifest = {
                "format": "huggingface_pretrained",
                "alpha": alpha,
                "formula": "target_base + alpha * transported_task_vector",
                "output_dtype": output_dtype,
                "max_shard_size": max_shard_size,
                **(manifest_context or {}),
            }
            (temporary_dir / "theseus_export.json").write_text(
                json.dumps(manifest, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            temporary_dir.replace(output_dir)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        print(f"Saved complete HF checkpoint: {output_dir}")
        records.append({"alpha": alpha, "path": str(output_dir)})

    return records
