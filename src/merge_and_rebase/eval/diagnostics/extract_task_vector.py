"""Extract and audit a dense task vector from full Hugging Face checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


_OUTPUT_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)(?:\.|$)")


def _layer_index(key: str) -> int | None:
    match = _LAYER_PATTERN.search(key)
    return int(match.group(1)) if match else None


def extract_task_vector(
    base: dict[str, torch.Tensor],
    tuned: dict[str, torch.Tensor],
    *,
    output_dtype: torch.dtype = torch.float32,
    near_zero_threshold: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Return ``tuned - base`` plus per-tensor and aggregate diagnostics.

    Only floating-point tensors participate. Key sets and shapes are strict because
    a partial task vector can otherwise look valid while silently omitting updates.
    """
    base_float = {key: value for key, value in base.items() if value.is_floating_point()}
    tuned_float = {key: value for key, value in tuned.items() if value.is_floating_point()}
    missing_in_tuned = sorted(set(base_float) - set(tuned_float))
    missing_in_base = sorted(set(tuned_float) - set(base_float))
    if missing_in_tuned or missing_in_base:
        raise KeyError(
            "Floating checkpoint keys differ.\n"
            f"Missing in tuned ({len(missing_in_tuned)}): {missing_in_tuned[:20]}\n"
            f"Missing in base ({len(missing_in_base)}): {missing_in_base[:20]}"
        )

    delta: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    total_sq = 0.0
    nonzero_tensors = 0
    near_zero_tensors = 0
    for key in sorted(base_float):
        base_tensor = base_float[key].detach().cpu()
        tuned_tensor = tuned_float[key].detach().cpu()
        if base_tensor.shape != tuned_tensor.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': base={tuple(base_tensor.shape)}, tuned={tuple(tuned_tensor.shape)}"
            )

        work_delta = tuned_tensor.float() - base_tensor.float()
        if not torch.isfinite(work_delta).all():
            raise ValueError(f"NaN or Inf found in task vector tensor '{key}'")
        delta_norm = float(torch.linalg.vector_norm(work_delta).item())
        base_norm = float(torch.linalg.vector_norm(base_tensor.float()).item())
        total_sq += delta_norm * delta_norm
        is_near_zero = delta_norm <= near_zero_threshold
        nonzero_tensors += int(delta_norm > 0.0)
        near_zero_tensors += int(is_near_zero)
        layer_idx = _layer_index(key)
        rows.append(
            {
                "layer_idx": "" if layer_idx is None else layer_idx,
                "weight_name": key,
                "shape": "x".join(str(dim) for dim in work_delta.shape),
                "numel": work_delta.numel(),
                "base_dtype": str(base_tensor.dtype).removeprefix("torch."),
                "tuned_dtype": str(tuned_tensor.dtype).removeprefix("torch."),
                "delta_norm_fro": delta_norm,
                "base_norm_fro": base_norm,
                "relative_norm": delta_norm / base_norm if base_norm > 0.0 else math.inf,
                "near_zero": is_near_zero,
            }
        )
        delta[key] = work_delta.to(dtype=output_dtype).contiguous()

    summary = {
        "tensor_count": len(delta),
        "nonzero_tensor_count": nonzero_tensors,
        "near_zero_tensor_count": near_zero_tensors,
        "total_norm_fro": math.sqrt(total_sq),
        "output_dtype": str(output_dtype).removeprefix("torch."),
        "near_zero_threshold": near_zero_threshold,
    }
    if not delta:
        raise ValueError("No floating-point tensors found in the checkpoints")
    if summary["total_norm_fro"] <= near_zero_threshold:
        raise ValueError("The extracted task vector is effectively zero; base and tuned may be the same checkpoint")
    return delta, rows, summary


def _layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[int, dict[str, float | int]] = defaultdict(lambda: {"norm_sq": 0.0, "tensor_count": 0, "numel": 0})
    for row in rows:
        if row["layer_idx"] == "":
            continue
        layer_idx = int(row["layer_idx"])
        totals[layer_idx]["norm_sq"] = float(totals[layer_idx]["norm_sq"]) + float(row["delta_norm_fro"]) ** 2
        totals[layer_idx]["tensor_count"] = int(totals[layer_idx]["tensor_count"]) + 1
        totals[layer_idx]["numel"] = int(totals[layer_idx]["numel"]) + int(row["numel"])
    return [
        {
            "layer_idx": layer_idx,
            "delta_norm_fro": math.sqrt(float(values["norm_sq"])),
            "tensor_count": values["tensor_count"],
            "numel": values["numel"],
        }
        for layer_idx, values in sorted(totals.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_full_model_state(ref: str, *, trust_remote_code: bool) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ref,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if isinstance(value, torch.Tensor)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tau = tuned - base from two full dense HF checkpoints.")
    parser.add_argument("--base", required=True, help="Base Hugging Face checkpoint ID or local directory.")
    parser.add_argument("--tuned", required=True, help="Full tuned Hugging Face checkpoint ID or local directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output .safetensors task-vector path.")
    parser.add_argument("--output-dtype", choices=sorted(_OUTPUT_DTYPES), default="fp32")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-8)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.suffix != ".safetensors":
        raise ValueError("--output must end in .safetensors")
    if args.near_zero_threshold < 0.0:
        raise ValueError("--near-zero-threshold must be >= 0")

    print(f"Loading base checkpoint on CPU: {args.base}")
    base = _load_full_model_state(args.base, trust_remote_code=args.trust_remote_code)
    print(f"Loading tuned checkpoint on CPU: {args.tuned}")
    tuned = _load_full_model_state(args.tuned, trust_remote_code=args.trust_remote_code)
    print("Computing tau = tuned - base in fp32...")
    delta, rows, summary = extract_task_vector(
        base,
        tuned,
        output_dtype=_OUTPUT_DTYPES[args.output_dtype],
        near_zero_threshold=args.near_zero_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    save_file(
        delta,
        str(temporary),
        metadata={
            "format": "task_vector",
            "formula": "tuned-base",
            "base": args.base,
            "tuned": args.tuned,
            "output_dtype": args.output_dtype,
        },
    )
    temporary.replace(args.output)

    stats_path = args.output.with_suffix(".stats.csv")
    layers_path = args.output.with_suffix(".layers.csv")
    summary_path = args.output.with_suffix(".summary.json")
    _write_csv(stats_path, rows)
    layer_rows = _layer_rows(rows)
    _write_csv(layers_path, layer_rows)
    summary.update(
        {
            "base": args.base,
            "tuned": args.tuned,
            "task_vector": str(args.output),
            "parameter_stats": str(stats_path),
            "layer_stats": str(layers_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved task vector: {args.output}")
    print(f"Saved per-parameter norms: {stats_path}")
    print(f"Saved per-layer norms: {layers_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Total Frobenius norm: {summary['total_norm_fro']:.8g}")
    print(f"Nonzero tensors: {summary['nonzero_tensor_count']}/{summary['tensor_count']}")
    print(f"Near-zero tensors: {summary['near_zero_tensor_count']}/{summary['tensor_count']}")


if __name__ == "__main__":
    main()
