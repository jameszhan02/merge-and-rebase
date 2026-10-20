"""Audit where two shape-compatible Hugging Face checkpoints differ.

This diagnostic intentionally does not save a task vector.  It reports both
absolute update energy and size-normalized statistics: a raw Frobenius norm
alone systematically makes large MLP matrices look more important.
"""

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


_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)(?:\.|$)")


def _layer_index(key: str) -> int | None:
    match = _LAYER_PATTERN.search(key)
    return int(match.group(1)) if match else None


def _component(key: str) -> str:
    known = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "post_feedforward_layernorm",
    )
    for name in known:
        if f".{name}." in f".{key}.":
            return name
    if "embed_tokens" in key:
        return "embed_tokens"
    if key.startswith("lm_head.") or ".lm_head." in key:
        return "lm_head"
    if key in {"model.norm.weight", "transformer.norm.weight"}:
        return "final_norm"
    return "other"


def _aggregate(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], dict[str, float | int]] = defaultdict(
        lambda: {
            "delta_sq": 0.0,
            "base_sq": 0.0,
            "numel": 0,
            "changed_numel": 0,
            "tensor_count": 0,
            "max_abs_delta": 0.0,
        }
    )
    for row in rows:
        group_key = tuple(row[field] for field in fields)
        values = totals[group_key]
        values["delta_sq"] = float(values["delta_sq"]) + float(row["delta_norm_fro"]) ** 2
        values["base_sq"] = float(values["base_sq"]) + float(row["base_norm_fro"]) ** 2
        values["numel"] = int(values["numel"]) + int(row["numel"])
        values["changed_numel"] = int(values["changed_numel"]) + int(row["changed_numel"])
        values["tensor_count"] = int(values["tensor_count"]) + 1
        values["max_abs_delta"] = max(float(values["max_abs_delta"]), float(row["max_abs_delta"]))

    total_delta_sq = sum(float(values["delta_sq"]) for values in totals.values())
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(totals.items(), key=lambda item: tuple(str(part) for part in item[0])):
        delta_norm = math.sqrt(float(values["delta_sq"]))
        base_norm = math.sqrt(float(values["base_sq"]))
        numel = int(values["numel"])
        output.append(
            {
                **dict(zip(fields, group_key, strict=True)),
                "tensor_count": int(values["tensor_count"]),
                "numel": numel,
                "delta_norm_fro": delta_norm,
                "base_norm_fro": base_norm,
                "relative_norm": delta_norm / base_norm if base_norm else math.inf,
                "delta_rms": delta_norm / math.sqrt(numel) if numel else 0.0,
                "changed_fraction": int(values["changed_numel"]) / numel if numel else 0.0,
                "max_abs_delta": float(values["max_abs_delta"]),
                "update_energy_fraction": float(values["delta_sq"]) / total_delta_sq if total_delta_sq else 0.0,
            }
        )
    return output


def compare_states(
    base: dict[str, torch.Tensor], tuned: dict[str, torch.Tensor]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return parameter, layer and component rows plus an aggregate summary."""
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

    rows: list[dict[str, Any]] = []
    for key in sorted(base_float):
        base_tensor = base_float[key].detach().cpu()
        tuned_tensor = tuned_float[key].detach().cpu()
        if base_tensor.shape != tuned_tensor.shape:
            raise ValueError(f"Shape mismatch for {key!r}: {tuple(base_tensor.shape)} != {tuple(tuned_tensor.shape)}")
        base_fp32 = base_tensor.float()
        delta = tuned_tensor.float() - base_fp32
        if not torch.isfinite(delta).all():
            raise ValueError(f"NaN or Inf in delta tensor {key!r}")
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        base_norm = float(torch.linalg.vector_norm(base_fp32).item())
        numel = delta.numel()
        changed_numel = int(torch.count_nonzero(delta).item())
        rows.append(
            {
                "layer_idx": _layer_index(key) if _layer_index(key) is not None else "outside",
                "component": _component(key),
                "weight_name": key,
                "shape": "x".join(str(dim) for dim in delta.shape),
                "numel": numel,
                "delta_norm_fro": delta_norm,
                "base_norm_fro": base_norm,
                "relative_norm": delta_norm / base_norm if base_norm else math.inf,
                "delta_rms": delta_norm / math.sqrt(numel) if numel else 0.0,
                "changed_numel": changed_numel,
                "changed_fraction": changed_numel / numel if numel else 0.0,
                "max_abs_delta": float(delta.abs().max().item()) if numel else 0.0,
            }
        )

    layer_rows = _aggregate(rows, ("layer_idx",))
    component_rows = _aggregate(rows, ("component",))
    total_delta_sq = sum(float(row["delta_norm_fro"]) ** 2 for row in rows)
    for row in rows:
        row["update_energy_fraction"] = (
            float(row["delta_norm_fro"]) ** 2 / total_delta_sq if total_delta_sq else 0.0
        )
    total_base_sq = sum(float(row["base_norm_fro"]) ** 2 for row in rows)
    total_numel = sum(int(row["numel"]) for row in rows)
    total_changed = sum(int(row["changed_numel"]) for row in rows)
    summary = {
        "tensor_count": len(rows),
        "numel": total_numel,
        "changed_numel": total_changed,
        "changed_fraction": total_changed / total_numel if total_numel else 0.0,
        "delta_norm_fro": math.sqrt(total_delta_sq),
        "base_norm_fro": math.sqrt(total_base_sq),
        "relative_norm": math.sqrt(total_delta_sq / total_base_sq) if total_base_sq else math.inf,
        "delta_rms": math.sqrt(total_delta_sq / total_numel) if total_numel else 0.0,
    }
    return rows, layer_rows, component_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    path: Path,
    parameter_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    def fmt(value: Any) -> str:
        return f"{float(value):.6g}"

    lines = [
        "# Checkpoint difference report",
        "",
        f"- Base: `{summary['base']}`",
        f"- Tuned: `{summary['tuned']}`",
        f"- Global relative norm: `{fmt(summary['relative_norm'])}`",
        f"- Global delta RMS: `{fmt(summary['delta_rms'])}`",
        f"- Changed elements: `{summary['changed_numel']}/{summary['numel']}` "
        f"(`{float(summary['changed_fraction']):.4%}`)",
        "",
        "Relative norm and delta RMS are size-normalized. Update energy fraction is the share of squared "
        "delta norm; it can favor large tensors and should not be interpreted alone.",
        "",
        "## Decoder layer summary",
        "",
        "| Layer | Relative norm | Delta RMS | Update energy | Changed |",
        "|---:|---:|---:|---:|---:|",
    ]
    decoder_layers = [row for row in layer_rows if isinstance(row["layer_idx"], int)]
    for row in sorted(decoder_layers, key=lambda item: int(item["layer_idx"])):
        lines.append(
            f"| {row['layer_idx']} | {fmt(row['relative_norm'])} | {fmt(row['delta_rms'])} | "
            f"{float(row['update_energy_fraction']):.3%} | {float(row['changed_fraction']):.3%} |"
        )

    lines.extend(
        [
            "",
            "## Component summary",
            "",
            "| Component | Relative norm | Delta RMS | Update energy | Changed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(component_rows, key=lambda item: float(item["relative_norm"]), reverse=True):
        lines.append(
            f"| `{row['component']}` | {fmt(row['relative_norm'])} | {fmt(row['delta_rms'])} | "
            f"{float(row['update_energy_fraction']):.3%} | {float(row['changed_fraction']):.3%} |"
        )

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    outside: list[dict[str, Any]] = []
    for row in parameter_rows:
        if isinstance(row["layer_idx"], int):
            by_layer[int(row["layer_idx"])].append(row)
        else:
            outside.append(row)
    lines.extend(["", "## Per-layer parameter detail"])
    for layer_idx in sorted(by_layer):
        lines.extend(
            [
                "",
                f"### Layer {layer_idx}",
                "",
                "| Weight | Shape | Relative norm | Delta RMS | Update energy | Changed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(by_layer[layer_idx], key=lambda item: str(item["component"])):
            lines.append(
                f"| `{row['component']}` | `{row['shape']}` | {fmt(row['relative_norm'])} | "
                f"{fmt(row['delta_rms'])} | {float(row['update_energy_fraction']):.3%} | "
                f"{float(row['changed_fraction']):.3%} |"
            )

    if outside:
        lines.extend(
            [
                "",
                "## Parameters outside decoder layers",
                "",
                "| Weight | Shape | Relative norm | Delta RMS | Update energy | Changed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in outside:
            lines.append(
                f"| `{row['weight_name']}` | `{row['shape']}` | {fmt(row['relative_norm'])} | "
                f"{fmt(row['delta_rms'])} | {float(row['update_energy_fraction']):.3%} | "
                f"{float(row['changed_fraction']):.3%} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, layer_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    decoder_rows = [row for row in layer_rows if isinstance(row["layer_idx"], int)]
    decoder_rows.sort(key=lambda row: int(row["layer_idx"]))
    if decoder_rows:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        layers = [int(row["layer_idx"]) for row in decoder_rows]
        axes[0].bar(layers, [float(row["relative_norm"]) for row in decoder_rows])
        axes[0].set(title="Base to tuned: relative update by layer", xlabel="Decoder layer", ylabel="||delta|| / ||base||")
        axes[1].bar(layers, [float(row["update_energy_fraction"]) for row in decoder_rows])
        axes[1].set(title="Share of decoder-layer update energy", xlabel="Decoder layer", ylabel="Fraction")
        fig.savefig(output_dir / "layers.png", dpi=180)
        plt.close(fig)

    ordered = sorted(component_rows, key=lambda row: float(row["relative_norm"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(ordered))), constrained_layout=True)
    ax.barh([str(row["component"]) for row in reversed(ordered)], [float(row["relative_norm"]) for row in reversed(ordered)])
    ax.set(title="Base to tuned: relative update by component", xlabel="||delta|| / ||base||")
    fig.savefig(output_dir / "components.png", dpi=180)
    plt.close(fig)


def _load_state(ref: str, *, trust_remote_code: bool) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ref,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if isinstance(value, torch.Tensor)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure where two shape-compatible HF checkpoints differ.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading base checkpoint on CPU: {args.base}")
    base = _load_state(args.base, trust_remote_code=args.trust_remote_code)
    print(f"Loading tuned checkpoint on CPU: {args.tuned}")
    tuned = _load_state(args.tuned, trust_remote_code=args.trust_remote_code)
    print("Computing checkpoint differences in fp32...")
    parameter_rows, layer_rows, component_rows, summary = compare_states(base, tuned)
    summary.update({"base": args.base, "tuned": args.tuned})

    _write_csv(args.output_dir / "parameters.csv", parameter_rows)
    _write_csv(args.output_dir / "layers.csv", layer_rows)
    _write_csv(args.output_dir / "components.csv", component_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(args.output_dir / "report.md", parameter_rows, layer_rows, component_rows, summary)
    if args.plots:
        _write_plots(args.output_dir, layer_rows, component_rows)

    print(f"Saved audit to: {args.output_dir}")
    print(f"Global relative norm: {summary['relative_norm']:.8g}")
    print(f"Changed elements: {summary['changed_numel']}/{summary['numel']} ({summary['changed_fraction']:.4%})")
    print("Largest layers by relative norm:")
    decoder_rows = [row for row in layer_rows if isinstance(row["layer_idx"], int)]
    for row in sorted(decoder_rows, key=lambda item: float(item["relative_norm"]), reverse=True)[:5]:
        print(
            f"  layer {row['layer_idx']}: relative_norm={float(row['relative_norm']):.6g} "
            f"energy_fraction={float(row['update_energy_fraction']):.3%}"
        )
    print("Largest components by relative norm:")
    for row in sorted(component_rows, key=lambda item: float(item["relative_norm"]), reverse=True)[:8]:
        print(
            f"  {row['component']}: relative_norm={float(row['relative_norm']):.6g} "
            f"energy_fraction={float(row['update_energy_fraction']):.3%}"
        )


if __name__ == "__main__":
    main()
