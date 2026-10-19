"""Preflight model/config audit for cross-family decoder THESEUS experiments.

This command intentionally instantiates the source architecture with empty (meta)
weights when printing its module tree.  It therefore does not download checkpoint
weights or consume enough RAM/VRAM to hold a 1B model merely to inspect names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


_SHAPE_FIELDS = ("num_hidden_layers", "hidden_size", "intermediate_size")
_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    *_SHAPE_FIELDS,
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)
_INTERESTING_MODULE_PARTS = (
    "attn",
    "attention",
    "mlp",
    "feed_forward",
    "ffn",
    "norm",
)


def _config_summary(config: Any, checkpoint: str) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        **{field: getattr(config, field, None) for field in _CONFIG_FIELDS},
    }


def _same_architecture(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = (
        "model_type",
        *_SHAPE_FIELDS,
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _shape_matches(summary: dict[str, Any], expected: tuple[int, int, int]) -> bool:
    actual = tuple(summary.get(field) for field in _SHAPE_FIELDS)
    return actual == expected


def _first_layer_prefix(model: Any) -> str | None:
    """Find the most plausible decoder-layer ModuleList and return its first-layer path."""
    candidates: list[tuple[int, str]] = []
    expected_depth = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    for name, module in model.named_modules():
        if module.__class__.__name__ != "ModuleList" or not name:
            continue
        length = len(module)
        if not length:
            continue
        score = 0
        if expected_depth and length == expected_depth:
            score += 10
        if any(part in name.lower() for part in ("layer", "block", "transformer")):
            score += 5
        candidates.append((score, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return f"{candidates[0][1]}.0"


def _module_tree(checkpoint: str, *, trust_remote_code: bool) -> tuple[str | None, list[str]]:
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    prefix = _first_layer_prefix(model)
    if prefix is None:
        return None, []

    rows: list[str] = []
    for name, module in model.named_modules():
        if name != prefix and not name.startswith(prefix + "."):
            continue
        relative = name.removeprefix(prefix).lstrip(".") or "<layer>"
        if relative == "<layer>" or any(part in relative.lower() for part in _INTERESTING_MODULE_PARTS):
            rows.append(f"{name}: {type(module).__name__}")
    return prefix, rows


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n===== {label} =====")
    for field in ("checkpoint", *_CONFIG_FIELDS):
        print(f"{field}: {summary.get(field)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit source-base/source-tuned/target shapes and source decoder module names before THESEUS."
    )
    parser.add_argument("--source-base", required=True, help="OLMo base checkpoint ID or local path.")
    parser.add_argument("--source-tuned", required=True, help="Matching OLMo RLVR checkpoint ID or local path.")
    parser.add_argument(
        "--target",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Target checkpoint ID or local path.",
    )
    parser.add_argument("--expected-layers", type=int, default=16)
    parser.add_argument("--expected-hidden-size", type=int, default=2048)
    parser.add_argument("--expected-intermediate-size", type=int, default=8192)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-module-tree",
        action="store_true",
        help="Only inspect configs; do not instantiate the source architecture with empty weights.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path for the structured audit result.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    refs = {
        "source_base": args.source_base,
        "source_tuned": args.source_tuned,
        "target": args.target,
    }
    summaries: dict[str, dict[str, Any]] = {}
    for label, checkpoint in refs.items():
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=args.trust_remote_code)
        summaries[label] = _config_summary(config, checkpoint)
        _print_summary(label, summaries[label])

    expected = (args.expected_layers, args.expected_hidden_size, args.expected_intermediate_size)
    checks = {
        "source_base_expected_shape": _shape_matches(summaries["source_base"], expected),
        "source_tuned_expected_shape": _shape_matches(summaries["source_tuned"], expected),
        "target_expected_shape": _shape_matches(summaries["target"], expected),
        "source_base_tuned_same_architecture": _same_architecture(
            summaries["source_base"], summaries["source_tuned"]
        ),
    }

    module_prefix: str | None = None
    module_rows: list[str] = []
    if not args.skip_module_tree:
        module_prefix, module_rows = _module_tree(
            args.source_base,
            trust_remote_code=args.trust_remote_code,
        )
        print("\n===== source base first-layer module tree (empty weights) =====")
        print(f"detected_layer_prefix: {module_prefix}")
        if module_rows:
            print("\n".join(module_rows))
        else:
            print("No decoder-layer module tree detected.")
        checks["source_layer_detected"] = module_prefix is not None

    print("\n===== preflight checks =====")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    payload = {
        "models": summaries,
        "expected_shape": {
            "num_hidden_layers": expected[0],
            "hidden_size": expected[1],
            "intermediate_size": expected[2],
        },
        "checks": checks,
        "source_first_layer_prefix": module_prefix,
        "source_first_layer_modules": module_rows,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nSaved JSON audit: {args.json_output}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
