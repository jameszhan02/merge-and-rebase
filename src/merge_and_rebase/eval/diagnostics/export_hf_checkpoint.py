"""Export a complete rebased ``state_dict`` as a Hugging Face checkpoint directory."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


_OUTPUT_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def load_complete_state(path: Path) -> dict[str, torch.Tensor]:
    """Load and validate the direct state-dict format emitted by ``--save-merged``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError(f"Merged state at '{path}' is not a non-empty state_dict mapping.")
    bad_keys = [key for key in payload if not isinstance(key, str)]
    bad_values = [key for key, value in payload.items() if not isinstance(value, torch.Tensor)]
    if bad_keys or bad_values:
        raise TypeError(
            "Merged state must map string keys directly to tensors; "
            f"bad_keys={bad_keys[:5]}, non_tensor_values={bad_values[:5]}"
        )
    return dict(payload)


def export_hf_checkpoint(
    *,
    base: str,
    state_path: Path,
    output_dir: Path,
    output_dtype: torch.dtype = torch.float32,
    max_shard_size: str = "2GB",
    safe_serialization: bool = True,
    save_tokenizer: bool = True,
    tokenizer_source: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load a rebased full state strictly into the target architecture and export it."""
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: '{output_dir}'. Choose a new path so stale shards cannot survive."
        )
    if not state_path.is_file():
        raise FileNotFoundError(f"Merged state file not found: '{state_path}'")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    state = load_complete_state(state_path)
    print(f"Loading target architecture from: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=output_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Merged state does not exactly match the target architecture: "
            f"missing={result.missing_keys[:20]}, unexpected={result.unexpected_keys[:20]}"
        )
    del state

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        print(f"Saving Hugging Face model to temporary directory: {temporary_dir}")
        model.save_pretrained(
            temporary_dir,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        if save_tokenizer:
            tokenizer_ref = tokenizer_source or base
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_ref,
                trust_remote_code=trust_remote_code,
            )
            tokenizer.save_pretrained(temporary_dir)

        manifest = {
            "format": "huggingface_pretrained",
            "base": base,
            "merged_state": str(state_path),
            "output_dtype": str(output_dtype).removeprefix("torch."),
            "safe_serialization": bool(safe_serialization),
            "max_shard_size": max_shard_size,
            "tokenizer_source": (tokenizer_source or base) if save_tokenizer else None,
        }
        (temporary_dir / "theseus_export.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print("Strict state_dict validation: PASS")
    print(f"Saved Hugging Face checkpoint: {output_dir}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a complete .pt state_dict produced by LLM rebase as a standard Hugging Face directory."
    )
    parser.add_argument("--base", required=True, help="Target HF checkpoint used as the architecture/config template.")
    parser.add_argument("--state", required=True, type=Path, help="Complete merged .pt state_dict from --save-merged.")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; it must not already exist.")
    parser.add_argument("--output-dtype", choices=sorted(_OUTPUT_DTYPES), default="fp32")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write safetensors shards (default); use --no-safe-serialization for pytorch_model.bin.",
    )
    parser.add_argument(
        "--save-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy tokenizer files into the exported directory (default).",
    )
    parser.add_argument("--tokenizer-source", default=None, help="Tokenizer checkpoint; defaults to --base.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _parser().parse_args()
    export_hf_checkpoint(
        base=args.base,
        state_path=args.state,
        output_dir=args.output,
        output_dtype=_OUTPUT_DTYPES[args.output_dtype],
        max_shard_size=args.max_shard_size,
        safe_serialization=args.safe_serialization,
        save_tokenizer=args.save_tokenizer,
        tokenizer_source=args.tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
