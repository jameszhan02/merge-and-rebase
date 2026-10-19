"""Build support- and per-tensor-norm-matched random controls for THESEUS exports."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


def _alpha_name(alpha: float) -> str:
    return f"alpha_{format(float(alpha), '.12g')}"


def _norm_matched_gaussian(
    reference_delta: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return an FP32 Gaussian direction with the reference tensor's Frobenius norm."""
    reference = reference_delta.detach().cpu().float()
    target_norm = torch.linalg.vector_norm(reference)
    if not torch.isfinite(target_norm) or float(target_norm) <= 0.0:
        raise ValueError("Reference delta must have a finite, non-zero Frobenius norm.")
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    noise_norm = torch.linalg.vector_norm(noise)
    if not torch.isfinite(noise_norm) or float(noise_norm) <= 0.0:
        raise RuntimeError("Generated a non-finite or zero random direction.")
    return noise.mul_(target_norm / noise_norm)


def _dtype_matched_candidate(
    base: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: float,
    target_effective_norm: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """Match the effective post-cast norm, compensating for BF16/FP16 rounding."""
    base_fp32 = base.detach().cpu().float()
    if alpha == 0.0 or target_effective_norm == 0.0:
        candidate = base_fp32.to(dtype=dtype)
        return candidate, 0.0, 1.0

    scale = 1.0
    best: tuple[torch.Tensor, float, float] | None = None
    best_error = float("inf")
    for _ in range(8):
        candidate = base_fp32.add(direction, alpha=alpha * scale).to(dtype=dtype)
        effective = float(torch.linalg.vector_norm(candidate.float() - base_fp32))
        error = abs(effective - target_effective_norm)
        if error < best_error:
            best = (candidate, effective, scale)
            best_error = error
        if effective == 0.0:
            scale *= 2.0
        else:
            scale *= target_effective_norm / effective
        if not math.isfinite(scale) or scale <= 0.0:
            break
    if best is None:
        raise RuntimeError("Could not materialize a finite dtype-matched random candidate.")
    return best


def _load_models(base: str, reference: str, *, trust_remote_code: bool) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM

    print(f"Loading alpha=0/base checkpoint: {base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    ).cpu()
    print(f"Loading alpha=1/reference checkpoint: {reference}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference,
        dtype="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    ).cpu()
    return base_model, reference_model


def build_random_controls(
    *,
    base: str,
    reference: str,
    output_root: Path,
    alphas: list[float],
    seeds: list[int],
    expected_changed_tensors: int | None = 64,
    max_shard_size: str = "2GB",
    tokenizer_source: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Derive the effective THESEUS delta and export matched Gaussian controls."""
    alphas = [float(value) for value in alphas]
    seeds = [int(value) for value in seeds]
    if not alphas or not seeds:
        raise ValueError("At least one alpha and one seed are required.")
    if any(not math.isfinite(value) or value < 0.0 for value in alphas):
        raise ValueError("Alphas must be finite and non-negative.")
    if len(set(alphas)) != len(alphas) or len(set(seeds)) != len(seeds):
        raise ValueError("Alphas and seeds must not contain duplicates.")

    output_dirs = [
        output_root / f"seed_{seed}" / _alpha_name(alpha)
        for seed in seeds
        for alpha in alphas
    ]
    collisions = [path for path in output_dirs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Output directories already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in collisions)
        )

    base_model, reference_model = _load_models(
        base,
        reference,
        trust_remote_code=trust_remote_code,
    )
    base_state = base_model.state_dict()
    reference_state = reference_model.state_dict()
    if set(base_state) != set(reference_state):
        missing = sorted(set(base_state) - set(reference_state))
        unexpected = sorted(set(reference_state) - set(base_state))
        raise ValueError(
            f"Checkpoint keyspaces differ: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    reference_deltas: dict[str, torch.Tensor] = {}
    base_changed: dict[str, torch.Tensor] = {}
    for key in sorted(base_state):
        base_tensor = base_state[key].detach().cpu()
        reference_tensor = reference_state[key].detach().cpu()
        if tuple(base_tensor.shape) != tuple(reference_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {key}: base={tuple(base_tensor.shape)}, "
                f"reference={tuple(reference_tensor.shape)}"
            )
        if not torch.equal(base_tensor, reference_tensor):
            delta = reference_tensor.float() - base_tensor.float()
            if float(torch.linalg.vector_norm(delta)) > 0.0:
                reference_deltas[key] = delta
                base_changed[key] = base_tensor.clone()

    changed_count = len(reference_deltas)
    print(f"Detected changed tensors in alpha=1 versus alpha=0: {changed_count}")
    if expected_changed_tensors is not None and changed_count != expected_changed_tensors:
        raise ValueError(
            f"Expected {expected_changed_tensors} changed tensors, found {changed_count}. "
            "Check that --base is alpha_0 and --reference is alpha_1, or pass "
            "--expected-changed-tensors 0 to disable this guard."
        )
    if changed_count == 0:
        raise ValueError("Reference checkpoint is identical to the base checkpoint.")

    parameter_map = dict(base_model.named_parameters())
    missing_parameters = [key for key in reference_deltas if key not in parameter_map]
    if missing_parameters:
        raise ValueError(
            "Changed state entries are not model parameters: " + ", ".join(missing_parameters[:20])
        )

    del reference_model, reference_state, base_state

    from transformers import AutoTokenizer

    tokenizer_ref = tokenizer_source or base
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    reference_norms = {
        key: float(torch.linalg.vector_norm(value)) for key, value in reference_deltas.items()
    }

    for seed in seeds:
        print(f"Generating per-tensor norm-matched Gaussian direction (seed={seed})...")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_direction = {
            key: _norm_matched_gaussian(reference_deltas[key], generator=generator)
            for key in sorted(reference_deltas)
        }

        for alpha in alphas:
            output_dir = output_root / f"seed_{seed}" / _alpha_name(alpha)
            print(f"Materializing random control seed={seed}, alpha={alpha:.12g}...")
            effective_sq = 0.0
            per_tensor_effective_norm: dict[str, float] = {}
            per_tensor_rounding_scale: dict[str, float] = {}
            with torch.no_grad():
                for key in sorted(random_direction):
                    parameter = parameter_map[key]
                    candidate, effective_norm, rounding_scale = _dtype_matched_candidate(
                        base_changed[key],
                        random_direction[key],
                        alpha=alpha,
                        target_effective_norm=abs(alpha) * reference_norms[key],
                        dtype=parameter.dtype,
                    )
                    parameter.copy_(candidate)
                    per_tensor_effective_norm[key] = effective_norm
                    per_tensor_rounding_scale[key] = rounding_scale
                    effective_sq += effective_norm * effective_norm

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
            )
            manifest = {
                "format": "theseus_norm_matched_random_control",
                "base": base,
                "reference": reference,
                "reference_delta": "reference - base",
                "distribution": "gaussian",
                "normalization": "per_tensor_frobenius",
                "seed": seed,
                "alpha": alpha,
                "changed_tensor_count": changed_count,
                "reference_tensor_norms": reference_norms,
                "effective_tensor_norms_after_dtype_cast": per_tensor_effective_norm,
                "rounding_compensation_scales": per_tensor_rounding_scale,
                "effective_global_norm_after_dtype_cast": math.sqrt(effective_sq),
                "model_dtype": str(next(base_model.parameters()).dtype).removeprefix("torch."),
            }
            try:
                base_model.save_pretrained(
                    temporary_dir,
                    safe_serialization=True,
                    max_shard_size=max_shard_size,
                )
                tokenizer.save_pretrained(temporary_dir)
                (temporary_dir / "theseus_random_control.json").write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_dir.replace(output_dir)
            except BaseException:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise
            print(f"Saved random-control HF checkpoint: {output_dir}")
            records.append({
                "seed": seed,
                "alpha": alpha,
                "path": str(output_dir),
                "effective_global_norm_after_dtype_cast": math.sqrt(effective_sq),
            })

        del random_direction

    summary = {
        "base": base,
        "reference": reference,
        "changed_tensor_count": changed_count,
        "alphas": alphas,
        "seeds": seeds,
        "records": records,
    }
    (output_root / "random_control_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary: {output_root / 'random_control_summary.json'}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Gaussian random-control HF checkpoints matched to the support and "
            "per-tensor Frobenius norms of an exported THESEUS alpha=1 update."
        )
    )
    parser.add_argument("--base", required=True, help="THESEUS alpha_0 HF directory.")
    parser.add_argument("--reference", required=True, help="THESEUS alpha_1 HF directory.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alphas", required=True, type=float, nargs="+")
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument(
        "--expected-changed-tensors",
        type=int,
        default=64,
        help="Safety check (default: 64); pass 0 to disable.",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_random_controls(
        base=args.base,
        reference=args.reference,
        output_root=args.output_root,
        alphas=args.alphas,
        seeds=args.seeds,
        expected_changed_tensors=(
            None if args.expected_changed_tensors == 0 else args.expected_changed_tensors
        ),
        max_shard_size=args.max_shard_size,
        tokenizer_source=args.tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
