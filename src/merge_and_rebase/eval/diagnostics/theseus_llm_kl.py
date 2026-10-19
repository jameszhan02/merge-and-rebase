"""Paired answer-token KL and gold-NLL diagnostics for local causal-LM checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch


ALPACA_PREFIX = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"
)

_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _distribution_metrics(
    reference_log_probs: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    gold_tokens: torch.Tensor,
) -> dict[str, float]:
    """Return sums over answer-token positions; callers divide by ``token_count``."""
    if reference_log_probs.shape != candidate_log_probs.shape:
        raise ValueError("Reference and candidate log-probability tensors must have the same shape.")
    if reference_log_probs.ndim != 2 or gold_tokens.ndim != 1:
        raise ValueError("Expected [tokens, vocab] log-probabilities and [tokens] gold ids.")
    if reference_log_probs.shape[0] != gold_tokens.shape[0]:
        raise ValueError("Gold-token count does not match the log-probability positions.")

    reference_probs = reference_log_probs.exp()
    kl_by_token = (
        reference_probs * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1)
    reference_gold = reference_log_probs.gather(1, gold_tokens[:, None]).squeeze(1)
    candidate_gold = candidate_log_probs.gather(1, gold_tokens[:, None]).squeeze(1)
    top1_flips = reference_log_probs.argmax(dim=-1).ne(candidate_log_probs.argmax(dim=-1))
    return {
        "token_count": float(gold_tokens.numel()),
        "kl_sum": float(kl_by_token.sum()),
        "reference_nll_sum": float(-reference_gold.sum()),
        "candidate_nll_sum": float(-candidate_gold.sum()),
        "top1_flip_count": float(top1_flips.sum()),
    }


def _answer_batch(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
    max_length: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
    prefix = ALPACA_PREFIX.format(question=question)
    full_text = prefix + answer
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0]
    inputs = {key: value.to(device) for key, value in encoded.items()}
    if inputs["input_ids"].shape[1] < 2:
        return None

    # Token j is predicted by logits at j-1. Include tokens that overlap the
    # answer substring and exclude zero-width special tokens.
    label_offsets = offsets[1:]
    answer_mask = (label_offsets[:, 1] > len(prefix)) & (
        label_offsets[:, 1] > label_offsets[:, 0]
    )
    if not bool(answer_mask.any()):
        return None
    return inputs, answer_mask.to(device)


def _selected_log_probs(
    model: Any,
    inputs: dict[str, torch.Tensor],
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(**inputs, use_cache=False).logits[0, :-1]
    selected = logits[answer_mask].float()
    del logits
    return torch.log_softmax(selected, dim=-1)


def compare_candidate(
    *,
    reference_model: Any,
    candidate_model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_length: int,
    device: str,
    print_every: int,
) -> dict[str, Any]:
    totals = {
        "token_count": 0.0,
        "kl_sum": 0.0,
        "reference_nll_sum": 0.0,
        "candidate_nll_sum": 0.0,
        "top1_flip_count": 0.0,
    }
    rows: list[dict[str, Any]] = []
    skipped = 0

    with torch.inference_mode():
        for index, example in enumerate(examples):
            batch = _answer_batch(
                tokenizer,
                question=str(example["question"]),
                answer=str(example["answer"]),
                max_length=max_length,
                device=device,
            )
            if batch is None:
                skipped += 1
                continue
            inputs, answer_mask = batch
            gold_tokens = inputs["input_ids"][0, 1:][answer_mask]
            reference_log_probs = _selected_log_probs(reference_model, inputs, answer_mask)
            candidate_log_probs = _selected_log_probs(candidate_model, inputs, answer_mask)
            metrics = _distribution_metrics(
                reference_log_probs,
                candidate_log_probs,
                gold_tokens,
            )
            for key in totals:
                totals[key] += metrics[key]
            count = metrics["token_count"]
            rows.append({
                "example_index": int(example["_index"]),
                "answer_tokens": int(count),
                "kl_reference_to_candidate": metrics["kl_sum"] / count,
                "reference_gold_nll": metrics["reference_nll_sum"] / count,
                "candidate_gold_nll": metrics["candidate_nll_sum"] / count,
                "gold_nll_delta": (
                    metrics["candidate_nll_sum"] - metrics["reference_nll_sum"]
                ) / count,
                "top1_flip_rate": metrics["top1_flip_count"] / count,
            })
            del reference_log_probs, candidate_log_probs, inputs, answer_mask, gold_tokens
            if print_every > 0 and (index + 1) % print_every == 0:
                print(f"  processed {index + 1}/{len(examples)} examples")

    token_count = totals["token_count"]
    if token_count <= 0:
        raise RuntimeError("No answer tokens survived tokenization/truncation.")
    aggregate = {
        "examples_requested": len(examples),
        "examples_scored": len(rows),
        "examples_skipped": skipped,
        "answer_tokens": int(token_count),
        "mean_kl_reference_to_candidate": totals["kl_sum"] / token_count,
        "reference_gold_nll": totals["reference_nll_sum"] / token_count,
        "candidate_gold_nll": totals["candidate_nll_sum"] / token_count,
        "gold_nll_delta": (
            totals["candidate_nll_sum"] - totals["reference_nll_sum"]
        ) / token_count,
        "top1_flip_rate": totals["top1_flip_count"] / token_count,
    }
    return {"aggregate": aggregate, "per_example": rows}


def run_llm_kl(
    *,
    reference: str,
    candidates: list[str],
    output: Path,
    num_examples: int = 200,
    seed: int = 0,
    max_length: int = 512,
    device: str = "cuda",
    dtype: str = "bf16",
    dataset_path: str = "gsm8k",
    dataset_name: str = "main",
    dataset_split: str = "test",
    print_every: int = 10,
    trust_remote_code: bool = False,
    local_files_only: bool = True,
) -> dict[str, Any]:
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if num_examples <= 0 or max_length <= 1:
        raise ValueError("num_examples must be positive and max_length must be greater than 1.")
    if not candidates:
        raise ValueError("At least one candidate checkpoint is required.")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        reference,
        use_fast=True,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for answer-span offset mapping.")

    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(num_examples, len(indices))]
    examples = [{**dict(dataset[index]), "_index": index} for index in indices]
    print(
        f"Loaded {len(examples)} examples from {dataset_path}/{dataset_name}[{dataset_split}] "
        f"with seed={seed}."
    )

    model_dtype = _DTYPES[dtype]
    print(f"Loading KL reference model: {reference}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    ).to(device).eval()

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"\nLoading candidate: {candidate}")
        candidate_model = AutoModelForCausalLM.from_pretrained(
            candidate,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()
        if int(candidate_model.config.vocab_size) != int(reference_model.config.vocab_size):
            raise ValueError(
                f"Vocabulary size mismatch: reference={reference_model.config.vocab_size}, "
                f"candidate={candidate_model.config.vocab_size}"
            )
        result = compare_candidate(
            reference_model=reference_model,
            candidate_model=candidate_model,
            tokenizer=tokenizer,
            examples=examples,
            max_length=max_length,
            device=device,
            print_every=print_every,
        )
        result["candidate"] = candidate
        results.append(result)
        aggregate = result["aggregate"]
        print(
            "  mean_KL(base||candidate)="
            f"{aggregate['mean_kl_reference_to_candidate']:.8e}  "
            f"gold_NLL_delta={aggregate['gold_nll_delta']:+.8e}  "
            f"top1_flip_rate={aggregate['top1_flip_rate']:.6%}"
        )
        del candidate_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "kind": "causal_lm_answer_token_kl",
        "reference": reference,
        "prompt_template": ALPACA_PREFIX,
        "dataset": {
            "path": dataset_path,
            "name": dataset_name,
            "split": dataset_split,
            "num_examples": len(examples),
            "indices": indices,
            "seed": seed,
        },
        "max_length": max_length,
        "device": device,
        "dtype": dtype,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"\nSaved KL diagnostics: {output}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local causal-LM checkpoints on GSM8K answer-token KL and gold NLL."
    )
    parser.add_argument("--reference", required=True, help="Reference/base HF checkpoint.")
    parser.add_argument("--candidates", required=True, nargs="+", help="Candidate HF checkpoints.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--dataset-path", default="gsm8k")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_llm_kl(
        reference=args.reference,
        candidates=args.candidates,
        output=args.output,
        num_examples=args.num_examples,
        seed=args.seed,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        print_every=args.print_every,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
