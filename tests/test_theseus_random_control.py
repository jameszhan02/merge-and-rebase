import pytest
import torch

from merge_and_rebase.eval.diagnostics.theseus_random_control import (
    _dtype_matched_candidate,
    _norm_matched_gaussian,
)


def test_norm_matched_gaussian_is_deterministic_and_matches_frobenius_norm() -> None:
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    first = _norm_matched_gaussian(reference, generator=torch.Generator().manual_seed(7))
    again = _norm_matched_gaussian(reference, generator=torch.Generator().manual_seed(7))
    other = _norm_matched_gaussian(reference, generator=torch.Generator().manual_seed(8))

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(
        torch.linalg.vector_norm(reference).item(), rel=1e-6
    )


def test_norm_matched_gaussian_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        _norm_matched_gaussian(torch.zeros(2, 2), generator=torch.Generator().manual_seed(0))


def test_dtype_matched_candidate_matches_fp32_target() -> None:
    base = torch.ones(2, 2)
    direction = torch.full((2, 2), 0.5)
    candidate, effective_norm, scale = _dtype_matched_candidate(
        base,
        direction,
        alpha=0.5,
        target_effective_norm=0.5,
        dtype=torch.float32,
    )
    assert effective_norm == pytest.approx(0.5)
    assert scale == pytest.approx(1.0)
    assert torch.equal(candidate, torch.full((2, 2), 1.25))
