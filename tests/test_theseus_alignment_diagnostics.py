from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics.theseus_alignment import (
    fidelity_score,
    functional_gram,
    norm_ratio,
    procrustes_map,
    rank_at_variance,
    recon_error,
)


def test_recon_error_is_zero_when_b_equals_a_times_identity_map() -> None:
    torch.manual_seed(0)
    a = torch.randn(50, 4)
    b = a.clone()  # perfectly aligned, identity map
    t = torch.eye(4)
    a_gram = a.T @ a
    b_gram = b.T @ b
    cross_cov = a.T @ b
    assert recon_error(a_gram, b_gram, cross_cov, t) == pytest.approx(0.0, abs=1e-6)


def test_recon_error_is_positive_and_bounded_below_for_misaligned_spaces() -> None:
    torch.manual_seed(1)
    a = torch.randn(50, 4)
    b = torch.randn(50, 4)  # unrelated to a
    t = torch.eye(4)
    err = recon_error(a.T @ a, b.T @ b, a.T @ b, t)
    assert err > 0.1


def test_procrustes_map_recovers_a_pure_rotation() -> None:
    torch.manual_seed(2)
    a = torch.randn(200, 5)
    # Build an orthogonal (rotation) matrix via QR.
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    b = a @ q
    cross_cov = a.T @ b
    t = procrustes_map(cross_cov)
    recovered = recon_error(a.T @ a, b.T @ b, cross_cov, t)
    assert recovered == pytest.approx(0.0, abs=1e-4)


def test_rank_at_variance_is_one_for_rank_one_cross_covariance() -> None:
    u = torch.randn(10, 1)
    v = torch.randn(10, 1)
    cross_cov = u @ v.T  # rank 1
    assert rank_at_variance(cross_cov, threshold=0.9) == 1


def test_rank_at_variance_is_full_for_isotropic_identity() -> None:
    cross_cov = torch.eye(8)  # all singular values equal: needs ~90% of all 8
    r = rank_at_variance(cross_cov, threshold=0.9)
    assert r >= 7


def test_norm_ratio_is_one_for_orthogonal_transforms() -> None:
    torch.manual_seed(3)
    tau_a = torch.randn(6, 4)
    t_in, _ = torch.linalg.qr(torch.randn(4, 4))
    t_out, _ = torch.linalg.qr(torch.randn(6, 6))
    assert norm_ratio(tau_a, t_in, t_out) == pytest.approx(1.0, abs=1e-5)


def test_norm_ratio_scales_with_non_orthogonal_transform() -> None:
    tau_a = torch.eye(3)
    t_in = torch.eye(3) * 2.0  # not orthogonal: scales norm by 2
    t_out = torch.eye(3)
    assert norm_ratio(tau_a, t_in, t_out) == pytest.approx(2.0, abs=1e-5)


def test_functional_gram_shape_is_m_by_m() -> None:
    delta_h = torch.randn(12, 7)
    h_out = torch.randn(12, 7)
    g = functional_gram(delta_h, h_out)
    assert g.shape == (12, 12)


def test_functional_gram_rejects_mismatched_sample_axis() -> None:
    with pytest.raises(ValueError, match="sample axis"):
        functional_gram(torch.randn(12, 7), torch.randn(10, 7))


def test_fidelity_score_is_one_when_gram_matrices_match() -> None:
    g = torch.randn(9, 9)
    assert fidelity_score(g, g.clone()) == pytest.approx(1.0, abs=1e-6)


def test_fidelity_score_drops_as_gram_matrices_diverge() -> None:
    torch.manual_seed(4)
    g_a = torch.randn(9, 9)
    g_b_close = g_a + 0.01 * torch.randn(9, 9)
    g_b_far = torch.randn(9, 9) * 10.0
    assert fidelity_score(g_a, g_b_close) > fidelity_score(g_a, g_b_far)
