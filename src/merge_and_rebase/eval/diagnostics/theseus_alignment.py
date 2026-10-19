"""Experiment-1-style alignment diagnostics for cross-family Theseus transport.

Pure, dependency-free math only -- no model loading, no hooks, no CLI. The functions here
read the same statistics ``rebase.methods.theseus.collect_activations`` already accumulates
(cross-covariance and, optionally, each side's own Gram matrix) plus raw per-sample output
activations, and turn them into the per-layer diagnostic numbers used to decide which layers
are worth transporting (Theseus Experiment 1 / Experiment 3 layer selection).

Provenance, so results aren't misattributed when writing this up:
  - ``norm_ratio`` is the paper's own identity (arXiv 2602.12952, Eq. 11): Procrustes transport
    preserves a task vector's Frobenius norm exactly. Any deviation beyond float error means the
    transform itself is wrong, not that the layer is a bad transport candidate.
  - ``functional_gram`` follows the paper's Eq. 3 definition of ``G_A = (delta_H_out) @ H_out.T``,
    an (M, M) Gram matrix over calibration samples (not over feature dimensions), where
    ``delta_H_out`` is the task vector's induced change to that layer's output activations.
    Comparing ``G_A`` (source) against the analogous ``G_B`` built from the *transported* vector
    on the target's own activations is how "did the task vector's functional effect survive
    transport" gets measured here.
  - ``fidelity_score`` (the scalar ``1 - ||G_A - G_B||_F^2 / ||G_A||_F^2`` reduction) and
    ``recon_error`` / ``rank_at_variance`` (CCA-style alignment-quality numbers from the
    cross-covariance SVD) are *not* named metrics in the paper -- they are this project's own
    operational read of "how well do the two layers' representations/effects line up",
    introduced because the paper doesn't ship a diagnostic suite for cross-family, non-isomorphic
    transport. Treat them as a defensible proxy, not a reproduction of a published number.
"""

from __future__ import annotations

import torch

from ...rebase.methods._shared import _compute_procrustes_map_from_cov


def procrustes_map(cross_cov: torch.Tensor, *, device: str = "cpu") -> torch.Tensor:
    """``T`` solving the orthogonal Procrustes problem for ``cross_cov = A^T B`` (A's domain -> B's domain)."""
    return _compute_procrustes_map_from_cov(cross_cov, device=device)


def recon_error(a_gram: torch.Tensor, b_gram: torch.Tensor, cross_cov: torch.Tensor, t: torch.Tensor) -> float:
    """``||A T - B||_F^2 / ||B||_F^2`` computed from Gram matrices alone (no raw activations needed).

    ``||A T - B||_F^2 = tr(T^T A^T A T) - 2 tr(T^T A^T B) + tr(B^T B)``, i.e. ``a_gram = A^T A``,
    ``b_gram = B^T B``, ``cross_cov = A^T B``. Centre both sides first if you want the centred variant.
    """
    t = t.double()
    a_gram = a_gram.double()
    b_gram = b_gram.double()
    cross_cov = cross_cov.double()
    numerator = torch.trace(t.T @ a_gram @ t) - 2.0 * torch.trace(t.T @ cross_cov) + torch.trace(b_gram)
    denominator = torch.trace(b_gram)
    if denominator <= 0:
        raise ValueError("recon_error: ||B||_F^2 must be positive")
    return float((numerator / denominator).clamp_min(0.0))


def rank_at_variance(cross_cov: torch.Tensor, *, threshold: float = 0.9) -> int:
    """Number of singular values of ``cross_cov`` needed for their cumulative energy to reach ``threshold``.

    A small number means most of the aligned cross-covariance energy lives in a low-dimensional
    shared subspace; a number close to ``min(cross_cov.shape)`` means alignment energy is spread
    thin across nearly every direction (the "no shared structure" failure mode).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    singular_values = torch.linalg.svdvals(cross_cov.double())
    total = singular_values.sum()
    if total <= 0:
        return int(singular_values.numel())
    cumulative = torch.cumsum(singular_values, dim=0) / total
    hit = torch.nonzero(cumulative >= threshold, as_tuple=False)
    return int(hit[0].item()) + 1 if hit.numel() else int(singular_values.numel())


def norm_ratio(tau_a: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor) -> float:
    """``||tau_B||_F / ||tau_A||_F`` for ``tau_B = t_out^T @ tau_A @ t_in``. Must be ~1.0 (paper Eq. 11)."""
    tau_a = tau_a.double()
    tau_b = t_out.double().T @ tau_a @ t_in.double()
    norm_a = torch.linalg.matrix_norm(tau_a, ord="fro")
    if norm_a <= 0:
        raise ValueError("norm_ratio: ||tau_A||_F must be positive")
    return float(torch.linalg.matrix_norm(tau_b, ord="fro") / norm_a)


def functional_gram(delta_h_out: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 3's ``G = delta_H_out @ H_out^T``, an (M, M) Gram matrix over calibration samples.

    ``delta_h_out`` is the task vector's induced change to this layer's output activations
    (``H_in @ tau^T`` for a linear layer); ``h_out`` is the pre-update output activations on the
    same calibration rows. Both arguments must share the same sample axis (dim 0 = M) and the
    same row order, so ``G_A`` (source) and ``G_B`` (target, transported vector) are directly
    comparable even though the two models' hidden widths differ.
    """
    if delta_h_out.shape[0] != h_out.shape[0]:
        raise ValueError("functional_gram: delta_h_out and h_out must share the sample axis (dim 0)")
    return delta_h_out.double() @ h_out.double().T


def fidelity_score(g_a: torch.Tensor, g_b: torch.Tensor) -> float:
    """``1 - ||G_A - G_B||_F^2 / ||G_A||_F^2``: did the task vector's functional effect survive transport.

    Distinct from ``recon_error``: this compares the *task vector's induced effect*, not the raw
    activation spaces. A layer can have low recon_error (representations align well) and still
    score low here if tau_A's energy sits in directions the alignment map doesn't preserve.
    """
    if g_a.shape != g_b.shape:
        raise ValueError("fidelity_score: G_A and G_B must have the same (M, M) shape")
    g_a = g_a.double()
    g_b = g_b.double()
    norm_a_sq = torch.linalg.matrix_norm(g_a, ord="fro") ** 2
    if norm_a_sq <= 0:
        raise ValueError("fidelity_score: ||G_A||_F^2 must be positive")
    diff_sq = torch.linalg.matrix_norm(g_a - g_b, ord="fro") ** 2
    return float(1.0 - diff_sq / norm_a_sq)
