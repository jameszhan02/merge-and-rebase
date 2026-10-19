import pytest
import torch

from merge_and_rebase.eval.diagnostics.theseus_llm_kl import _distribution_metrics


def test_distribution_metrics_identical_distributions_have_zero_kl() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    metrics = _distribution_metrics(log_probs, log_probs, torch.tensor([2, 0]))

    assert metrics["token_count"] == 2
    assert metrics["kl_sum"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["reference_nll_sum"] == pytest.approx(metrics["candidate_nll_sum"])
    assert metrics["top1_flip_count"] == 0


def test_distribution_metrics_detects_gold_nll_and_top1_changes() -> None:
    reference = torch.log_softmax(torch.tensor([[4.0, 1.0], [1.0, 4.0]]), dim=-1)
    candidate = torch.log_softmax(torch.tensor([[1.0, 4.0], [1.0, 4.0]]), dim=-1)
    metrics = _distribution_metrics(reference, candidate, torch.tensor([0, 1]))

    assert metrics["kl_sum"] > 0
    assert metrics["candidate_nll_sum"] > metrics["reference_nll_sum"]
    assert metrics["top1_flip_count"] == 1
