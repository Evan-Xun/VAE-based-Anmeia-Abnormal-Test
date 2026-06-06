import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import SemiSupervisedVAE, semi_supervised_vae_loss  # noqa: E402


def test_semi_supervised_vae_forward_shapes():
    model = SemiSupervisedVAE(input_dim=8, latent_dim=4)
    x = torch.randn(5, 8)

    x_hat, mu, log_var, logits = model(x)

    assert x_hat.shape == (5, 8)
    assert mu.shape == (5, 4)
    assert log_var.shape == (5, 4)
    assert logits.shape == (5,)


def test_semi_supervised_vae_loss_adds_classification_term():
    x = torch.zeros(2, 2)
    x_hat = torch.zeros(2, 2)
    mu = torch.ones(2, 2)
    log_var = torch.zeros(2, 2)
    logits = torch.tensor([0.0, 0.0])
    y = torch.tensor([0.0, 1.0])

    total_loss, recon_loss, kl_loss, classification_loss = semi_supervised_vae_loss(
        x,
        x_hat,
        mu,
        log_var,
        logits,
        y,
        kl_weight=0.25,
        classification_weight=2.0,
    )

    assert recon_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(1.0)
    assert classification_loss.item() == pytest.approx(0.693147, rel=1e-4)
    assert total_loss.item() == pytest.approx(0.25 + 2.0 * classification_loss.item(), rel=1e-4)
