import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import vae_loss  # noqa: E402
from train import kl_weight_for_epoch  # noqa: E402


def test_vae_loss_applies_kl_weight():
    x = torch.zeros(2, 2)
    x_hat = torch.zeros(2, 2)
    mu = torch.ones(2, 2)
    log_var = torch.zeros(2, 2)

    total_loss, recon_loss, kl_loss = vae_loss(x, x_hat, mu, log_var, kl_weight=0.25)

    assert recon_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(1.0)
    assert total_loss.item() == pytest.approx(0.25)


def test_kl_weight_for_epoch_linear_warmup():
    assert kl_weight_for_epoch(epoch=1, beta=0.1, warmup_epochs=50) == pytest.approx(0.002)
    assert kl_weight_for_epoch(epoch=50, beta=0.1, warmup_epochs=50) == pytest.approx(0.1)
    assert kl_weight_for_epoch(epoch=80, beta=0.1, warmup_epochs=50) == pytest.approx(0.1)
    assert kl_weight_for_epoch(epoch=1, beta=0.1, warmup_epochs=0) == pytest.approx(0.1)
