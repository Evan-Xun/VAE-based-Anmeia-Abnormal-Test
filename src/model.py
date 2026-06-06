"""Small fully-connected VAE for tabular CBC indicators."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _load_legacy_vae_module() -> ModuleType:
    legacy_path = Path(__file__).resolve().parent.parent / "model" / "VAE.py"
    spec = importlib.util.spec_from_file_location("_legacy_vae_model", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy VAE module from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy_vae_module = _load_legacy_vae_module()
VAEAnomalyDetection = _legacy_vae_module.VAEAnomalyDetection
VAEAnomalyTabular = _legacy_vae_module.VAEAnomalyTabular


class VAE(nn.Module):
    """Variational autoencoder for tabular CBC data."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_log_var = nn.Linear(8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into latent mean and log variance."""
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_log_var(hidden)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to CBC feature space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructed input, latent mean, and latent log variance."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var


class SemiSupervisedVAE(VAE):
    """VAE with an end-to-end classifier head on the latent representation."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 2):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, max(8, latent_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(8, latent_dim * 2), 1),
        )

    def classify(self, mu: torch.Tensor) -> torch.Tensor:
        """Return logits for the positive class from latent means."""
        return self.classifier(mu).squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, latent stats, and classifier logits."""
        x_hat, mu, log_var = super().forward(x)
        logits = self.classify(mu)
        return x_hat, mu, log_var, logits


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss, reconstruction loss, and KL loss."""
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl_loss = torch.mean(latent_kl_divergence(mu, log_var, reduction="sum"))
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def semi_supervised_vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    kl_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine reconstruction, KL, and classification losses."""
    vae_total, recon_loss, kl_loss = vae_loss(x, x_hat, mu, log_var, kl_weight=kl_weight)
    classification_loss = F.binary_cross_entropy_with_logits(logits, y.float())
    total_loss = vae_total + classification_weight * classification_loss
    return total_loss, recon_loss, kl_loss, classification_loss


def latent_kl_divergence(mu: torch.Tensor, log_var: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute per-sample KL divergence from q(z|x) to a standard normal prior."""
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    if reduction == "mean":
        return torch.mean(kl_per_dim, dim=1)
    if reduction == "sum":
        return torch.sum(kl_per_dim, dim=1)
    raise ValueError("reduction must be either 'mean' or 'sum'")


@torch.no_grad()
def anomaly_scores(model: VAE, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute score = reconstruction error + latent KL divergence for each sample."""
    model.eval()
    mu, log_var = model.encode(x)
    x_hat = model.decode(mu)
    recon_error = torch.mean((x - x_hat).pow(2), dim=1)
    kl_div = latent_kl_divergence(mu, log_var, reduction="mean")
    score = recon_error + kl_div
    return score, recon_error, kl_div
