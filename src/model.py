"""Small fully-connected VAE for tabular CBC indicators."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


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


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss, reconstruction loss, and KL loss."""
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


@torch.no_grad()
def anomaly_scores(model: VAE, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute score = reconstruction error + latent distance for each sample."""
    model.eval()
    mu, log_var = model.encode(x)
    x_hat = model.decode(mu)
    recon_error = torch.mean((x - x_hat).pow(2), dim=1)
    latent_dist = 0.5 * torch.mean(mu.pow(2), dim=1)
    score = recon_error + latent_dist
    return score, recon_error, latent_dist
