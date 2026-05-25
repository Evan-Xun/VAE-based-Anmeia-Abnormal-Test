"""Train the anemia VAE baseline on normal CBC samples only."""

from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_anemia_data
from model import VAE, anomaly_scores, vae_loss


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args) -> dict[str, list[float]]:
    """Train the VAE and save model artifacts."""
    set_seed(args.seed)
    data = load_anemia_data(args.data, seed=args.seed)
    loader = DataLoader(data.train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=4, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"total": [], "recon": [], "kl": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch_x)
            loss, recon_loss, kl_loss = vae_loss(batch_x, x_hat, mu, log_var)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_x)
            epoch_total += loss.item() * batch_size
            epoch_recon += recon_loss.item() * batch_size
            epoch_kl += kl_loss.item() * batch_size

        n = len(data.train_dataset)
        history["total"].append(epoch_total / n)
        history["recon"].append(epoch_recon / n)
        history["kl"].append(epoch_kl / n)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={history['total'][-1]:.4f} "
                f"recon={history['recon'][-1]:.4f} "
                f"kl={history['kl'][-1]:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_scores, _, _ = anomaly_scores(model, data.train_x.to(device))
    threshold = float(torch.quantile(train_scores.cpu(), 0.95))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": data.feature_columns,
            "scaler_mean": data.scaler.mean_,
            "scaler_scale": data.scaler.scale_,
            "threshold": threshold,
            "history": history,
            "latent_dim": args.latent_dim,
        },
        args.output_dir / "vae_anemia.pt",
    )
    np.save(args.output_dir / "training_history.npy", history)
    print(f"Saved model to {args.output_dir / 'vae_anemia.pt'}")
    print(f"95th percentile training threshold: {threshold:.4f}")
    return history


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("anemia.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
