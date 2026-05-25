# VAE-based Anemia Abnormal Detection

This project implements a Variational Autoencoder (VAE) baseline for
unsupervised multivariate anomaly detection on anemia-related Complete Blood
Count (CBC) indicators.

The current implementation follows the Session 6 task brief:

- Train the VAE only on normal samples (`Result = 0`).
- Use four CBC indicators as model inputs.
- Use the `Result` label only for evaluation, not for training.
- Compute anomaly scores from reconstruction error and latent distance.
- Evaluate with AUROC, precision, recall, and F1-score.
- Generate the required visualisations for presentation.

## Dataset

The dataset file is:

```text
anemia.csv
```

It contains 1,421 samples with the following columns:

| Column | Usage |
| --- | --- |
| `Gender` | Available in the dataset, not used in the current VAE baseline |
| `Hemoglobin` | Input feature |
| `MCH` | Input feature |
| `MCHC` | Input feature |
| `MCV` | Input feature |
| `Result` | Evaluation label only; `0 = normal`, `1 = anemia` |

Current label distribution:

```text
Normal samples (Result = 0): 801
Anemia samples (Result = 1): 620
Total samples: 1421
```

The VAE is trained only on normal samples after an 80/20 stratified split.
The four input features are standardised with `StandardScaler`.

## Method

The model is a small fully connected VAE designed for four tabular features.

Encoder:

```text
Input(4) -> Linear(4, 16) -> ReLU -> Linear(16, 8) -> ReLU
```

The encoder outputs:

```text
mu, log_var
```

Decoder:

```text
Latent(2) -> Linear(2, 8) -> ReLU -> Linear(8, 16) -> ReLU -> Linear(16, 4)
```

Training loss:

```text
total_loss = reconstruction_loss + kl_loss
```

Anomaly score:

```text
reconstruction_error = MSE(x, x_hat)
latent_dist = 0.5 * mean(mu^2)
anomaly_score = reconstruction_error + latent_dist
```

Higher anomaly scores indicate samples that deviate more from the normal CBC
patterns learned by the VAE.

## Current Results

The current baseline was trained for 100 epochs with:

```text
latent_dim = 2
batch_size = 32
learning_rate = 0.001
seed = 42
```

Evaluation on the held-out test set:

```text
AUROC: 0.9188
AUPRC: 0.9249
Accuracy: 0.8526
Precision: 0.9184
Recall: 0.7258
F1-score: 0.8108
```

Confusion matrix:

```text
              Predicted Normal   Predicted Anemia
Actual Normal              153                  8
Actual Anemia               34                 90
```

The current threshold is based on the 95th percentile of training normal
anomaly scores. This gives high precision and low false positives, but recall
is moderate. For screening-oriented use, a lower threshold may be preferred.

Using a Youden-style threshold during analysis gave:

```text
Precision: 0.8607
Recall: 0.8468
F1-score: 0.8537
```

## Generated Outputs

Model and evaluation files:

```text
results/vae_anemia.pt
results/evaluation_outputs.npz
results/training_history.npy
```

Required plots:

```text
results/plots/training_loss_curve.png
results/plots/anomaly_score_distribution.png
results/plots/roc_curve.png
results/plots/latent_space_scatter.png
results/plots/reconstruction_examples.png
```

Plot meanings:

- `training_loss_curve.png`: shows total loss, reconstruction loss, and KL loss
  across epochs to confirm training convergence.
- `anomaly_score_distribution.png`: compares anomaly score distributions for
  normal and anemia samples. Anemia samples should generally shift to higher
  scores.
- `roc_curve.png`: shows the model's ability to separate normal and anemia
  samples across thresholds. The current AUROC is 0.9188.
- `latent_space_scatter.png`: visualises the 2D latent mean values
  `(mu_1, mu_2)` coloured by label.
- `reconstruction_examples.png`: compares original and reconstructed CBC values
  for selected normal and anemia records.

## How to Run

Activate the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Train the VAE:

```powershell
python src\train.py --epochs 100 --device cuda
```

Evaluate the trained model:

```powershell
python src\evaluate.py --device cuda
```

Generate the five required plots:

```powershell
python src\visualise.py --device cuda
```

If CUDA is unavailable, replace `cuda` with `cpu`.

## Project Structure

```text
src/dataset.py      Data loading, train/test split, and feature scaling
src/model.py        VAE model, loss function, and anomaly score calculation
src/train.py        Training loop and checkpoint saving
src/evaluate.py     AUROC, precision, recall, F1, and saved evaluation outputs
src/visualise.py    Required Session 6 plots
```

## Current Limitation

The main issue observed so far is KL collapse:

```text
KL loss becomes almost zero during training.
```

This suggests that the latent space is not being used strongly, and the model
may behave more like a standard autoencoder. The next improvement should focus
on KL weighting, such as beta-VAE or KL annealing:

```text
loss = reconstruction_loss + beta * kl_loss
```

Candidate experiments:

```text
beta = 1.0
beta = 0.1
beta = 0.05
beta = 0.01
```

The goal is to check whether a better KL balance improves latent space
separation, recall, and overall anomaly detection performance.
