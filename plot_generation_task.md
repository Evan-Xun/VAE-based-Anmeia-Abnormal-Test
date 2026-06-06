# Plot Generation Task — VAE Anemia Detection

## Context

This is a VAE-based anomaly detection project for anemia using CBC (Complete Blood Count) data.
The experiment has already been run. All model training is complete.
Your job is to **load the saved model and data, then generate the required plots**.

---

## Assumed Project Structure

```
project/
├── cbc_8_features_reports_only.csv
├── results/
│   └── final_reports_only_keep_duplicates/
│       ├── evaluation_outputs.npz
│       ├── hybrid_predictions.npz
│       └── training_history.npy
├── plots/          ← create this if it doesn't exist
└── src/visualise.py
```

Adjust paths to match the actual project structure.

---

## What You Need to Generate

### Plot 1 — Training Loss Curve

**Filename:** `plots/training_loss_curve.png`

**What to plot:**
- X-axis: Epoch (0 to 100)
- Y-axis: Loss value
- Three lines on the same plot:
  - `Total Loss` (blue)
  - `Reconstruction Loss` (orange)
  - `KL Loss` (green)
- Add a vertical dashed line at epoch 50 with label `"KL Warmup End"` (this is when KL warmup finishes)

**How to get the data:**
- If loss values were logged during training (e.g. stored in a list or dict), load them directly
- If not logged, re-run training with logging added — do NOT skip this plot

**Style requirements:**
- `figsize=(10, 5)`
- Title: `"VAE Training Loss Curve"`
- Legend visible
- Grid on
- Save as PNG at 150 dpi minimum

---

### Plot 2 — Anomaly Score Distribution

**Filename:** `plots/anomaly_score_distribution.png`

**What to plot:**
- Histogram of anomaly scores on the **test set**, split by true label
- Two overlapping histograms:
  - `Healthy` samples (blue, alpha=0.6)
  - `Anemia` samples (red, alpha=0.6)
- Add a vertical dashed line at the selected threshold (value: `0.4767`) with label `"Threshold = 0.4767"`

**How to get the data:**
- Load the test set from `cbc_8_features_reports_only.csv`
- Use the same 60/20/20 stratified split with `seed=42` to reconstruct the test set
- Run the trained VAE to compute anomaly scores for each test sample
- Anomaly score = reconstruction error + lambda_kl × per-sample KL divergence
  - Per-sample KL = `-0.5 * mean(1 + log_var - mu^2 - exp(log_var))` (full correct formula)
  - If lambda_kl was used during training, use the same value here

**Style requirements:**
- `figsize=(10, 5)`
- `bins=40`
- Title: `"Anomaly Score Distribution — Test Set"`
- X-axis label: `"Anomaly Score"`
- Y-axis label: `"Count"`
- Legend visible
- Grid on
- Save as PNG at 150 dpi minimum

---

### Plot 3 — ROC Curve

**Filename:** `plots/roc_curve_comparison.png`

**What to plot:**
- ROC curves for the following models on the **test set**, all in one figure:
  1. `Baseline VAE` — use anomaly score as the decision variable (higher score = predicted Anemia)
  2. `Supervised Only + Logistic Regression` — use predicted probability from the LR model
  3. `Hybrid + Logistic Regression` — use predicted probability from the hybrid LR model

- For each curve, show its AUROC in the legend label, e.g.:
  - `"Baseline VAE (AUROC = 0.5146)"`
  - `"Supervised LR (AUROC = 0.6202)"`
  - `"Hybrid LR (AUROC = 0.6332)"`

- Add a diagonal dashed grey line for random baseline (`"Random (AUROC = 0.50)"`)

**How to get the data:**
- Use `sklearn.metrics.roc_curve` and `roc_auc_score`
- Positive class = Anemia (label = 1)
- Reconstruct the same test set using `seed=42` and the same stratified split

**Style requirements:**
- `figsize=(8, 7)`
- Title: `"ROC Curve Comparison"`
- X-axis: `"False Positive Rate"`
- Y-axis: `"True Positive Rate"`
- Legend in lower-right (`loc='lower right'`)
- Grid on
- Save as PNG at 150 dpi minimum

---

## General Instructions

1. All three plots must be saved to the `plots/` directory
2. Use `matplotlib` for all plots; `seaborn` is optional for styling
3. Do not display plots interactively (`plt.show()`); only save to file
4. Use the formal `reports-only` run outputs under `results/final_reports_only_keep_duplicates/` as the source of truth. If any model needs to be re-run to get scores, use the **same hyperparameters** as the original experiment:
   - `latent_dim = 2`
   - `epochs = 100`
   - `batch_size = 32`
   - `learning_rate = 0.001`
   - `beta = 0.1`
   - `kl_warmup_epochs = 50`
   - `seed = 42`
5. Label encoding: `Healthy = 0`, `Anemia = 1`

---

## Deliverables

When done, confirm:
- [x] `plots/training_loss_curve.png` saved
- [x] `plots/anomaly_score_distribution.png` saved
- [x] `plots/roc_curve_comparison.png` saved
- [x] No hardcoded absolute paths (use relative paths from project root)
