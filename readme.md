# Beta-VAE + KL Warm-up + Full-KL Score 的贫血异常检测

本项目在原有 VAE 贫血异常检测模型基础上，针对训练中出现的 KL collapse 问题进行了优化。优化重点包括 KL 权重调整、KL warm-up，以及异常分数中 latent 部分的完整 KL divergence 计算。

## 优化内容

### 1. Beta-VAE 损失

原始 VAE 使用标准损失：

```text
loss = reconstruction_loss + kl_loss
```

优化后引入 KL 权重 `beta`：

```text
loss = reconstruction_loss + beta_t * kl_loss
```

当前实验使用：

```text
beta = 0.1
```

这样可以降低 KL 项在早期训练中的压制作用，缓解 posterior collapse，使模型更充分利用潜变量。

### 2. KL Warm-up

训练初期不直接使用最终 KL 权重，而是线性增加：

```text
beta_t = beta * min(1, epoch / kl_warmup_epochs)
```

当前实验使用：

```text
kl_warmup_epochs = 50
```

该策略让模型先学习重构，再逐步加强潜空间正则化。

### 3. Full-KL Anomaly Score

原始 anomaly score 使用简化 latent distance：

```text
reconstruction_error = mean((x - x_hat)^2)
latent_dist = 0.5 * mean(mu^2)
anomaly_score = reconstruction_error + latent_dist
```

优化后改为完整逐样本 KL divergence：

```text
reconstruction_error = mean((x - x_hat)^2)
kl_div = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
anomaly_score = reconstruction_error + kl_div
```

相比只考虑 `mu` 偏离 0，完整 KL 同时考虑 `mu` 和 `log_var`，与 VAE 的概率建模目标更一致。

## 当前实验设置

```text
dataset = diagnosed_cbc_data_v4.csv
target = anemia
input_dim = 14
latent_dim = 2
epochs = 100
batch_size = 32
learning_rate = 0.001
beta = 0.1
kl_warmup_epochs = 50
fixed_threshold = 0.97
seed = 42
```

数据清洗结果：

```text
original_rows = 1150
rows_after_cleaning = 1081
dropped_rows = 69
```

## 优化结果

训练结束后，优化模型的损失为：

```text
final_total_loss = 0.6987
final_reconstruction_loss = 0.5984
final_kl_loss = 1.0031
final_kl_weight = 0.1
```

KL loss 不再接近 0，说明原先的 KL collapse 已经被明显缓解，潜空间被模型实际使用。

### 固定阈值 0.97

```text
AUROC = 0.9112
Precision = 0.8726
Recall = 0.8954
F1-score = 0.8839
```

固定阈值下，模型召回率较高，说明漏检减少；precision 相对下降，说明误报略有增加。

### 重新校准阈值

根据 `threshold_sweep.csv`，最佳 F1 阈值为：

```text
threshold = 0.7651
Precision = 0.8613
Recall = 0.9739
F1-score = 0.9141
```

重新校准后，F1 从固定阈值下的 `0.8839` 提升到 `0.9141`，Recall 提升到 `0.9739`。

## 与旧版 Score 的对比

在相同的 `beta = 0.1` 和 `kl_warmup_epochs = 50` 设置下：

| Anomaly score | AUROC | 固定阈值 F1 | 最佳阈值 F1 |
| --- | ---: | ---: | ---: |
| 简化 latent distance | 0.8933 | 0.8639 | 0.8982 |
| 完整 KL divergence | 0.9112 | 0.8839 | 0.9141 |

完整 KL anomaly score 同时提升了 AUROC、固定阈值 F1 和最佳阈值 F1，说明该修改不仅更符合 VAE 理论，也改善了实际异常检测效果。

## 结论

本分支的主要优化是：

- 使用 `beta = 0.1` 降低 KL 项权重。
- 使用 50 epoch 的 KL warm-up 缓解 KL collapse。
- 使用完整 KL divergence 替代简化 latent distance 作为 anomaly score 的 latent 部分。
- 对优化后的异常分数重新校准阈值。

最终结果表明，优化后的 VAE 成功缓解 KL collapse，并在使用完整 KL anomaly score 后提升了异常检测性能。重新校准阈值后，模型达到：

```text
AUROC = 0.9112
Best F1-score = 0.9141
Recall = 0.9739
```

## 运行方式

运行完整训练、评估、可视化和阈值扫描流程：

```bash
python run_all.py --data diagnosed_cbc_data_v4.csv --cleaning range --epochs 100 --device cpu
```

显式指定当前优化参数：

```bash
python run_all.py \
  --data diagnosed_cbc_data_v4.csv \
  --cleaning range \
  --epochs 100 \
  --device cpu \
  --beta 0.1 \
  --kl-warmup-epochs 50 \
  --threshold 0.97
```

主要输出文件：

```text
results/vae_anemia.pt
results/training_history.npy
results/evaluation_outputs.npz
results/threshold_sweep.csv
results/plots/
```
