# 基于 VAE 的贫血异常检测

本项目实现了一个基于变分自编码器（Variational Autoencoder, VAE）的无监督多变量异常检测流程，用于分析与贫血相关的血常规（Complete Blood Count, CBC）指标。

当前实现对应 Session 6 任务要求，主要流程包括：

- 仅使用正常样本训练 VAE。
- 使用数值型 CBC 指标作为模型输入。
- 诊断标签只用于评估，不参与训练。
- 默认执行贫血检测任务：`Healthy` 视为正常样本，贫血诊断视为阳性样本，非贫血疾病样本会从默认贫血实验中排除。
- 使用重构误差和潜变量距离计算异常分数。
- 使用 AUROC、精确率、召回率和 F1 分数评估模型。
- 生成展示所需的训练、分布、ROC、潜空间和重构可视化图像。

## 数据集

默认数据集文件为：

```text
diagnosed_cbc_data_v4.csv
```

该数据集包含 1,281 条样本、14 个 CBC 输入指标和 1 个诊断标签：

| 字段 | 用途 |
| --- | --- |
| `WBC`, `LYMp`, `NEUTp`, `LYMn`, `NEUTn` | 输入特征 |
| `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC` | 输入特征 |
| `PLT`, `PDW`, `PCT` | 输入特征 |
| `Diagnosis` | 仅用于评估；`Healthy = 正常`，贫血诊断 = 阳性 |

当前标签分布如下：

```text
健康样本: 336
贫血诊断样本: 814
默认贫血任务中排除的非贫血疾病样本: 131
总样本数: 1281
```

训练时会先进行 80/20 的分层划分，随后只使用正常样本训练 VAE。数值输入特征会使用 `StandardScaler` 标准化。

旧版 `anemia.csv` 数据格式仍然保留兼容支持，但不再作为默认数据源。

## 方法说明

模型是一个用于表格型 CBC 特征的小型全连接 VAE。输入维度会根据加载的数据集自动推断。

编码器结构：

```text
Input(n_features) -> Linear(n_features, 16) -> ReLU -> Linear(16, 8) -> ReLU
```

编码器输出：

```text
mu, log_var
```

解码器结构：

```text
Latent(2) -> Linear(2, 8) -> ReLU -> Linear(8, 16) -> ReLU -> Linear(16, n_features)
```

训练损失：

```text
total_loss = reconstruction_loss + beta_t * kl_loss
beta_t = beta * min(1, epoch / kl_warmup_epochs)
```

默认训练使用 `beta = 0.1` 和 50 个 epoch 的 KL warm-up，以缓解 KL collapse。设置 `--kl-warmup-epochs 0` 可关闭退火，设置 `--beta 1.0` 可恢复标准 VAE 权重。

异常分数：

```text
reconstruction_error = MSE(x, x_hat)
latent_dist = 0.5 * mean(mu^2)
anomaly_score = reconstruction_error + latent_dist
```

异常分数越高，表示该样本与 VAE 学到的正常 CBC 模式偏离越明显。

## 当前结果

当前基线模型训练 100 个 epoch，主要参数如下：

```text
latent_dim = 2
batch_size = 32
learning_rate = 0.001
beta = 0.1
kl_warmup_epochs = 50
seed = 42
threshold = 0.97
```

在测试集上的评估结果：

```text
AUROC: 0.9158
Precision: 0.9200
Recall: 0.8679
F1-score: 0.8932
```

当前固定异常判定阈值为：

```text
threshold = 0.97
```

该阈值接近分析过程中找到的 Youden 风格阈值：

```text
Youden threshold: 0.9797
Precision: 0.9200
Recall: 0.8679
F1-score: 0.8932
```

## 生成文件

模型与评估文件：

```text
results/vae_anemia.pt
results/evaluation_outputs.npz
results/training_history.npy
```

所需图像：

```text
results/plots/training_loss_curve.png
results/plots/anomaly_score_distribution.png
results/plots/roc_curve.png
results/plots/latent_space_scatter.png
results/plots/reconstruction_examples.png
```

图像含义：

- `training_loss_curve.png`：展示总损失、重构损失和 KL 损失随训练轮数的变化，用于确认训练是否收敛。
- `anomaly_score_distribution.png`：比较正常样本和贫血样本的异常分数分布；贫血样本通常应整体偏向更高分数。
- `roc_curve.png`：展示模型在不同阈值下区分正常样本和贫血样本的能力。当前 AUROC 约为 0.9188。
- `latent_space_scatter.png`：展示二维潜变量均值 `(mu_1, mu_2)`，并按标签着色。
- `reconstruction_examples.png`：对比部分正常样本和贫血样本的原始 CBC 数值与重构结果。

## 运行方式

先激活项目虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

使用一条命令运行完整流程：

```powershell
.\.venv\Scripts\python.exe run_all.py --epochs 100 --device cpu
```

流程开始前，程序会提示选择项目目录下的 CSV 数据集，并选择数据清洗方式。随后会完成训练、评估，并重新生成 `results/plots` 下的全部图像，同时写出阈值比较表：

```text
results/threshold_sweep.csv
```

默认异常判定阈值固定为 `0.97`。也可以显式传入：

```powershell
.\.venv\Scripts\python.exe run_all.py --epochs 100 --device cpu --threshold 0.97
```

如果希望跳过交互式提示，可以直接指定数据集和清洗方法：

```powershell
.\.venv\Scripts\python.exe run_all.py --data diagnosed_cbc_data_v4.csv --cleaning range --epochs 100 --device cpu
```

可用清洗选项：

```text
range = 基于 CBC 合理取值范围的数据清洗
none  = 不进行取值范围清洗
```

默认目标是贫血检测。如果要运行更宽泛的血液异常检测任务：

```powershell
.\.venv\Scripts\python.exe run_all.py --epochs 100 --device cpu --target abnormal
```

默认情况下，流程会在训练前执行 CBC 取值范围清洗。该步骤会删除完全重复的行，以及包含缺失值、非数值或明显不合理特征值的行，例如 `MCH = 3117`、`HCT = 316`、`MCV = 990`。清洗报告会保存到：

```text
results/data_cleaning_report.csv
```

如果不想执行该清洗步骤：

```powershell
.\.venv\Scripts\python.exe run_all.py --epochs 100 --device cpu --no-cleaning
```

也可以分别运行各个步骤。训练 VAE：

```powershell
python src\train.py --epochs 100 --device cuda
```

评估训练后的模型：

```powershell
python src\evaluate.py --device cuda
```

生成五张所需图像：

```powershell
python src\visualise.py --device cuda
```

如果没有可用 CUDA，请将 `cuda` 替换为 `cpu`。

## 项目结构

```text
src/dataset.py      数据加载、训练/测试划分和特征标准化
src/model.py        VAE 模型、损失函数和异常分数计算
src/train.py        训练循环和模型保存
src/evaluate.py     AUROC、精确率、召回率、F1 分数和评估输出保存
src/visualise.py    Session 6 所需图像生成
```

## 当前局限

此前观察到的主要问题是 KL collapse：

```text
KL loss becomes almost zero during training.
```

这说明潜空间没有被充分利用，模型可能更接近普通自编码器。当前训练脚本已经支持 beta-VAE 和 KL annealing：

```text
python run_all.py --beta 0.1 --kl-warmup-epochs 50
```

候选实验参数：

```text
python run_all.py --beta 1.0 --kl-warmup-epochs 0
python run_all.py --beta 0.1 --kl-warmup-epochs 50
python run_all.py --beta 0.05 --kl-warmup-epochs 50
python run_all.py --beta 0.01 --kl-warmup-epochs 50
```

目标是观察更合适的 KL 平衡是否能改善潜空间分离效果、召回率和整体异常检测性能。
