# 基于 VAE 的 CBC 贫血异常检测实验报告

## 1. 引言

### 1.1 项目背景

贫血是临床中常见的血液系统异常，早期识别对后续诊断和干预具有实际意义。全血细胞计数（Complete Blood Count, CBC）是最常见、成本较低且获取方便的检查方式，因此基于 CBC 指标开展贫血检测具有明确应用价值。

在传统机器学习设定下，若数据标签质量较高且特征完整，则监督学习模型通常可以取得较好结果。但在真实医疗数据场景中，常见问题包括：

- 标签不完全准确或粒度较粗；
- 数据来源不一致；
- 结构化特征有限；
- 正常与异常之间的边界并不稳定。

因此，本项目首先实现一个基于变分自编码器（Variational Autoencoder, VAE）的异常检测基线模型，希望通过学习正常样本分布来识别贫血异常；随后进一步探索基于 VAE 的改进路线，包括松耦合混合模型与端到端半监督模型，并比较它们在不同数据条件下的表现。

### 1.2 研究目标

本项目的核心目标包括四点：

1. 实现并验证 VAE 基线模型在 CBC 贫血检测任务中的可行性。
2. 比较纯 VAE、纯监督学习与混合模型的性能差异。
3. 观察模型在不同数据集条件下的稳定性与泛化表现。
4. 分析数据质量、特征完整性和模型结构对结果的影响。

### 1.3 研究问题

围绕上述目标，本项目主要回答以下问题：

1. 单纯依赖 VAE 异常检测能否有效完成贫血检测？
2. 将 VAE 衍生特征与监督分类器结合后，是否能稳定优于传统 VAE？
3. 当数据集从结构化、可分性较强的环境转向更复杂的报告数据时，模型性能会发生怎样的变化？
4. 端到端半监督学习是否比“VAE 提特征 + 外部分类器”的松耦合方式更有效？

## 2. 正文

### 2.1 整体实验流程

本项目的实验流程可以概括为三条主线：

1. 实现并评估 VAE 基线；
2. 构建混合模型，与纯监督模型进行对比；
3. 进一步实现端到端半监督 VAE，并与前两类方法比较。

整体流程如下：

```text
数据准备
  -> 数据清洗
  -> train / validation / test 划分
  -> 训练 baseline VAE
  -> 在验证集选择阈值
  -> 测试集评估 baseline
  -> 构建 supervised_only / vae_only / hybrid
  -> 构建 end-to-end semi-supervised VAE
  -> 对比不同数据集与不同模型结果
```

### 2.2 数据集构建

本项目在实验过程中使用了三种数据条件，它们分别代表了不同的数据复杂度。

#### 2.2.1 v4 结构化数据集

结构化数据集文件为 [diagnosed_cbc_data_v4.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/diagnosed_cbc_data_v4.csv)。该数据集具有以下特点：

- 标签来自结构化诊断字段；
- 特征较完整；
- 数据来源相对统一；
- 在健康与贫血之间具有较强可分性。

在该数据集上，项目主要采用 `Healthy vs Anemia` 的二分类设定，并使用现有清洗逻辑与标准切分方式完成实验。

#### 2.2.2 报告数据集

报告数据最终整理为 [cbc_8_features_reports_only.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/cbc_8_features_reports_only.csv)。该数据集来自 CBC 文本报告解析后的结构化结果，只保留 8 个核心指标：

- `WBC`
- `RBC`
- `HGB`
- `HCT`
- `MCV`
- `MCH`
- `MCHC`
- `PLT`

数据规模如下：

```text
总样本数 = 962
Anemia   = 490
Healthy  = 472
```

相比 v4 数据集，报告数据具有如下特点：

- 特征更少；
- 标签粒度更粗；
- 数据来源更接近真实报告场景；
- 类间边界更模糊。

#### 2.2.3 融合数据集

为了检验模型在跨来源数据上的适应能力，项目还构建了 8 指标融合数据集 `cbc_8_features_merged.csv`。该数据集由：

- 报告数据；
- v4 数据中映射到相同 8 指标的样本

组合而成。

融合数据集反映的是更复杂的应用场景：即模型需要面对来源不同、分布差异更大的混合型数据。

### 2.3 数据清洗与切分策略

数据清洗规则主要包括：

- 剔除缺失值样本；
- 剔除无法解析为数值的样本；
- 剔除超出生理范围的样本。

报告数据清洗结果为：

```text
original_rows       = 1000
rows_after_cleaning = 962
dropped_rows        = 38
```

实验中统一采用分层切分：

```text
train / validation / test = 60 / 20 / 20
seed = 42
```

其中：

- baseline VAE 只用训练集中的健康样本训练；
- 阈值只在验证集选择；
- 测试集只用于最终评估。

这样的设计避免了直接在测试集上选阈值造成的信息泄漏。

### 2.3 模型设计

本项目的方法部分按照“baseline VAE -> 松耦合 hybrid -> 端到端 semi-supervised VAE”的路线逐步展开。这样设计的原因是：首先需要验证基于正常样本分布建模的异常检测是否可行；在此基础上，再考察 VAE 学到的表示能否作为额外特征提升分类效果；最后进一步检验，把表示学习与分类学习放入同一优化框架后，是否能够减少目标错位并获得更强的任务相关表示。

#### 2.3.1 Baseline VAE 模型实现

项目中最早完成的基线模型是 VAE 异常检测器，对应实现主要位于：

- [src/model.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/model.py)
- [src/train.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/train.py)
- [src/evaluate.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/evaluate.py)

模型思路如下：

```text
输入 CBC 特征 x
  -> Encoder
  -> 潜变量分布参数 mu, log_var
  -> 采样得到 z
  -> Decoder 重构得到 x_hat
  -> 根据 reconstruction error + KL 计算 anomaly score
```

从具体实现看，本项目采用的是一个面向表格型 CBC 数据的小型全连接 VAE，而不是图像任务中常见的卷积结构。其原因在于：输入仅由少量数值型血常规指标构成，特征维度低、语义明确，使用过深网络容易带来不必要的参数冗余和过拟合风险。因此，编码器被设计为 `input_dim -> 16 -> 8` 的两层 MLP，激活函数统一使用 ReLU；随后分别通过两个线性层输出潜变量分布的均值 `mu` 与对数方差 `log_var`。解码器部分采用与编码器近似对称的结构 `latent_dim -> 8 -> 16 -> input_dim`，用于从潜变量重构原始 CBC 特征。默认设定下 `latent_dim = 2`，这样既便于控制模型复杂度，也有助于后续观察潜空间是否学到了可分的正常样本结构。

在前向传播过程中，模型先根据编码器输出得到 `q(z|x)` 的参数，然后使用 reparameterization trick 完成采样：

```text
z = mu + sigma * epsilon,  epsilon ~ N(0, I)
```

这一设计使采样过程能够参与反向传播，从而稳定地训练变分模型。需要注意的是，在最终异常评分阶段，项目并没有再次对 `z` 随机采样，而是直接使用 `mu` 进入解码器计算重构结果。这样做的目的，是减少随机采样带来的分数波动，使同一样本在推理阶段具有更稳定的 anomaly score，更适合后续基于阈值的贫血判别。

VAE 的训练目标由两部分组成：

1. 重构损失：保证模型能够学习正常样本结构；
2. KL 散度：约束潜变量分布。

总损失可以写为：

```text
L = L_recon + beta * L_KL
```

其中 `L_recon` 在代码中使用均方误差（MSE）实现，衡量重构输入 `x_hat` 与原始输入 `x` 的差异；`L_KL` 用于约束潜变量分布接近标准正态分布，从而避免潜空间无约束漂移。在本项目中，`beta` 默认设为 0.1，这意味着训练过程更强调“先学会稳定重构正常样本”，而不是过早地把潜变量压得过于接近先验分布。对于医疗小样本、低维表格数据来说，这种较轻的 KL 约束通常比 `beta = 1` 更稳妥。

此外，项目还引入了 `kl_warmup` 机制，即在前 50 个 epoch 内将 KL 权重从 0 线性增加到目标 `beta`。这样做的原因是：如果在训练初期就施加完整的 KL 正则，编码器容易过早塌缩到先验分布，导致解码器学不到有效的正常模式；而 warmup 可以让模型先建立基本重构能力，再逐步规范潜空间，缓解 posterior collapse 风险。

在本项目中，异常分数主要由：

- reconstruction error
- latent KL contribution

共同构成，即：

```text
anomaly_score = recon_error + kl_divergence
```

其直观含义是：如果某个样本既难以被正常样本分布准确重构，又在潜空间上偏离健康样本模式较远，那么它更可能是异常样本。这个设计比单独使用 reconstruction error 更稳健，因为有些样本虽然重构误差不大，但潜变量分布已经明显偏离训练时学到的正常区域。

在训练数据使用上，baseline VAE 只使用训练集中的健康样本进行拟合，而不直接利用贫血样本标签。这一设定符合异常检测的基本思想：先建模“什么是正常”，再判断哪些样本偏离正常分布。与此同时，特征标准化也是仅在健康训练样本上拟合得到的，这可以避免异常样本对正常分布估计造成干扰。阈值选择则放在验证集完成，测试集只用于最终报告结果，从而尽量避免信息泄漏。

默认训练参数为：

```text
latent_dim = 2
epochs = 100
batch_size = 32
learning_rate = 0.001
beta = 0.1
kl_warmup_epochs = 50
```

#### 2.3.2 监督学习对照模型实现

除了生成式路线之外，本项目还显式构建了监督学习对照模型，即 `supervised_only`。设置这一对照组的目的，是回答一个非常关键的问题：在当前 CBC 数据条件下，是否必须引入 VAE 才能取得较好结果，还是仅依赖原始结构化指标就已经足够完成贫血分类。如果不加入这条监督学习基线，就很难判断后续 hybrid 或端到端模型的性能提升究竟来自“更好的方法设计”，还是仅仅来自数据本身已经具有较强可分性。

`supervised_only` 的输入直接使用标准化后的原始 CBC 指标，不包含任何 VAE 衍生特征。在实现上，项目统一复用了与 hybrid 相同的数据划分方式，即 train / validation / test = 60 / 20 / 20，并保持相同的随机种子与评价指标，从而保证不同模型路线之间具有可比性。与 baseline VAE 不同的是，监督学习模型在训练时直接使用训练集中的健康与贫血标签，因此它学习的是从原始特征到类别标签的判别边界，而不是正常样本分布本身。

在分类器选择上，项目使用了两类具有代表性的监督模型：

- Logistic Regression
- Random Forest

其中，Logistic Regression 作为线性模型，适合衡量“原始 CBC 指标是否已经可以被简单线性边界区分”；Random Forest 则可以进一步捕捉非线性关系与特征交互，用来检验原始指标中是否存在更复杂的判别结构。将这两个模型作为监督学习基线，有助于从“线性可分性”和“非线性可分性”两个角度理解数据难度。如果 `supervised_only` 已经表现很强，那么说明数据本身标签与特征关系较明确；如果它表现有限，则更能凸显引入 VAE 表示学习或生成式建模的必要性。

此外，`supervised_only` 在本项目中不仅是一个简单的 baseline，更承担了解释实验结果的重要角色。例如，在结构化 v4 数据集上，监督学习模型接近饱和，说明该数据集本身更适合验证方法可行性；而在 reports-only 数据集上，监督模型虽优于 baseline VAE，但整体表现仍然有限，这进一步说明真实报告数据的任务难度更高。因此，监督学习对照组的加入，有助于让后续的 hybrid 与端到端结果具备更完整的参照系。

#### 2.3.3 松耦合混合模型实现

为改进传统 VAE 基线，本项目进一步构建了松耦合混合模型，对应实现主要位于：

- [src/hybrid_train.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/hybrid_train.py)

构建该模型的核心动机，是 baseline VAE 的优化目标本质上仍然是“重构正常样本”，而不是直接区分 Healthy 与 Anemia。即使 VAE 学到了一定的异常信息，这些信息也未必能通过单一 anomaly score 被充分利用。因此，本项目将 VAE 从“直接判别器”改造为“特征提取器”，再交由监督分类器完成最终决策，以检验生成式表示是否能够补充原始 CBC 指标。

该部分并非只训练一个模型，而是系统比较三种特征使用方式：

1. `supervised_only`：只使用原始 CBC 指标；
2. `vae_only`：只使用 VAE 衍生特征；
3. `hybrid`：同时使用原始 CBC 指标与 VAE 衍生特征。

其中，VAE 衍生特征包括：

- reconstruction error
- KL divergence
- anomaly score
- latent `mu`
- latent `log_var`
- 各特征的绝对重构误差
- 各特征的平方重构误差

松耦合混合模型流程如下：

```text
输入 CBC 特征
  -> 训练 VAE
  -> 提取 VAE 衍生特征
  -> 与原始特征组合或单独使用
  -> 送入外部分类器
  -> 输出 Healthy / Anemia 预测
```

之所以要显式保留 `mu`、`log_var` 和逐维重构误差，而不是只保留一个总分，是因为贫血相关信息可能分散在不同层面：有些样本整体异常分数不高，但某几个核心血常规指标的重构残差非常集中；另一些样本的输入值本身接近正常范围，但潜空间位置已经出现偏移。将这些信息展开为结构化特征后，监督分类器可以学习更复杂的判别边界，而不必被迫依赖单一阈值。

外部分类器统一采用：

- Logistic Regression
- Random Forest

选择这两个分类器也有明确考虑。Logistic Regression 代表线性、可解释性较强的基准模型，便于观察 VAE 特征是否带来稳定的线性增益；Random Forest 则能够刻画非线性关系与特征交互，适合检验 VAE 衍生特征是否包含更复杂的补充信息。这里需要强调：`vae_only + classifier` 不作为主基线，而更适合作为消融实验，用于检验“仅依赖 VAE 表示”时模型本身是否已经包含足够判别力；如果 `hybrid` 优于 `supervised_only`，则说明 VAE 特征对原始 CBC 指标确实提供了额外信息。

#### 2.3.4 端到端半监督 VAE 实现

考虑到松耦合方法存在目标错位问题，即 VAE 训练目标与最终分类目标并不完全一致，本项目进一步实现了端到端半监督 VAE，对应实现主要位于：

- [src/end_to_end_train.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/end_to_end_train.py)
- [src/model.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/model.py)

松耦合方法虽然增强了特征表达能力，但仍存在一个关键局限：VAE 与分类器分开训练，编码器优化时并不知道最终分类目标是什么。这会导致潜表示更偏向“重构友好”，却未必“分类友好”。为此，本项目进一步构建端到端半监督 VAE，使编码器在同一个训练过程中同时服务于生成任务与判别任务。

其结构如下：

```text
输入 x
  -> Encoder
  -> latent z / mu / log_var
     -> Decoder 重构 x_hat
     -> Classifier Head 输出 y_hat
```

在实现上，端到端模型继承 baseline VAE 的编码器和解码器结构，并额外在 latent 表示上接一个小型分类头。分类头使用 `latent_dim -> max(8, 2 * latent_dim) -> 1` 的两层全连接结构，输出二分类 logits。这里分类分支直接使用编码器输出的 `mu` 进行判别，而不是对随机采样得到的 `z` 分类，原因同样是为了减少采样噪声，让监督信号更稳定地作用于潜空间中心位置。

总损失由三部分组成：

```text
L = L_recon + beta * L_KL + lambda * L_cls
```

其中：

- `L_recon` 为重构损失；
- `L_KL` 为潜变量正则项；
- `L_cls` 为分类损失。

其中分类项在代码中采用 `binary_cross_entropy_with_logits` 实现，并通过 `classification_weight` 与 VAE 损失共同加权。这样一来，模型不会只关心“样本是否像正常人”，还会显式学习“哪些潜空间方向最有助于区分贫血”。与松耦合方法相比，这种结构的核心优势在于：编码器在学习数据表示时，会同时受到重构目标与分类目标约束，从而使潜在表示更直接服务于贫血检测任务。

从研究动机上看，这一步改进主要是为了解决两个问题。第一，baseline VAE 的 anomaly score 是固定形式的，表达能力有限；第二，hybrid 模型虽然能利用更多特征，但 VAE 与分类器之间仍存在优化目标不一致的问题。端到端半监督 VAE 则尝试把“表示学习”和“分类学习”放进同一优化框架中，理论上更有机会学到既保留正常样本结构、又对贫血检测敏感的潜在表示。当然，这种方法也更依赖超参数平衡，例如 `latent_dim`、`beta`、`classification_weight` 的设置都会影响重构与分类之间的取舍，因此它是一个更有研究价值、但也更需要调参与验证的改进方向。

### 2.4 评价指标

本项目主要使用以下指标评估模型表现：

- `F1-score`
- `AUROC`
- `Precision`
- `Recall`
- `Specificity`
- `AUPRC`

其中：

- `F1-score` 作为主指标，用于综合反映给定阈值下的分类效果；
- `AUROC` 用于评估模型整体排序能力；
- `Recall` 反映漏检情况；
- `Specificity` 反映误报情况。

在医疗背景下，仅有较高的 AUROC 并不足以说明模型足够实用，还需要结合 Recall 与 Specificity 进行判断。

## 3. 实验结果

### 3.1 v4 数据集上的结果

#### 3.1.1 Baseline VAE

在 v4 数据集上，baseline VAE 的正式测试结果位于：

- [results/v4_restored_run/baseline_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/v4_restored_run/baseline_metrics.csv)

其测试集表现如下：

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline VAE | 0.9223 | 0.9608 | 0.8533 | 0.9874 | 0.9155 | 0.5970 |

这一结果说明，VAE 基线模型在结构化、可分性较强的 CBC 数据上具有较强的异常排序能力，AUROC 已达到 `0.91` 以上。

#### 3.1.2 纯监督与混合模型

v4 数据集上的主要对比结果位于：

- [results/v4_restored_run/hybrid_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/v4_restored_run/hybrid_metrics.csv)
- [results/end_to_end_v4/end_to_end_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/end_to_end_v4/end_to_end_metrics.csv)

主要结果如下：

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline VAE | 0.9223 | 0.9608 | 0.8533 | 0.9874 | 0.9155 | 0.5970 |
| supervised_only + random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid + random_forest | 0.9999 | 1.0000 | 1.0000 | 0.9811 | 0.9905 | 1.0000 |
| hybrid + logistic_regression | 0.9986 | 0.9994 | 0.9753 | 0.9937 | 0.9844 | 0.9403 |
| end-to-end semi-supervised VAE | 0.9868 | 0.9931 | 0.9870 | 0.9560 | 0.9712 | 0.9701 |

从 v4 数据集的结果可以得到两点：

1. baseline VAE 是有效的，说明项目最初的基线实施是成功的；
2. 但 v4 数据集本身可分性过强，纯监督模型已经接近满分，因此该数据集不再适合区分改进方法之间的细微优劣。

### 3.2 报告数据集上的结果

#### 3.2.1 Baseline VAE

报告数据集的 baseline 结果位于：

- [results/final_reports_only_keep_duplicates/baseline_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/final_reports_only_keep_duplicates/baseline_metrics.csv)

其测试集结果如下：

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline VAE | 0.5146 | 0.5262 | 0.5111 | 0.9388 | 0.6619 | 0.0737 |

与 v4 相比，baseline VAE 在报告数据上的 AUROC 从 `0.92` 左右下降到 `0.51` 左右，几乎接近随机水平。

#### 3.2.2 纯监督、混合与端到端模型

报告数据上的主要结果位于：

- [results/final_reports_only_keep_duplicates/hybrid_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/final_reports_only_keep_duplicates/hybrid_metrics.csv)
- [results/end_to_end_reports/end_to_end_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/end_to_end_reports/end_to_end_metrics.csv)

主要结果如下：

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline VAE | 0.5146 | 0.5262 | 0.5111 | 0.9388 | 0.6619 | 0.0737 |
| supervised_only + logistic_regression | 0.6202 | 0.5826 | 0.6090 | 0.8265 | 0.7013 | 0.4526 |
| vae_only + logistic_regression | 0.6309 | 0.5957 | 0.5971 | 0.8469 | 0.7004 | 0.4105 |
| hybrid + logistic_regression | 0.6332 | 0.5939 | 0.5816 | 0.8367 | 0.6862 | 0.3789 |
| supervised_only + random_forest | 0.6159 | 0.5607 | 0.5821 | 0.7959 | 0.6724 | 0.4105 |
| hybrid + random_forest | 0.6072 | 0.5684 | 0.5714 | 0.7755 | 0.6580 | 0.4000 |
| end-to-end semi-supervised VAE | 0.6339 | 0.5912 | 0.6126 | 0.6939 | 0.6507 | 0.5474 |

这一部分结果说明：

1. 报告数据集显著更难；
2. 所有模型表现都低于 v4；
3. 纯监督模型优于 baseline VAE；
4. 混合模型和端到端模型并未在报告数据上体现稳定优势。

### 3.3 融合数据集上的结果（`latent_dim = 4`）

为了进一步分析跨来源数据条件下的模型表现，项目对融合数据集进行了实验，并统一设置 `latent_dim = 4`。相关结果位于：

- [results/merged_latent4_run/baseline_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/merged_latent4_run/baseline_metrics.csv)
- [results/merged_latent4_run/hybrid_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/merged_latent4_run/hybrid_metrics.csv)
- [results/end_to_end_merged_latent4/end_to_end_metrics.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/end_to_end_merged_latent4/end_to_end_metrics.csv)

各模型主要结果如下：

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline VAE | 0.6509 | 0.6948 | 0.6320 | 0.9222 | 0.7500 | 0.1481 |
| supervised_only + logistic_regression | 0.8301 | 0.8589 | 0.7429 | 0.9222 | 0.8229 | 0.4938 |
| supervised_only + random_forest | 0.9062 | 0.9370 | 0.7678 | 0.9650 | 0.8552 | 0.5370 |
| vae_only + logistic_regression | 0.8355 | 0.8501 | 0.7618 | 0.9455 | 0.8438 | 0.5309 |
| vae_only + random_forest | 0.8651 | 0.9108 | 0.7756 | 0.9416 | 0.8506 | 0.5679 |
| hybrid + logistic_regression | 0.8496 | 0.8607 | 0.7752 | 0.9261 | 0.8440 | 0.5741 |
| hybrid + random_forest | 0.8947 | 0.9311 | 0.7859 | 0.9572 | 0.8632 | 0.5864 |
| end-to-end semi-supervised VAE | 0.8992 | 0.9311 | 0.7654 | 0.9650 | 0.8537 | 0.5309 |

从融合数据集结果可以看出：

1. baseline VAE 明显弱于其他方法；
2. 纯监督、松耦合混合与端到端半监督三类模型都具有较高表现；
3. 在 `latent_dim = 4` 的设定下，最佳结果来自 `hybrid + random_forest`，其 `F1 = 0.8632`；
4. 端到端半监督 VAE 非常接近最优，但没有超过 `hybrid + random_forest`。

### 3.4 结果对比分析

综合三种数据条件，可以得到一个清晰结论：

#### 3.4.1 baseline VAE 对数据条件高度敏感

baseline VAE 在三种数据条件下的 AUROC 分别为：

| 数据集 | AUROC | F1 |
| --- | ---: | ---: |
| v4 | 0.9223 | 0.9155 |
| reports-only | 0.5146 | 0.6619 |
| merged (`latent_dim = 4`) | 0.6509 | 0.7500 |

这说明 VAE 基线模型对以下因素高度敏感：

- 特征完整性；
- 标签纯度；
- 数据来源一致性；
- 正常样本分布是否稳定。

#### 3.4.2 纯监督模型在结构化和可分性强的数据上更稳定

在 v4 数据集上，`supervised_only + random_forest` 已达到满分；在报告数据上，虽然分数明显下降，但仍稳定优于 baseline VAE。这说明：

> 当原始 CBC 特征已经提供足够判别信息时，纯监督模型往往更直接、更稳定。

#### 3.4.3 hybrid 的优势不稳定，但在融合数据上有一定价值

项目最初提出混合模型，是希望在 VAE 学习正常分布的基础上，再利用监督分类提升结果。从实验看：

- 在 v4 上，hybrid 有效，但没有超过最强纯监督模型；
- 在 reports-only 上，hybrid 没有体现稳定优势；
- 在 merged 数据集上，hybrid 表现较好，并取得当前 `latent_dim = 4` 设定下的最优结果。

因此，hybrid 的价值更适合作如下表述：

> VAE 衍生特征在某些复杂数据条件下可以提供补充信息，但这种增益并不稳定，也不足以证明 hybrid 在所有数据场景下都显著优于纯监督方法。

#### 3.4.4 端到端半监督方案更合理，但并未普遍胜出

从理论上，端到端半监督 VAE 比松耦合模型更一致，因为它让表示学习与分类目标共同优化。实验结果表明：

- 它在 v4 上明显优于 baseline VAE；
- 它在 reports-only 上没有超过纯监督模型；
- 它在 merged 数据集上接近最优，但在 `latent_dim = 4` 条件下仍略低于 `hybrid + random_forest`。

因此，端到端半监督 VAE 是一个合理的改进方向，但当前版本尚未形成稳定显著优势。

### 3.5 可视化建议

根据作业要求，结果部分建议在最终提交版本中补充以下统计图：

1. **柱状图**：比较不同模型在三个数据集上的 `F1`；
2. **柱状图**：比较不同模型在三个数据集上的 `AUROC`；
3. **ROC 曲线**：至少选择 baseline VAE、supervised-only、hybrid、end-to-end 四类方法进行对比；
4. **异常分数分布图**：展示 baseline VAE 在不同数据集上正常类与贫血类分数重叠程度；
5. **训练损失曲线**：展示 baseline VAE 和端到端模型在训练过程中的收敛情况。

若后续需要，我可以继续把这些图的绘制脚本补入项目中。

## 4. Conclusion

### 4.1 Results Analysis

本项目从实现 VAE 基线出发，逐步扩展到纯监督、松耦合混合以及端到端半监督方法，并在不同 CBC 数据条件下进行了系统比较。综合结果可以得出以下结论：

1. **VAE 基线在结构化、可分性较强的 v4 数据集上表现良好。**  
   baseline VAE 在 v4 上达到 `AUROC = 0.9223`、`F1 = 0.9155`，说明其作为贫血异常检测基线是可行的。

2. **v4 数据集不足以充分区分改进方法优劣。**  
   在 v4 上，纯监督随机森林已达到几乎完美结果，导致该数据集更适合验证模型可行性，而不适合严肃比较模型优劣。

3. **报告数据集更接近真实困难场景。**  
   当实验转向报告数据时，baseline VAE 的 AUROC 下降到 `0.5146`，而 `supervised_only + logistic_regression` 与 `hybrid + logistic_regression` 的 AUROC 分别为 `0.6202` 和 `0.6332`。这说明实际任务难度远高于 v4，同时也说明 baseline VAE 对数据条件高度敏感。

4. **混合模型具有一定价值，但优势不稳定。**  
   在某些场景，尤其是融合数据集上，hybrid 能取得较好结果；但在报告数据上，hybrid 并未稳定优于纯监督模型。

5. **端到端半监督 VAE 是合理的研究方向，但当前实现尚未形成决定性优势。**  
   它在理论上比松耦合模型更一致，实验上也在部分设定下接近最优，但仍需要进一步调参与结构优化。

从研究角度看，本项目最重要的发现并不是“某个改进模型必然更强”，而是：

> 模型性能高度依赖于数据条件。结构化程度高、标签纯度高、特征完整的数据集，更容易让 VAE 和监督模型都取得很高结果；而在更真实、更复杂的数据条件下，模型优劣会发生明显变化。

这说明：

- 不能只用一个“容易”的数据集评价方法有效性；
- 不能只看 AUROC，而忽略 F1、Recall 和 Specificity；
- 贫血检测任务中的模型设计必须与数据条件共同考虑。

本项目的意义主要体现在以下几个方面：

1. 完成了 VAE 基线模型的工程实现与系统评估；
2. 比较了纯 VAE、纯监督、混合和端到端半监督四类方法；
3. 揭示了不同 CBC 数据集之间任务难度的显著差异；
4. 明确了当前方法的适用边界，为后续改进提供了方向。

因此，该项目不是简单证明“某个模型一定最强”，而是系统研究：

> 基于 VAE 的贫血异常检测方法在不同数据条件下能做到什么程度，以及它的改进空间在哪里。

### 4.2 Future Improvements

后续工作可以从以下方向展开：

1. **增加特征维度**  
   当前报告数据只保留了 8 个指标，可进一步加入 `RDW`、`MPV`、分类百分比与绝对计数等指标。

2. **改善标签质量**  
   报告数据目前标签较粗，后续可以结合更细粒度的诊断信息提升监督信号质量。

3. **开展多 seed 或交叉验证实验**  
   当前结果以单次切分为主，后续应补充均值和标准差，提升结论稳健性。

4. **继续优化端到端半监督结构**  
   可进一步搜索 `latent_dim`、分类损失权重等关键超参数。

5. **尝试更适合异常检测的生成式方法**  
   例如 Deep SVDD-VAE、条件 VAE 或其他更强的半监督异常检测框架。

### 4.3 Future Study Topics

本项目后续可延伸的研究主题包括：

1. 医疗表格数据中的半监督异常检测；
2. 跨数据源 CBC 检测模型的鲁棒性研究；
3. 噪声标签条件下的贫血检测；
4. 结构化 CBC 数据与文本报告联合建模；
5. 面向临床使用的低误报、高召回贫血筛查模型设计。

## 5. Reference

以下参考文献建议在最终论文中统一采用同一种格式。本文先使用统一的编号格式列出：

1. D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *International Conference on Learning Representations (ICLR)*, 2014.
2. J. An and S. Cho, “Variational Autoencoder based Anomaly Detection using Reconstruction Probability,” *Special Lecture on IE*, vol. 2, no. 1, pp. 1-18, 2015.
3. L. Ruff, R. A. Vandermeulen, N. Görnitz, et al., “Deep One-Class Classification,” *International Conference on Machine Learning (ICML)*, 2018.
4. F. Pedregosa, G. Varoquaux, A. Gramfort, et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
5. A. Paszke, S. Gross, F. Massa, et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

---

## 附录：建议在答辩或提交时强调的主结论

如果需要用一段简洁的话概括本项目，可以使用下面这段：

> 本项目首先实现了基于 VAE 的 CBC 贫血异常检测基线，并在结构化 v4 数据集上取得了较高的 AUROC，证明该方法在规则、可分性强的数据条件下具有可行性。随后，我们引入纯监督、松耦合混合和端到端半监督模型，并在报告数据与融合数据上进行比较。实验结果表明，模型性能对数据质量、特征完整性和标签纯度高度敏感；当任务转向更复杂的数据环境时，传统 VAE 基线性能显著下降，而混合与半监督方法仅在部分设定下表现出有限增益。这说明，贫血异常检测的改进不仅依赖模型结构设计，也依赖数据条件本身。
