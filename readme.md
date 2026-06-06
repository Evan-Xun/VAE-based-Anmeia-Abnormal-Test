# 8 指标 CBC 报告数据集说明

当前项目不再使用融合数据集，也不再使用 `v4` 诊断数据。

现在项目只保留一个数据集：

- [cbc_8_features_reports_only.csv](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/cbc_8_features_reports_only.csv)

这个文件来自之前文本 CBC 报告数据的结构化结果，只包含报告来源样本，不包含 `diagnosed_cbc_data_v4.csv` 的任何样本。

## 数据集规模

总样本数：

```text
962
```

类别分布：

```text
Anemia  = 490
Healthy = 472
```

## 保留的 8 个指标

```text
WBC
RBC
HGB
HCT
MCV
MCH
MCHC
PLT
```

完整列：

```text
WBC,RBC,HGB,HCT,MCV,MCH,MCHC,PLT,Diagnosis,Source
```

其中 `Source` 目前固定为：

```text
report_text
```

## 默认数据入口

默认数据路径已经切换到：

- [src/dataset.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/src/dataset.py)
- [run_all.py](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/run_all.py)

默认文件：

```text
cbc_8_features_reports_only.csv
```

## 使用方式

```bash
./.venv/bin/python run_all.py \
  --data cbc_8_features_reports_only.csv \
  --cleaning range \
  --target anemia \
  --epochs 100 \
  --device cpu
```

如果不传 `--data`，项目默认也会使用这个数据集。

## 当前有效实验结果

当前与这个数据集一致的实验输出目录是：

- [results/report_dataset_run](/Users/fengxun/workspace/pyCharm_project/VAE-based-Anmeia-Abnormal-Test-master/results/report_dataset_run)

这是项目当前应当引用的报告数据正式结果。融合数据集对应的实验结果不再作为主线使用。
