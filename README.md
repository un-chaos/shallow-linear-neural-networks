# 单层神经网络：teacher / student 线性回归实验

一个可以直接跑的单层（线性）神经网络实验框架：**teacher 网络生成数据，student 网络用 full-batch 梯度下降拟合，观察 train / test loss 曲线**。

---

## 1. 模型与数据

**teacher 网络**（权重固定，不训练）：

$$
y = \bar{w}^{\top}x + \epsilon,\qquad
x\sim\mathcal N(0, I_N),\qquad
\bar w \sim \mathcal N\!\left(0, \tfrac{1}{N}I_N\right),\qquad
\epsilon \sim \mathcal N(0, \sigma^2)
$$

- `x` 是 `N` 维输入，`\bar w` 是 teacher 的固定权重。
- `\bar w` 每个分量方差取 `1/N`，因此 `\bar w^T x` 的量级始终是 `O(1)`，不随 `N` 增长。
- `\epsilon` 是高斯标签噪声，`\sigma` 由 `--noise-std` 控制，对应不可约误差下界（noise floor）`\sigma^2`。

**student 网络**（被训练）：

$$
\hat y = w(t)^{\top}x,\qquad
w \leftarrow w - \eta\,\nabla_w \mathrm{MSE},\qquad
\mathrm{MSE}(w)=\frac1P\lVert Xw-y\rVert^2
$$

梯度有闭式解（纯 NumPy 手写，不用 autograd）：

$$
\nabla_w \mathrm{MSE} = \frac{2}{P}X^{\top}(Xw-y)
$$

> 若开启 `--weight-decay`，目标函数为 `MSE + wd * ||w||^2`，梯度额外加 `2*wd*w`。

---

## 2. 训练 / 测试协议（关键）

| 阶段 | 数据 | 频率 |
| --- | --- | --- |
| 训练 | teacher **一次性**生成 `P` 个样本，之后 student 反复对这同一批样本做 full-batch 迭代 | 每个 epoch 记一次 train loss |
| 测试 | teacher **每次测试都重新生成** `P` 个全新样本 | 每 `test_every` 个 epoch 测一次 test loss |

所以 test loss 衡量的是真正的泛化误差（在无穷多新数据上的期望风险），而不是对训练集的拟合程度。
每次测试还额外记录两个诊断量：

- `test_loss_analytic = ||w - \bar w||^2 + \sigma^2`：在**无穷多**新数据上的精确期望 MSE（理论上应等于测得的 `test_loss` 加上有限样本波动）；
- `weight_distance = ||w - \bar w||_2`：student 权重与 teacher 权重的距离，`0` 表示完美恢复。

---

## 3. 可调参数

四个主要参数：

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `-N` / `--N` | 输入维度（= student 参数量） | 32 |
| `-P` / `--P` | 训练样本数（也等于测试样本数） | 64 |
| `--epoch` | 训练 epoch 数（= full-batch 梯度步数） | 2000 |
| `--test-every` | 每多少个 epoch 测一次 test loss | 10 |

其他（都有默认值）：`--lr`、`--noise-std`、`--weight-decay`、`--n-test`、`--student-init`、`--seed`、`--out-root`、`--run-name`。

---

## 4. 运行

```bash
pip install -r requirements.txt

# 默认实验
python main.py

# 指定四个核心参数
python main.py -N 32 -P 64 --epoch 2000 --test-every 10

# 关闭中文/进度输出，改噪声与学习率
python main.py -N 16 -P 32 --epoch 5000 --test-every 25 --noise-std 0.2 --lr 0.05
```

---

## 5. 输出

结果写入 `experiments/<tag>/`，`<tag>` 由网络特征命名，例如
`experiments/N32_P64_ep2000_te10/`：

```
experiments/N32_P64_ep2000_te10/
├── loss.csv          # epoch, train_loss, test_loss, test_loss_analytic, weight_distance
├── loss_curves.png   # train 与 test loss 画在同一张图（含 PDF 矢量版）
├── diagnostics.png   # 实测 test loss vs 解析值；权重距离随 epoch 的变化
└── metadata.json     # 完整配置 + 汇总指标 + 最终的 w 与 \bar w
```

`loss.csv` 说明：`train_loss` 每个 epoch 都有；`test_loss` 只在测试的 epoch 上有值（其余留空）；
`epoch=0` 那一行是训练前（初始化）的基线，此时 `train_loss` 为空。
输出的曲线图里，红线是 test loss，蓝线是 train loss，灰色虚线是噪声下界 `σ²`。

> 如果同时运行多组实验，可以传 `--run-name` 指定目录名，避免 tag 冲突。

---

## 6. 代码结构

每个功能一个子程序/模块，主程序只负责编排：

| 文件 | 职责 |
| --- | --- |
| `config.py` | `ExperimentConfig` 数据类（含校验）+ 命令行解析 |
| `data.py` | `Teacher` 类：固定权重、生成训练/测试数据、噪声下界 |
| `model.py` | `Student` 类：前向、MSE 梯度、一步更新、`fit()`；另含解析 ridge 解 |
| `train.py` | `run_training()`：full-batch 训练主循环 + 测试调度 + `TrainingHistory` |
| `io_utils.py` | 写 `loss.csv` 与 `metadata.json` |
| `plot.py` | 画 loss 曲线与诊断图 |
| `main.py` | 主程序：解析参数 → 建 teacher/数据/student → 训练 → 存文件 → 打印汇总 |
| `smoke_test.py` | 自检：数值一致性、CSV 结构、loss 下降性 |

参数校验在 `ExperimentConfig.__post_init__` 里集中完成，非法输入（如 `P<=0`、`lr<=0`）会在开始计算前就报错。

---

## 7. 自检

```bash
python smoke_test.py
```

检查项：训练集形状、CSV 行列结构、train loss 相对初始值下降、实测 test loss 与解析值 `||w-\bar w||^2+\sigma^2` 的相对误差在有限样本波动范围内。

---

## 8. 数值提示

- student 与数据用**互相独立**的随机流（student 用 `seed+10000`），改初始化不会打乱数据集。
- 默认 `lr=0.1`，对 `E[xx^T]=I` 且 `P>=N` 的情形稳定；若 `P<N` 或输入协方差病态，full-batch GD 的收敛由 `X^TX/P` 的最大特征值决定，必要时调小 `--lr`。
- `--noise-std 0` 时 test loss 可以降到机器精度附近；此时曲线会直接跌到底噪，属于预期行为。
