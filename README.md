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
> 注意 `experiments/` 在 `.gitignore` 里：结果不入库，仓库只保存代码，随时可用命令重新生成。

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
| `convergence_reference.py` | 调参参考：实测各 `P/N` 下的收敛速度、`lr` 稳定上限、可达的 test 下限 |
| `main.py` | 主程序：解析参数 → 建 teacher/数据/student → 训练 → 存文件 → 打印汇总 |
| `sweep.py` | 批量扫 `N`/`P`/`epoch`：每个格点跑一次完整实验，另出汇总表与叠加对比图 |
| `smoke_test.py` | 自检：数值一致性、CSV 结构、loss 下降性 |
| `docs/pipeline.html` | **流程图**（可交互，浏览器打开）：实验步骤、数据流与两个关键数值性质 |
| `docs/pipeline.workflow.json` | 流程图的源文件；改完用 archify 重新生成 HTML |

参数校验在 `ExperimentConfig.__post_init__` 里集中完成，非法输入（如 `P<=0`、`lr<=0`）会在开始计算前就报错。

---

## 7. 怎么调参

**参数只有一个入口：命令行。** 定义在 `config.py`（`ExperimentConfig` + `parse_args`），`main.py` 与
`sweep.py` 共用同一套；也可以改 dataclass 字段默认值，但那样会改动所有实验的可复现记录，不如用命令行。

### 调 epoch 前先看 P/N

训练损失对 `w` 是二次函数，full-batch GD 因此是一个线性迭代
`w_{t+1} = (I - 2·lr·G) w_t + …`（`G = XᵀX/P`）：每个特征方向按 `(1 - 2·lr·λ)` 衰减，
所以**需要多少 epoch 由最小特征值 `λ_min` 决定，即由条件数 κ 决定**，而 κ 由 `P/N` 决定。
`G` 的谱服从 Marchenko–Pastur 律，`N/P ≤ 1` 时 `λ ∈ [(1-√(N/P))², (1+√(N/P))²]`。

实测（`N=32`、`σ=0.01`、`lr=0.1`，误差降 100 倍所需 epoch）：

| P/N | κ | `lr` 上限 `1/λ_max` | epoch 实测 | epoch 理论 | 可达 test 下限 |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 奇异（λ_min = 0） | 0.19 | 18 | ∞ | 3.2e-1 |
| **1.0** | **6963** | 0.27 | **>3000** | 43812 | 9.2e-3 |
| 1.5 | 49 | 0.38 | 68 | 424 | 3.0e-4 |
| 2.0 | 26 | 0.41 | 43 | 241 | 1.9e-4 |
| 4.0 | 6.7 | 0.54 | 22 | 81 | 1.3e-4 |
| 8.0 | 3.8 | 0.59 | 15 | 50 | 1.1e-4 |
| 16.0 | 2.4 | 0.68 | 14 | 36 | 1.09e-4 |

（`λ_min` 与 κ 随具体样本有波动；`epoch 理论` 是最慢模式 `(1-2·lr·λ_min)^t` 的保守上界，
实测更快是因为初始误差主要落在特征值大的方向上。`可达 test 下限` = `‖w_ols-w̄‖² + σ²`。）

三条结论：

1. **`P/N ≥ 2` 用默认 `lr=0.1`，`epoch` 给 500 足够**；`P/N ≥ 4` 时 200 就够。
2. **避开 `P/N = 1`**：数据矩阵接近奇异（κ≈7000），GD 在最小特征值方向上爬得极慢，
   而且这一档 test loss 比噪声地板高两个数量级（样本噪声被拟合进权重）。
3. **`lr` 必须小于 `1/λ_max`**，实测在 0.19~0.68 之间，所以默认 0.1 对所有配置都稳定；
   想加速可到 0.3（`P/N ≥ 2` 时仍在上限内），超过上限会直接发散。

重新测量（换 `N` 或 `σ` 后建议重跑）：

```bash
python convergence_reference.py
```

### 三套现成配方

```bash
# A. 标准泛化实验：曲线最干净，test 收敛到噪声地板
python main.py -N 32 -P 128 --epoch 500 --test-every 5 --n-test 100000
#    -> train 7.1e-5, test 1.27e-4 (sigma^2 = 1e-4)

# B. 过拟合 / P<N 实验：训练集插值到机器精度，test 远高于地板，且不会 grok
python main.py -N 64 -P 16 --epoch 2000 --test-every 20
#    -> train 1.5e-35, test 0.80

# C. 扫参数找规律
python sweep.py -N 8 16 --P 8 16 32 --epoch 1000 --test-every 100
```

配方 A 里的 `--n-test 100000` 是关键：默认测试集只有 `P` 个样本，蒙特卡洛噪声约
`2‖w-w̄‖/√P`，会让 test 曲线抖动明显；想要 test 曲线贴合解析值就用大测试集。

### 症状速查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| loss 变 NaN / 爆炸 | `lr > 1/λ_max` | 把 `--lr` 降到 0.1 以下 |
| test 曲线抖得看不出趋势 | 测试集太小（= P） | 加 `--n-test 100000` |
| train loss 停在地板不动 | 正常：噪声在列空间正交补上的投影 | 想更低就增大 `P`（地板 `σ²(P-N)/P`） |
| `P = N` 收敛特别慢 | 数据矩阵接近奇异 | 换 `P/N ≥ 1.5` |
| 想看纯泛化、无噪声 | — | `--noise-std 0` |

---

## 8. 流程图

用浏览器打开 `docs/pipeline.html`（自包含单文件，无需联网）：

- 从左到右是「准备 → 训练与测试 → 交付」三个阶段，每行 lane 对应一个源文件；
- 主线走 `main.py`：读参数 → 取全量样本 → 更新 w 并记 loss → 写 `loss.csv`；
- 虚线回环表示「epoch 未满就继续」，即训练循环；
- 下方三张卡片给出四个可调旋钮、train / test 的关键区别，以及读曲线时要知道的两个数值性质。

改完图之后重新生成：

```bash
node ~/.dsh/skills/archify/bin/archify.mjs deliver workflow docs/pipeline.workflow.json docs/pipeline.html --quality showcase --json
```

---

## 9. 批量扫参数

```bash
python sweep.py -N 8 16 --P 8 16 32 --epoch 1000 --test-every 100
```

对 `(N, P, epoch)` 的笛卡尔积逐点跑实验，每个格点仍然写自己的 `experiments/<tag>/`，
另外在 `experiments/sweep_<...>/` 下生成：

- `sweep_summary.csv`：每个格点一行，含 `final_train_loss` / `final_test_loss` / `best_test_loss` / `final_weight_distance` 等；
- `sweep_test_loss.png`：所有格点的 test loss 曲线叠加在同一张图。

典型输出（`--epoch 1000 --test-every 100`，`σ=0.01`）：

| N | P | final train | final test |
| --- | --- | --- | --- |
| 8 | 32 | 7.9e-05 | 9.6e-05 |
| 16 | 8 | 5.4e-35 | 3.6e-01 |
| 16 | 32 | 3.8e-05 | 3.1e-04 |

`P < N` 时（如 `N=16, P=8`）训练集被完全插值到机器精度，但 test loss 高出噪声地板三个数量级——
这就是过参数化下的过拟合；`P >= N` 时 test loss 落到噪声地板 `σ²` 附近。

---

## 10. 自检

```bash
python smoke_test.py
```

9 项检查，包括：训练集形状与随机流独立性、测试调度是否恰好落在 `test_every` 与最后一个 epoch、
**训练损失轨迹与闭式 GD 递推 `w <- (I - 2ηXᵀX/P)w + 2ηXᵀy/P` 逐点吻合**、
`P<N` 时是否收敛到最小范数插值解、训练损失是否落到 OLS 噪声地板 `σ²(P-N)/P`、
以及解析风险 `||w-\bar w||²+σ²` 与 20 万样本蒙特卡洛估计是否一致。

---

## 11. 数值提示

- student 与数据用**互相独立**的随机流（student 用 `seed+10000`），改初始化不会打乱数据集。
- 默认 `lr=0.1`，对 `E[xx^T]=I` 且 `P>=N` 的情形稳定；若 `P<N` 或输入协方差病态，full-batch GD 的收敛由 `X^TX/P` 的最大特征值决定，必要时调小 `--lr`。
- **训练损失有下界**：噪声在数据列空间正交补上的投影给出 `E[train MSE] = σ²(P-N)/P`（`P>N` 时），
  所以 train loss 不会掉到 0，这是数据本身的性质，不是优化没收敛。
- **测试损失是蒙特卡洛估计**：每次只抽 `P` 个新样本，其波动约为 `2||w-\bar w||/√P`，
  因此在学生离 teacher 还远时（尤其 `epoch=0`）测得值会明显偏离解析值。
  想要平滑的曲线就加大 `--n-test`（例如 `--n-test 100000`），此时测得值与解析值应吻合到 1% 以内。
- `--noise-std 0` 时 test loss 可以降到机器精度附近；此时曲线会直接跌到底噪，属于预期行为。
