#!/usr/bin/env python
"""Tuning reference: how large must `epoch` and `lr` be for a given (N, P)?

The training loss of this model is a quadratic function of w, and full-batch
gradient descent on a quadratic is exactly a linear iteration:

    w_{t+1} = (I - 2*lr*G) w_t + ...,        G = X^T X / P

so every eigenmode of G decays by a factor (1 - 2*lr*lambda) per step.  Two
consequences drive all practical tuning:

    stability   : lr < 1 / lambda_max          (else the loss diverges)
    speed       : governed by lambda_min, i.e. by the condition number kappa

Because G = X^T X / P with i.i.d. standard-normal rows, its spectrum is the
Marchenko-Pastur law.  With y = N/P (the inverse aspect ratio):

    N/P <= 1 (overdetermined): lambda in [ (1-sqrt(y))^2 , (1+sqrt(y))^2 ]
    N/P >  1 (underdetermined): a point mass at 0, plus
                               [ (1-sqrt(P/N))^2 , (1+sqrt(P/N))^2 ]

This script measures the actual rate from the training trajectory, compares it
with lambda_max, and reports the epochs needed per P/N ratio.

Run with:  python convergence_reference.py
"""

from __future__ import annotations

import numpy as np

from config import ExperimentConfig
from data import Teacher
from model import Student

N = 32
NOISE = 0.01
LR = 0.1
PROBE_EPOCHS = 3000


def marchenko_pastur_edges(n: int, p: int) -> tuple[float, float]:
    """Predicted [lambda_min, lambda_max] of X^T X / P for N(0,1) rows."""
    if n <= p:                     # overdetermined: aspect ratio y = N/P <= 1
        y = n / p
        s = np.sqrt(y)
        return float((1 - s) ** 2), float((1 + s) ** 2)
    # underdetermined: P < N, so X^T X / P is singular (one edge at 0)
    y = p / n
    s = np.sqrt(y)
    return 0.0, float((1 + s) ** 2)


def measure(cfg: ExperimentConfig, probe: int = PROBE_EPOCHS,
            tol: float = 0.01) -> dict:
    """Measure how many epochs GD needs to shrink its distance to the optimum."""
    teacher = Teacher(cfg)
    train_set = teacher.sample(cfg.P)
    student = Student(cfg, np.random.default_rng(0))
    x, y = train_set.x, train_set.y

    gram = x.T @ x / cfg.P
    eig = np.linalg.eigvalsh(gram)
    lam_min, lam_max = float(eig[0]), float(eig[-1])

    w_opt = (np.linalg.solve(x.T @ x + 1e-12 * np.eye(cfg.N), x.T @ y)
             if cfg.P >= cfg.N else np.linalg.pinv(x) @ y)

    def excess() -> float:
        return float(np.sum((student.w - w_opt) ** 2))

    eps0 = excess()
    epoch_hit = 0
    for t in range(1, probe + 1):
        student.step(x, y)
        if excess() <= tol * eps0:
            epoch_hit = t
            break

    # Slowest mode: (1 - 2*lr*lam_min)^t, giving the closed-form epoch count.
    slow = cfg.lr * 2.0 * lam_min
    epochs_pred = (np.log(1.0 / tol) / -np.log1p(-slow)
                   if 0 < slow < 1 else float("inf"))

    return {
        "lam_min": lam_min,
        "lam_max": lam_max,
        "kappa": lam_max / lam_min if lam_min > 0 else float("inf"),
        "lr_max_theory": 1.0 / lam_max,
        "epochs_obs": epoch_hit,
        "epochs_pred": epochs_pred,
        "pred_lam_max": marchenko_pastur_edges(cfg.N, cfg.P)[1],
        "test_floor": float(np.sum((w_opt - teacher.w_bar) ** 2) + NOISE**2),
    }


def main() -> None:
    print(f"N = {N}, sigma = {NOISE}, lr = {LR}, probe = {PROBE_EPOCHS} epochs; "
          f"判据 = 距最优解的距离降 100 倍")
    print("=" * 104)
    print(f"{'P':>5} {'P/N':>5} {'lam_max':>8} {'MP pred':>8} {'lr 上限':>8} "
          f"{'lam_min':>8} {'kappa':>9} {'ep 实测':>8} {'ep 理论':>9} {'test 下限':>11}")
    print("-" * 104)
    for ratio in (0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0):
        p = max(1, round(N * ratio))
        cfg = ExperimentConfig(N=N, P=p, epoch=1, test_every=1,
                               noise_std=NOISE, lr=LR, verbose=False)
        m = measure(cfg)
        obs = m["epochs_obs"] if m["epochs_obs"] else f">{PROBE_EPOCHS}"
        pred = "inf" if m["epochs_pred"] == float("inf") else f"{m['epochs_pred']:,.0f}"
        print(f"{p:>5} {ratio:>5.1f} {m['lam_max']:>8.3f} {m['pred_lam_max']:>8.3f} "
              f"{m['lr_max_theory']:>8.3f} {m['lam_min']:>8.4f} {m['kappa']:>9.2f} "
              f"{str(obs):>8} {pred:>9} {m['test_floor']:>11.3e}")
    print("=" * 104)
    print("lr 上限   : 1/lam_max，超过这个值训练直接发散（P/N=1 时约 0.27，所以 lr=0.1 安全）")
    print("ep 理论   : 最慢模式 (1-2*lr*lam_min)^t 降到 1% 所需步数，是保守上界")
    print("test 下限 : ||w_ols - w_bar||^2 + sigma^2，这组 (N,P) 能达到的最好 test loss")
    print("P/N <= 1  : 样本噪声被拟合进权重，test 下限远高于 sigma^2；P/N >= 4 后基本等于 sigma^2")


if __name__ == "__main__":
    main()
