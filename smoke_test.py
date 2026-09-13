#!/usr/bin/env python
"""Self-checks for the experiment pipeline -- run with `python smoke_test.py`.

These are plain asserts (no test framework needed) covering:
  1. data shapes and reproducible teacher weights
  2. the test schedule forced by ``test_every``
  3. training actually reduces the loss and recovers the teacher weights
  4. CSV structure, and the measured test MSE matching ``||w-w_bar||^2+sigma^2``
  5. analytic MSE matching a large-sample Monte-Carlo estimate
  6. config validation rejecting illegal parameters
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from data import Teacher
from io_utils import save_history_csv
from model import Student
from train import run_training


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(name)


def test_shapes_and_reproducibility() -> None:
    cfg = ExperimentConfig(N=16, P=32, epoch=5, test_every=2, seed=7, verbose=False)
    t1, t2 = Teacher(cfg), Teacher(cfg)
    check("teacher weights reproducible for a fixed seed",
          np.allclose(t1.w_bar, t2.w_bar))

    ds = t1.sample(cfg.P)
    check("training set shapes", ds.x.shape == (cfg.P, cfg.N) and ds.y.shape == (cfg.P,))
    check("test_size defaults to P", cfg.test_size == cfg.P)

    ds2 = t1.sample(cfg.P)
    check("successive draws are independent", not np.allclose(ds.y, ds2.y))


def test_test_schedule() -> None:
    cfg = ExperimentConfig(N=8, P=16, epoch=25, test_every=10, verbose=False)
    hist = run_training(Student(cfg, np.random.default_rng(0)), Teacher(cfg),
                        Teacher(cfg).sample(cfg.P), cfg)
    check("test schedule every 10 epochs, plus epoch 0 and the last epoch",
          hist.test_epoch == [0, 10, 20, 25], f"got {hist.test_epoch}")
    check("train loss recorded every epoch", hist.train_epoch == list(range(1, 26)))


def test_training_converges() -> None:
    # With P > N the training set is consistent, so GD drives w to the OLS
    # solution.  The training loss then floors at the label noise that is
    # orthogonal to the data span: the OLS residual is (I - H) * epsilon with H
    # the hat matrix, so E[train MSE] = sigma^2 * (P - N) / P.  That floor is a
    # property of the data, not of the optimiser, so we assert on it together
    # with the weight distance and the analytic (infinite-data) risk.
    cfg = ExperimentConfig(N=16, P=64, epoch=4000, test_every=400,
                           noise_std=0.05, lr=0.1, seed=3, verbose=False)
    teacher = Teacher(cfg)
    train_set = teacher.sample(cfg.P)
    student = Student(cfg, np.random.default_rng(0))
    hist = run_training(student, teacher, train_set, cfg)

    noise_floor = cfg.noise_std**2
    train_floor = noise_floor * (cfg.P - cfg.N) / cfg.P
    check("train loss decreases", hist.final_train_loss < hist.train_loss[0],
          f"{hist.train_loss[0]:.3e} -> {hist.final_train_loss:.3e}")
    check("train loss reaches the OLS floor ~ sigma^2*(P-N)/P",
          hist.final_train_loss < 3.0 * train_floor,
          f"train MSE = {hist.final_train_loss:.3e}, floor ~ {train_floor:.3e}")
    check("student recovers the teacher weights",
          hist.weight_distance[-1] < 0.05, f"||w-w_bar|| = {hist.weight_distance[-1]:.3e}")
    check("test loss approaches the noise floor and does not beat it",
          noise_floor * 0.9 < hist.final_test_loss < noise_floor * 1.3,
          f"test MSE = {hist.final_test_loss:.3e}, sigma^2 = {noise_floor:.3e}")


def test_trajectory_matches_closed_form_gd() -> None:
    """The recorded training losses must equal the exact iterated GD map."""
    cfg = ExperimentConfig(N=12, P=20, epoch=50, test_every=25,
                           noise_std=0.1, lr=0.07, seed=11, verbose=False)
    teacher = Teacher(cfg)
    train_set = teacher.sample(cfg.P)
    student = Student(cfg, np.random.default_rng(0))
    hist = run_training(student, teacher, train_set, cfg)

    # w_{t+1} = (I - lr * 2/P * X^T X) w_t + lr * 2/P * X^T y  is linear, so the
    # whole trajectory can be reproduced with matrix multiplications.
    x, y = train_set.x, train_set.y
    a = np.eye(cfg.N) - cfg.lr * (2.0 / cfg.P) * (x.T @ x)
    b = cfg.lr * (2.0 / cfg.P) * (x.T @ y)
    w = np.zeros(cfg.N)
    exact = []
    for _ in range(cfg.epoch):
        w = a @ w + b
        exact.append(float(np.mean((x @ w - y) ** 2)))

    rel = np.max(np.abs(np.array(exact) - np.array(hist.train_loss))) / exact[-1]
    check("recorded train losses match the closed-form GD recurrence", rel < 1e-9,
          f"max relative deviation = {rel:.2e}")


def test_underdetermined_regime() -> None:
    """With P < N many interpolators exist; GD picks the minimum-norm one."""
    cfg = ExperimentConfig(N=40, P=20, epoch=20000, test_every=20000,
                           noise_std=0.0, lr=0.05, seed=2, verbose=False)
    teacher = Teacher(cfg)
    train_set = teacher.sample(cfg.P)
    student = Student(cfg, np.random.default_rng(0))
    hist = run_training(student, teacher, train_set, cfg)

    check("noise-free training set is interpolated",
          hist.final_train_loss < 1e-12, f"train MSE = {hist.final_train_loss:.3e}")
    # Minimum-norm interpolation w = X^+ y is the limit point of GD from zeros.
    w_minnorm = np.linalg.pinv(train_set.x) @ train_set.y
    gap = np.linalg.norm(student.w - w_minnorm) / np.linalg.norm(w_minnorm)
    check("GD from zero init converges to the minimum-norm interpolator", gap < 1e-3,
          f"relative gap = {gap:.2e}")
    check("generalisation is worse than the noise-free floor",
          hist.final_test_loss > 1e-3,
          f"test MSE = {hist.final_test_loss:.3e} (P<N overfits the sample)")


def test_csv_and_analytic_consistency() -> None:
    cfg = ExperimentConfig(N=16, P=64, epoch=300, test_every=50,
                           noise_std=0.1, lr=0.1, seed=1, verbose=False)
    teacher = Teacher(cfg)
    student = Student(cfg, np.random.default_rng(0))
    hist = run_training(student, teacher, teacher.sample(cfg.P), cfg)

    # A plain directory inside the workspace (tempfile would tighten the ACL on
    # Windows, which sandboxed runs cannot write into).
    scratch = Path(".tmp_smoke_csv")
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        path = save_history_csv(hist, scratch / "loss.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    check("CSV header", list(rows[0].keys()) ==
          ["epoch", "train_loss", "test_loss", "test_loss_analytic", "weight_distance"])
    check("CSV has one row per epoch plus the epoch-0 baseline",
          len(rows) == cfg.epoch + 1, f"got {len(rows)}")
    check("epoch 0 row is the untrained baseline",
          rows[0]["epoch"] == "0" and rows[0]["train_loss"] == "" and rows[0]["test_loss"] != "")
    test_rows = [r for r in rows if r["test_loss"]]
    check("test_loss only on test epochs", len(test_rows) == len(hist.test_epoch))
    check("train_loss on every epoch",
          all(r["train_loss"] for r in rows if r["epoch"] != "0"))

    # A P-sample test set estimates the risk with error ~ 2*||w-w_bar||/sqrt(P),
    # which is large exactly when the student is still far from the teacher.  So
    # for small P we only assert the estimator is unbiased on average and that
    # the logged analytic column follows the measured curve.
    gaps = np.array(hist.test_loss) - np.array(hist.test_analytic)
    bias = abs(gaps.mean()) / max(hist.test_analytic)
    check("P-sample test-loss estimates are unbiased in the mean", bias < 0.1,
          f"mean relative bias = {bias:.2%} over {len(gaps)} test points")
    check("measured test MSE never falls far below the noise floor",
          min(hist.test_loss) > cfg.noise_std**2 * 0.8,
          f"min test MSE = {min(hist.test_loss):.3e}, sigma^2 = {cfg.noise_std**2:.3e}")
    check("measured test MSE decreases overall",
          hist.test_loss[-1] < hist.test_loss[0],
          f"{hist.test_loss[0]:.3e} -> {hist.test_loss[-1]:.3e}")


def test_large_test_set_matches_analytic_every_epoch() -> None:
    """With a large fresh test set, measured and analytic curves must coincide."""
    cfg = ExperimentConfig(N=16, P=64, epoch=200, test_every=100, n_test=100_000,
                           noise_std=0.1, lr=0.1, seed=5, verbose=False)
    teacher = Teacher(cfg)
    student = Student(cfg, np.random.default_rng(0))
    hist = run_training(student, teacher, teacher.sample(cfg.P), cfg)

    worst = max(abs(t - a) / a for t, a in zip(hist.test_loss, hist.test_analytic))
    check("empirical risk matches ||w-w_bar||^2+sigma^2 at every test epoch",
          worst < 0.02, f"worst relative gap = {worst:.3%} over "
                        f"{len(hist.test_loss)} test points")
    check("n_test override decouples test size from P",
          cfg.test_size == 100_000 and len(hist.test_epoch) == 3,
          f"test epochs = {hist.test_epoch}")



def test_analytic_risk_matches_monte_carlo() -> None:
    """The closed-form risk must match a direct 200k-sample estimate."""
    cfg = ExperimentConfig(N=16, P=64, epoch=200, test_every=50,
                           noise_std=0.1, lr=0.1, seed=5, verbose=False)
    teacher = Teacher(cfg)
    student = Student(cfg, np.random.default_rng(0))
    run_training(student, teacher, teacher.sample(cfg.P), cfg)

    big = teacher.sample(200_000)
    mc = student.mse(big.x, big.y)
    analytic = student.analytic_mse(teacher.w_bar, cfg.noise_std**2)
    rel = abs(mc - analytic) / analytic
    check("analytic risk matches a 200k-sample Monte-Carlo estimate", rel < 0.02,
          f"MC={mc:.6f}, analytic={analytic:.6f}, rel={rel:.3%}")


def test_config_validation() -> None:
    for kwargs, label in [
        ({"N": 0}, "N<=0"),
        ({"P": 0}, "P<=0"),
        ({"epoch": 0}, "epoch<=0"),
        ({"test_every": 0}, "test_every<=0"),
        ({"lr": 0.0}, "lr<=0"),
        ({"noise_std": -1.0}, "noise_std<0"),
        ({"optimizer": "adam"}, "unsupported optimizer"),
    ]:
        try:
            ExperimentConfig(verbose=False, **kwargs)
        except (ValueError, NotImplementedError):
            check(f"rejects {label}", True)
        else:
            check(f"rejects {label}", False)


def main() -> int:
    tests = [
        test_shapes_and_reproducibility,
        test_test_schedule,
        test_training_converges,
        test_trajectory_matches_closed_form_gd,
        test_underdetermined_regime,
        test_csv_and_analytic_consistency,
        test_large_test_set_matches_analytic_every_epoch,
        test_analytic_risk_matches_monte_carlo,
        test_config_validation,
    ]
    print("=" * 68)
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 68)
    print(f"ALL {len(tests)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
