#!/usr/bin/env python
"""Main program: one teacher/student linear-regression experiment.

Pipeline
--------
1. build the fixed teacher  w_bar ~ N(0, 1/N)
2. the teacher draws the training set  {(x_i, y_i)}, i = 1..P
3. the student fits that fixed set by full-batch gradient descent for `epoch` steps
4. every `test_every` epochs the teacher draws a *fresh* test set and we record
   the test MSE; the training MSE is recorded every epoch
5. write ``loss.csv`` (epoch, train_loss, test_loss, ...) and the loss curves
   into a folder named after the run, e.g. ``experiments/N32_P64_ep2000_te10/``

Usage
-----
    python main.py -N 32 -P 64 --epoch 2000 --test-every 10
    python main.py --help
"""

from __future__ import annotations

import sys
import time

import numpy as np

from config import ExperimentConfig, parse_args
from data import Teacher
from io_utils import save_history_csv, save_metadata
from model import Student, ridge_solution
from plot import plot_diagnostics, plot_loss_curves
from train import run_training


def describe(cfg: ExperimentConfig) -> None:
    print("=" * 68)
    print("Single-layer teacher/student experiment")
    print("-" * 68)
    print(f"  N (input dim)        : {cfg.N}")
    print(f"  P (train samples)    : {cfg.P}")
    print(f"  epochs               : {cfg.epoch}")
    print(f"  test every           : {cfg.test_every} epochs")
    print(f"  test set size        : {cfg.test_size} (fresh samples each test)")
    print(f"  student              : y = w^T x, {cfg.optimizer}, lr={cfg.lr:g}, "
          f"wd={cfg.weight_decay:g}, init={cfg.student_init}")
    print(f"  teacher              : w_bar ~ N(0, {cfg.teacher_scale:g}/N), "
          f"noise_std={cfg.noise_std:g}, noise floor={cfg.noise_std**2:.4g}")
    print(f"  seed                 : {cfg.seed}")
    print(f"  output folder        : {cfg.run_dir}")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    describe(cfg)

    t0 = time.perf_counter()

    # 1. teacher with its fixed weights
    teacher = Teacher(cfg)

    # 2. training set, drawn once and then reused for every gradient step
    train_set = teacher.sample(cfg.P)
    print(f"\n[data] train set: x {train_set.x.shape}, y {train_set.y.shape}")

    # 3. student + full-batch gradient descent
    student = Student(cfg, rng=_student_rng(cfg))
    print("[train] full-batch GD ...")
    hist = run_training(student, teacher, train_set, cfg)

    # 4. reference numbers: closed-form ridge solution of the same training set
    w_star = ridge_solution(train_set.x, train_set.y, cfg.weight_decay)
    ref_test = student.analytic_mse(teacher.w_bar, teacher.noise_floor())
    ref_star = float(((w_star - teacher.w_bar) ** 2).sum() + teacher.noise_floor())
    print(f"\n[check] regime: P {'>' if cfg.P > cfg.N else ('<' if cfg.P < cfg.N else '=')} N "
          f"-> {'overdetermined' if cfg.P > cfg.N else 'underdetermined'}")
    print(f"[check] analytic test MSE of learned w : {ref_test:.6f}")
    print(f"[check] analytic test MSE of ridge OLS : {ref_star:.6f}")

    # 5. persist results
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_history_csv(hist, run_dir / "loss.csv")
    fig_path = plot_loss_curves(hist, cfg, run_dir / "loss_curves.png")
    diag_path = plot_diagnostics(hist, cfg, run_dir / "diagnostics.png")
    meta_path = save_metadata(cfg, hist, run_dir / "metadata.json", extra={
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "analytic_test_mse_learned": ref_test,
        "analytic_test_mse_ridge_ols": ref_star,
        "teacher_w_bar": teacher.w_bar.tolist(),
        "student_w_final": student.w.tolist(),
    })

    print("\n[output]")
    for p in (csv_path, fig_path, diag_path, meta_path):
        print(f"  {p}")
    print(f"\n[done] final train MSE = {hist.final_train_loss:.6e}, "
          f"final test MSE = {hist.final_test_loss:.6e}, "
          f"best test MSE = {hist.best_test_loss:.6e}")
    return 0


def _student_rng(cfg: ExperimentConfig) -> np.random.Generator:
    # Independent of the data stream so that changing the student's init never
    # reshuffles the training/test sets.
    return np.random.default_rng(cfg.seed + 10_000)


if __name__ == "__main__":
    sys.exit(main())
