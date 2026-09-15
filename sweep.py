#!/usr/bin/env python
"""Run a sweep over N / P / epoch and summarise the resulting runs.

Each grid point is one ordinary experiment (same pipeline as ``main.py``), so
every run gets its own ``experiments/<tag>/`` folder with ``loss.csv`` and the
curves.  In addition this script writes, inside the sweep folder:

    sweep_summary.csv   -- one row per run with the headline numbers
    sweep_test_loss.png -- the test-loss curves of all runs on one figure

Usage
-----
    python sweep.py --N 16 32 64 --P 32 64 128 --epoch 2000 --test-every 10
    python sweep.py --N 8 16 --P 8 16 32 --noise-std 0.2 --lr 0.05
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ExperimentConfig
from data import Teacher
from io_utils import save_history_csv, save_metadata
from model import Student
from plot import configure_style, plot_loss_curves
from train import run_training


def parse_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep N, P and epoch for the teacher/student linear experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-N", "--N", type=int, nargs="+", default=[16, 32, 64],
                   help="one or more input dimensions")
    p.add_argument("-P", "--P", type=int, nargs="+", default=[32, 64, 128],
                   help="one or more training-set sizes")
    p.add_argument("--epoch", type=int, nargs="+", default=[2000],
                   help="one or more training lengths")
    p.add_argument("--test-every", dest="test_every", type=int, default=10,
                   help="evaluate the test loss every this many epochs")
    p.add_argument("--noise-std", dest="noise_std", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", dest="out_root", type=str, default="experiments")
    p.add_argument("--sweep-name", dest="sweep_name", type=str, default=None,
                   help="folder for the summary; default: derived from the grid")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def build_grid(args: argparse.Namespace) -> list[ExperimentConfig]:
    """Cartesian product of (N, P, epoch) as validated configs."""
    base = ExperimentConfig(
        test_every=args.test_every,
        noise_std=args.noise_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        out_root=args.out_root,
        verbose=not args.quiet,
    )
    return [replace(base, N=n, P=p, epoch=e)
            for n, p, e in product(args.N, args.P, args.epoch)]


def run_one(cfg: ExperimentConfig) -> dict:
    """Run a single experiment and return its summary row."""
    teacher = Teacher(cfg)
    train_set = teacher.sample(cfg.P)
    student = Student(cfg, np.random.default_rng(cfg.seed + 10_000))

    if cfg.verbose:
        print(f"\n>>> N={cfg.N} P={cfg.P} epoch={cfg.epoch} "
              f"(P{'>' if cfg.P > cfg.N else '<'}N)")
    hist = run_training(student, teacher, train_set, cfg)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(hist, run_dir / "loss.csv")
    plot_loss_curves(hist, cfg, run_dir / "loss_curves.png")
    save_metadata(cfg, hist, run_dir / "metadata.json")

    return {
        "tag": cfg.tag,
        "N": cfg.N,
        "P": cfg.P,
        "epoch": cfg.epoch,
        "P_over_N": cfg.P / cfg.N,
        "lr": cfg.lr,
        "noise_std": cfg.noise_std,
        "seed": cfg.seed,
        "final_train_loss": hist.final_train_loss,
        "final_test_loss": hist.final_test_loss,
        "best_test_loss": hist.best_test_loss,
        "final_test_loss_analytic": hist.test_analytic[-1],
        "final_weight_distance": hist.weight_distance[-1],
        "run_dir": str(run_dir),
    }


def write_summary(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_sweep(rows: list[dict], sweep_dir: Path, sweep_name: str) -> Path:
    """Overlay the test-loss curves of every run in the sweep."""
    configure_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for row in rows:
        epochs, test_loss = _read_curve(Path(row["run_dir"]) / "loss.csv")
        ax.plot(epochs, test_loss, lw=1.4,
                label=f"N={row['N']}, P={row['P']}, ep={row['epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test MSE")
    ax.set_title(f"Test loss sweep  |  {sweep_name}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = sweep_dir / "sweep_test_loss.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _read_curve(path: Path) -> tuple[list[int], list[float]]:
    """Read (epochs, test_loss) pairs from a run's CSV."""
    epochs, test_loss = [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["test_loss"]:
                epochs.append(int(row["epoch"]))
                test_loss.append(float(row["test_loss"]))
    return epochs, test_loss


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(argv)
    grid = build_grid(args)
    sweep_name = args.sweep_name or (
        f"sweep_N{'-'.join(map(str, args.N))}_P{'-'.join(map(str, args.P))}"
        f"_ep{'-'.join(map(str, args.epoch))}"
    )
    sweep_dir = Path(args.out_root) / sweep_name

    print("=" * 68)
    print(f"Sweep over {len(grid)} configurations -> {sweep_dir}")
    print("=" * 68)

    rows = [run_one(cfg) for cfg in grid]

    summary = write_summary(rows, sweep_dir / "sweep_summary.csv")
    figure = plot_sweep(rows, sweep_dir, sweep_name)

    print("\n" + "=" * 68)
    print(f"{'N':>5} {'P':>5} {'epoch':>7} {'train':>12} {'test':>12} {'best test':>12}")
    print("-" * 68)
    for r in rows:
        print(f"{r['N']:>5} {r['P']:>5} {r['epoch']:>7} "
              f"{r['final_train_loss']:>12.4e} {r['final_test_loss']:>12.4e} "
              f"{r['best_test_loss']:>12.4e}")
    print("=" * 68)
    print(f"[output] {summary}\n         {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
