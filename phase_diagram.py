#!/usr/bin/env python
"""N-P phase diagrams: colour = loss, over the whole (N, P) grid.

For every grid point the exact same pipeline as `main.py` runs: the teacher draws
one training set, the student fits it by full-batch gradient descent, and the
test loss is measured on freshly drawn data.  Several seeds are averaged so the
colour map is not dominated by one lucky sample.

Two requested diagrams:

    phase_final_test_loss.png   final test loss  (at the last epoch)
    phase_best_test_loss.png    best test loss   (min over the evaluated epochs)

plus two companions that make the picture readable:

    phase_train_loss.png        final train loss -- shows the P<N interpolation
    phase_analytic_test.png     ||w - w_bar||^2 + sigma^2, the infinite-data risk

Outputs land in ``phase/<name>/``: one PNG per quantity, a shared CSV, and the
CSV gets a wide-format companion for quick eyeballing.

Usage
-----
    python phase_diagram.py                       # default grid
    python phase_diagram.py --n-values 8 16 32 --p-values 8 16 32 64
    python phase_diagram.py --seeds 3 --epoch 1500 --color log
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

from config import ExperimentConfig  # noqa: E402
from data import Teacher  # noqa: E402
from model import Student, stable_lr_bound  # noqa: E402
from plot import configure_style  # noqa: E402
from train import run_training  # noqa: E402

QUANTITIES = [
    # (csv key,            heatmap title,                                 colourmap)
    ("final_test_loss",    "final test loss（最后一个 epoch）",           "viridis"),
    ("best_test_loss",     "best test loss（所有测试点的最小值）",         "viridis"),
    ("final_test_analytic", r"解析 test loss  $\|w-\bar w\|^2+\sigma^2$", "viridis"),
]


# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep (N, P) and draw phase diagrams of the loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-values", dest="n_values", type=int, nargs="+",
                   default=[8, 12, 16, 24, 32, 48, 64], help="N grid")
    p.add_argument("--p-values", dest="p_values", type=int, nargs="+",
                   default=[8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                   help="P grid")
    p.add_argument("--epoch", type=int, default=1500)
    p.add_argument("--test-every", dest="test_every", type=int, default=100)
    p.add_argument("--n-test", dest="n_test", type=int, default=100_000,
                   help="test-set size per measurement (keeps the colour map smooth)")
    p.add_argument("--seeds", type=int, default=3, help="seeds averaged per grid point")
    p.add_argument("--noise-std", dest="noise_std", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--color", choices=["log", "linear", "asinh"], default="log",
                   help="colour scale only; the axes stay linear")
    p.add_argument("--out-root", dest="out_root", type=str, default="phase")
    p.add_argument("--name", type=str, default=None, help="output folder name")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
def run_grid_point(n: int, p: int, args: argparse.Namespace) -> dict:
    """Average one (N, P) cell over ``args.seeds`` independent experiments.

    Full-batch GD diverges when ``lr >= 1/lambda_max`` of that sample's
    ``X^T X / P``, and that limit drops well below the default for small ``P/N``
    at large ``N`` (e.g. N=64, P=12 allows only lr < 0.084).  Each cell therefore
    uses ``min(requested lr, safe lr)`` and reports the value it actually used.
    """
    acc: dict[str, list[float]] = {
        "final_test_loss": [], "best_test_loss": [], "final_train_loss": [],
        "final_test_analytic": [], "final_weight_distance": [],
    }
    lr_used = args.lr
    for seed in range(args.seeds):
        cfg = ExperimentConfig(
            N=n, P=p, epoch=args.epoch, test_every=args.test_every,
            noise_std=args.noise_std, lr=args.lr, n_test=args.n_test,
            seed=seed, verbose=False,
        )
        teacher = Teacher(cfg)
        train_set = teacher.sample(cfg.P)

        # same data for every seed, so the bound is computed once per cell
        lr_used = min(args.lr, stable_lr_bound(train_set.x))
        if lr_used != cfg.lr:
            cfg = replace(cfg, lr=lr_used)

        student = Student(cfg, np.random.default_rng(cfg.seed + 10_000))
        hist = run_training(student, teacher, train_set, cfg)

        acc["final_test_loss"].append(hist.final_test_loss)
        acc["best_test_loss"].append(hist.best_test_loss)
        acc["final_train_loss"].append(hist.final_train_loss)
        acc["final_test_analytic"].append(hist.test_analytic[-1])
        acc["final_weight_distance"].append(hist.weight_distance[-1])

    row = {"N": n, "P": p, "P_over_N": p / n, "seeds": args.seeds,
           "lr_requested": args.lr, "lr_used": lr_used}
    for key, values in acc.items():
        row[key] = float(np.mean(values))
        row[f"{key}_std"] = float(np.std(values))
    return row


def collect(args: argparse.Namespace) -> list[dict]:
    rows = []
    total = len(args.n_values) * len(args.p_values)
    done = 0
    for p in args.p_values:
        for n in args.n_values:
            done += 1
            row = run_grid_point(n, p, args)
            rows.append(row)
            if not args.quiet:
                lr_note = ("" if row["lr_used"] == args.lr
                           else f"  [lr {args.lr:g}->{row['lr_used']:.3f} 稳定保护]")
                print(f"  [{done:>3}/{total}] N={n:>3} P={p:>3} (P/N={row['P_over_N']:.2f})  "
                      f"final_test={row['final_test_loss']:.3e}  "
                      f"best_test={row['best_test_loss']:.3e}  "
                      f"train={row['final_train_loss']:.2e}{lr_note}", flush=True)
    return rows


# ----------------------------------------------------------------------
def color_norm(mode: str, vmin: float, vmax: float):
    """Colour normalisation only -- the axes themselves stay linear."""
    if mode == "log":
        return LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    if mode == "asinh":
        # asinh keeps the small values visible without a log axis
        return matplotlib.colors.AsinhNorm(linear_width=max(vmax * 0.01, 1e-9),
                                           vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def draw_phase(rows: list[dict], key: str, title: str, cmap: str,
               args: argparse.Namespace, path: Path,
               values: dict | None = None, label: str = "MSE",
               norm_override=None) -> Path:
    """Heat map of ``key`` (or of an explicit ``values`` map) over the (N, P) grid."""
    n_values = sorted({r["N"] for r in rows})
    p_values = sorted({r["P"] for r in rows})
    grid = np.full((len(p_values), len(n_values)), np.nan)
    idx_n = {n: i for i, n in enumerate(n_values)}
    idx_p = {p: i for i, p in enumerate(p_values)}
    for r in rows:
        grid[idx_p[r["P"]], idx_n[r["N"]]] = (values[r["N"], r["P"]] if values
                                              else r[key])

    configure_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    norm = norm_override or color_norm(args.color, vmin, vmax)

    mesh = ax.pcolormesh(np.array(n_values), np.array(p_values), grid,
                         cmap=cmap, norm=norm, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(label + ("（颜色为对数刻度，坐标轴仍为线性）"
                            if isinstance(norm, LogNorm) else ""))

    # P = N boundary: everything below it is underdetermined
    lim = [min(n_values) * 0.8, max(n_values) * 1.25]
    ax.plot(lim, lim, color="white", ls="--", lw=1.6)
    ax.text(max(n_values) * 1.02, max(p_values) * 0.92, "P = N", color="black",
            fontsize=10, ha="left", va="top")
    ax.set_xlabel("N  (输入维度 / student 参数量)")
    ax.set_ylabel("P  (训练样本数)")
    ax.set_title(f"{title}\nseeds={args.seeds}  epoch={args.epoch}  "
                 f"lr={args.lr}  σ={args.noise_std}  n_test={args.n_test:,}",
                 fontsize=10)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.25)
    ax.set_ylim(min(p_values) * 0.8, max(p_values) * 1.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def draw_measurement_bias(rows: list[dict], args: argparse.Namespace, path: Path) -> Path:
    """Measured minus analytic test loss: validates the analytic column.

    Measured and analytic values agree once the student has converged, so their
    difference shows only the Monte-Carlo error of re-drawing ``n_test`` fresh
    samples at each test epoch.  A negative bias means the last test draw happened
    to be lucky.  (The best-vs-final difference is *always* negative here and
    below ~2e-6, i.e. pure noise -- that is why it gets no diagram of its own.)
    """
    n_values = sorted({r["N"] for r in rows})
    p_values = sorted({r["P"] for r in rows})
    grid = np.full((len(p_values), len(n_values)), np.nan)
    idx_n = {n: i for i, n in enumerate(n_values)}
    idx_p = {p: i for i, p in enumerate(p_values)}
    for r in rows:
        a = r["final_test_analytic"]
        grid[idx_p[r["P"]], idx_n[r["N"]]] = (
            (r["final_test_loss"] - a) / a if a > 0 else 0.0)

    configure_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    span = max(abs(float(np.nanmin(grid))), abs(float(np.nanmax(grid))), 1e-4)
    mesh = ax.pcolormesh(np.array(n_values), np.array(p_values), grid,
                         cmap="RdBu_r", norm=Normalize(vmin=-span, vmax=span),
                         shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("(实测 − 解析) / 解析")

    lim = [min(n_values) * 0.8, max(n_values) * 1.25]
    ax.plot(lim, lim, color="black", ls="--", lw=1.6)
    ax.text(max(n_values) * 1.02, max(p_values) * 0.92, "P = N", color="black",
            fontsize=10, ha="left", va="top")
    ax.set_xlabel("N  (输入维度 / student 参数量)")
    ax.set_ylabel("P  (训练样本数)")
    ax.set_title("实测 test loss 与解析值之差\n"
                 f"应当只是 n_test={args.n_test:,} 的蒙特卡洛误差，无系统偏差",
                 fontsize=10)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.25)
    ax.set_ylim(min(p_values) * 0.8, max(p_values) * 1.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def draw_interpolation(rows: list[dict], args: argparse.Namespace, path: Path) -> Path:
    """Binary map of the interpolation threshold.

    ``final_train_loss`` reaches machine precision exactly when the model can fit
    the sample perfectly, which happens for ``P >= N`` (at ``P = N`` with a
    full-rank sample the fit is exact too); for ``P < N`` it stops at the OLS
    floor ``sigma^2 (P-N)/P``.  Its raw values span 31 decades, so a continuous
    colour scale is useless there; this map shows the threshold instead, which is
    the sharpest feature in the grid.
    """
    threshold = 1e-12
    n_values = sorted({r["N"] for r in rows})
    p_values = sorted({r["P"] for r in rows})
    grid = np.zeros((len(p_values), len(n_values)))
    idx_n = {n: i for i, n in enumerate(n_values)}
    idx_p = {p: i for i, p in enumerate(p_values)}
    for r in rows:
        grid[idx_p[r["P"]], idx_n[r["N"]]] = (
            1.0 if r["final_train_loss"] < threshold else 0.0)

    configure_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    mesh = ax.pcolormesh(np.array(n_values), np.array(p_values), grid,
                         cmap="coolwarm", vmin=0.0, vmax=1.0, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels([f"未插值（train loss 高于 {threshold:g}）",
                             f"已插值到 0（train loss 低于 {threshold:g}）"])

    lim = [min(n_values) * 0.8, max(n_values) * 1.25]
    ax.plot(lim, lim, color="black", ls="--", lw=1.6)
    ax.text(max(n_values) * 1.02, max(p_values) * 0.92, "P = N", color="black",
            fontsize=10, ha="left", va="top")
    ax.set_xlabel("N  (输入维度 / student 参数量)")
    ax.set_ylabel("P  (训练样本数)")
    ax.set_title("训练集能否被完全插值\n"
                 f"P >= N 时可以；epoch={args.epoch} 的 final train loss，σ={args.noise_std}",
                 fontsize=10)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.25)
    ax.set_ylim(min(p_values) * 0.8, max(p_values) * 1.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or (f"N{'-'.join(map(str, args.n_values))}"
                         f"_P{'-'.join(map(str, args.p_values))}"
                         f"_ep{args.epoch}_s{args.seeds}")
    out_dir = Path(args.out_root) / name

    cells = len(args.n_values) * len(args.p_values)
    print("=" * 78)
    print(f"N-P phase diagram: {cells} cells x {args.seeds} seeds "
          f"= {cells * args.seeds} runs -> {out_dir}")
    print("=" * 78)

    rows = collect(args)
    csv_path = write_csv(rows, out_dir / "phase_grid.csv")

    paths = []
    for key, title, cmap in QUANTITIES:
        # The loss quantities span ~4 decades, so a log *colour* scale keeps both
        # the near-floor region and the overfitting region readable.  The axes
        # themselves stay linear.  The analytic column is used to show agreement.
        norm = LogNorm(vmin=1e-4, vmax=1.0) if args.color == "log" else None
        paths.append(draw_phase(rows, key, title, cmap, args,
                                out_dir / f"phase_{key}.png", norm_override=norm))

    # measurement vs analytic: validates the analytic column of the grid
    paths.append(draw_measurement_bias(rows, args,
                                       out_dir / "phase_measured_vs_analytic.png"))
    paths.append(draw_interpolation(rows, args, out_dir / "phase_interpolation.png"))

    print("\n" + "=" * 78)
    print(f"{'N':>4} {'P':>4} {'P/N':>5} {'final test':>12} {'best test':>12} "
          f"{'final train':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['N']:>4} {r['P']:>4} {r['P_over_N']:>5.2f} "
              f"{r['final_test_loss']:>12.3e} {r['best_test_loss']:>12.3e} "
              f"{r['final_train_loss']:>12.3e}")
    print("=" * 78)
    print(f"[output] {csv_path}")
    for p in paths:
        print(f"         {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
