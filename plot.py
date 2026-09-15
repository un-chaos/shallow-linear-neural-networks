"""Plotting helpers: train and test loss on one set of axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from config import ExperimentConfig  # noqa: E402
from train import TrainingHistory  # noqa: E402


def configure_style() -> None:
    """Pick a font stack that can render both Latin and CJK labels."""
    plt.rcParams.update({
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "DejaVu Sans", "Arial",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })


def plot_loss_curves(hist: TrainingHistory, cfg: ExperimentConfig, path: Path) -> Path:
    """Train loss (every epoch) and test loss (fresh data) on one pair of axes."""
    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.plot(hist.train_epoch, hist.train_loss, color="#1f77b4", lw=1.2,
            label="train loss (MSE, 训练集)")
    ax.plot(hist.test_epoch, hist.test_loss, color="#d62728", lw=1.6,
            marker="o", ms=3.4, label="test loss (MSE, 新数据)")

    floor = cfg.noise_std**2
    if floor > 0:
        ax.axhline(floor, color="gray", ls="--", lw=1.0,
                   label=f"noise floor σ²={floor:.3g}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(_title(cfg), fontsize=10)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)

    # PDF alongside the PNG for papers.
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def plot_diagnostics(hist: TrainingHistory, cfg: ExperimentConfig, path: Path) -> Path:
    """Measured vs analytic test MSE, and the distance to the teacher weights."""
    configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax1.plot(hist.test_epoch, hist.test_loss, color="#d62728", lw=1.6,
             marker="o", ms=3.0, label="measured test MSE")
    ax1.plot(hist.test_epoch, hist.test_analytic, color="#2ca02c", lw=1.2,
             ls="-.", label=r"analytic $\|w-\bar w\|^2+\sigma^2$")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE")
    ax1.set_title("test loss: measured vs analytic")
    ax1.legend(loc="best")

    ax2.plot(hist.test_epoch, hist.weight_distance, color="#9467bd", lw=1.6,
             marker="o", ms=3.0)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(r"$\|w - \bar w\|_2$")
    ax2.set_title("distance to teacher weights")

    fig.suptitle(_title(cfg), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_interactive_loss(hist: TrainingHistory, cfg: ExperimentConfig, path: Path) -> Path:
    """Same curves as :func:`plot_loss_curves`, but as an interactive HTML figure.

    Adds the analytic risk as a third toggleable curve and the recorded weight
    distance, which the PNG deliberately leaves out to stay readable.
    """
    from plot_interactive import Series, hlines_to_series, write_interactive_curves

    series = [
        Series(name="train loss", x=np.asarray(hist.train_epoch, dtype=float),
               y=np.asarray(hist.train_loss, dtype=float), color="#1f77b4"),
        Series(name="test loss（新数据）", x=np.asarray(hist.test_epoch, dtype=float),
               y=np.asarray(hist.test_loss, dtype=float), color="#d62728",
               width=1.8, markers=True),
        Series(name="解析 test loss", x=np.asarray(hist.test_epoch, dtype=float),
               y=np.asarray(hist.test_analytic, dtype=float), color="#2ca02c",
               width=1.6),
        Series(name="‖w−w̄‖", x=np.asarray(hist.test_epoch, dtype=float),
               y=np.asarray(hist.weight_distance, dtype=float), color="#9467bd",
               visible=False),
    ]
    if cfg.noise_std > 0:
        series.insert(1, hlines_to_series("noise floor σ²", cfg.noise_std**2,
                                          min(hist.train_epoch),
                                          max(hist.train_epoch),
                                          color="#999"))

    return write_interactive_curves(
        path,
        title=f"N={cfg.N}, P={cfg.P}, epochs={cfg.epoch}",
        subtitle=(f"test_every={cfg.test_every}　lr={cfg.lr:g}　σ={cfg.noise_std:g}　"
                  f"seed={cfg.seed}　P/N={cfg.P / cfg.N:.2f}"),
        xlabel="epoch", ylabel="MSE",
        series=series,
        source="数据来源：<code>loss.csv</code>（同一份数据，PNG 用 matplotlib 画）",
        magnify_default=True,
    )


def _title(cfg: ExperimentConfig) -> str:
    return (f"Teacher/student linear net   N={cfg.N}  P={cfg.P}  "
            f"epochs={cfg.epoch}  test_every={cfg.test_every}\n"
            f"lr={cfg.lr:g}   sigma={cfg.noise_std:g}   seed={cfg.seed}")
