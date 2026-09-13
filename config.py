"""Global configuration for the teacher/student linear-network experiment.

The whole experiment has exactly four user-tunable knobs:

    N      -- input dimension (also the number of parameters of the student)
    P      -- number of training samples (and number of fresh test samples)
    epoch  -- number of full-batch gradient-descent steps on the training set
    test_every -- evaluate the test loss once every this many epochs

Everything else (learning rate, noise level, seeds, output paths) lives here as
a default and can be overridden from the command line.  Keeping the knobs in a
single frozen dataclass means every subroutine receives an immutable, explicit
description of the experiment instead of reaching for globals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable description of one teacher/student experiment run."""

    # ---- the four tunable parameters -------------------------------------
    N: int = 32
    P: int = 64
    epoch: int = 2000
    test_every: int = 10

    # ---- teacher ---------------------------------------------------------
    # w_bar ~ N(0, 1/N) so that <w_bar, x> stays O(1) as N grows.
    teacher_init: str = "normal"
    teacher_scale: float = 1.0  # variance of each w_bar component is teacher_scale / N

    # ---- data / noise ----------------------------------------------------
    x_dist: str = "normal"       # N(0, 1) inputs, E[xx^T] = I
    noise_std: float = 0.01      # epsilon ~ N(0, noise_std ** 2)
    n_test: int | None = None    # test-set size; None -> use P

    # ---- student / optimiser --------------------------------------------
    optimizer: str = "gd"        # plain full-batch gradient descent
    lr: float = 0.1
    student_init: str = "zeros"  # "zeros" (default) or "small_random"
    student_init_std: float = 0.01
    weight_decay: float = 0.0    # 0.0 == plain GD, no regularisation

    # ---- bookkeeping -----------------------------------------------------
    seed: int = 0
    run_name: str | None = None  # None -> derived from (N, P, epoch, ...)
    out_root: str = "experiments"
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.P <= 0:
            raise ValueError(f"P must be positive, got {self.P}")
        if self.epoch <= 0:
            raise ValueError(f"epoch must be positive, got {self.epoch}")
        if self.test_every <= 0:
            raise ValueError(f"test_every must be positive, got {self.test_every}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.optimizer != "gd":
            raise NotImplementedError(
                f"only plain gradient descent is implemented, got optimizer={self.optimizer!r}"
            )
        if self.n_test is not None and self.n_test <= 0:
            raise ValueError(f"n_test must be positive, got {self.n_test}")

    # ------------------------------------------------------------------
    @property
    def test_size(self) -> int:
        """Number of samples drawn into the *fresh* test set at every test."""
        return self.P if self.n_test is None else self.n_test

    @property
    def tag(self) -> str:
        """Folder/file tag describing the run, e.g. ``N32_P64_ep2000_te10``."""
        base = f"N{self.N}_P{self.P}_ep{self.epoch}_te{self.test_every}"
        extras = []
        if self.lr != 0.1:
            extras.append(f"lr{self.lr:g}")
        if self.noise_std != 0.01:
            extras.append(f"s{self.noise_std:g}")
        if self.weight_decay:
            extras.append(f"wd{self.weight_decay:g}")
        if self.student_init != "zeros":
            extras.append(self.student_init)
        if self.seed != 0:
            extras.append(f"seed{self.seed}")
        return base + ("_" + "_".join(extras) if extras else "")

    @property
    def run_dir(self) -> Path:
        name = self.run_name if self.run_name else self.tag
        return Path(self.out_root) / name

    def as_dict(self) -> dict:
        d = asdict(self)
        d["test_size"] = self.test_size
        d["tag"] = self.tag
        return d


# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse the command line into an :class:`ExperimentConfig`."""
    p = argparse.ArgumentParser(
        description="Teacher/student single-layer linear network: full-batch GD on MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-N", "--N", type=int, default=32, help="input dimension")
    p.add_argument("-P", "--P", type=int, default=64, help="number of training samples")
    p.add_argument("--epoch", type=int, default=2000, help="number of training epochs")
    p.add_argument("--test-every", dest="test_every", type=int, default=10,
                   help="evaluate the test loss every this many epochs")
    p.add_argument("--noise-std", dest="noise_std", type=float, default=0.01,
                   help="standard deviation of the teacher's Gaussian label noise")
    p.add_argument("--lr", type=float, default=0.1, help="learning rate (plain GD)")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0,
                   help="L2 coefficient; 0.0 means plain gradient descent")
    p.add_argument("--n-test", dest="n_test", type=int, default=None,
                   help="test-set size, defaults to P")
    p.add_argument("--student-init", dest="student_init", type=str, default="zeros",
                   choices=["zeros", "small_random"], help="student initialisation")
    p.add_argument("--student-init-std", dest="student_init_std", type=float, default=0.01,
                   help="std used when --student-init small_random")
    p.add_argument("--teacher-scale", dest="teacher_scale", type=float, default=1.0,
                   help="variance of w_bar components is teacher_scale / N")
    p.add_argument("--seed", type=int, default=0, help="master random seed")
    p.add_argument("--run-name", dest="run_name", type=str, default=None,
                   help="explicit output folder name (default: derived from N,P,epoch,...)")
    p.add_argument("--out-root", dest="out_root", type=str, default="experiments",
                   help="root folder for experiment outputs")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")

    a = p.parse_args(argv)
    return ExperimentConfig(
        N=a.N,
        P=a.P,
        epoch=a.epoch,
        test_every=a.test_every,
        teacher_scale=a.teacher_scale,
        noise_std=a.noise_std,
        n_test=a.n_test,
        lr=a.lr,
        weight_decay=a.weight_decay,
        student_init=a.student_init,
        student_init_std=a.student_init_std,
        seed=a.seed,
        run_name=a.run_name,
        out_root=a.out_root,
        verbose=not a.quiet,
    )
