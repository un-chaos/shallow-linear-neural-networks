"""Teacher network and data generation.

Teacher:      y = w_bar^T x + epsilon,   w_bar ~ N(0, teacher_scale / N),  epsilon ~ N(0, noise_std^2)
Inputs:       x ~ N(0, I_N)

The teacher is created once per run and then only *samples* data.  The training
set is drawn a single time (the student repeatedly re-fits that same set), while
a brand-new test set is drawn by the teacher at every evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ExperimentConfig


@dataclass(frozen=True)
class Dataset:
    """A supervised dataset holding inputs and (noisy) teacher labels."""

    x: np.ndarray  # shape (n, N)
    y: np.ndarray  # shape (n,)

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return self.x.shape[1]


class Teacher:
    """Fixed-weight linear teacher with additive Gaussian label noise."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        std = np.sqrt(cfg.teacher_scale / cfg.N)
        self.w_bar = rng.normal(0.0, std, size=cfg.N)
        # A single fresh stream for every subsequent sample draw, so that the
        # training set and all of the rolling test sets are independent.
        self._rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    # ------------------------------------------------------------------
    def sample(self, n: int) -> Dataset:
        """Draw ``n`` fresh i.i.d. samples ``(x, w_bar^T x + epsilon)``."""
        x = self._rng.normal(0.0, 1.0, size=(n, self.cfg.N))
        clean = x @ self.w_bar
        if self.cfg.noise_std > 0:
            clean = clean + self._rng.normal(0.0, self.cfg.noise_std, size=n)
        return Dataset(x=x, y=clean)

    # ------------------------------------------------------------------
    def noise_floor(self) -> float:
        """Irreducible MSE of the Bayes-optimal predictor: E[epsilon^2]."""
        return float(self.cfg.noise_std**2)
