"""Student network:  y = w(t)^T x,  trained by full-batch gradient descent on MSE.

Because the model is linear and the loss is the mean squared error, the
gradient has a closed form and no autodiff framework is needed.

For a batch (X, y) with X of shape (P, N):

    L(w)  = (1/P) ||X w - y||^2
    dL/dw = (2/P) X^T (X w - y)

Plain gradient descent (optionally with L2 weight decay) then reads

    w <- w - lr * (dL/dw + 2 * weight_decay * w)

The extra factor of 2 on the decay keeps the regularised objective equal to
``MSE + weight_decay * ||w||^2``.
"""

from __future__ import annotations

import numpy as np

from config import ExperimentConfig


class Student:
    """Linear student with a single closed-form gradient-descent step."""

    def __init__(self, cfg: ExperimentConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        if cfg.student_init == "zeros":
            self.w = np.zeros(cfg.N, dtype=np.float64)
        elif cfg.student_init == "small_random":
            self.w = rng.normal(0.0, cfg.student_init_std, size=cfg.N)
        else:  # pragma: no cover - config validation already rejects this
            raise ValueError(f"unknown student_init {cfg.student_init!r}")
        self._x = None  # cached training inputs, set by fit()
        self._y = None

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Linear forward pass ``x @ w``."""
        return x @ self.w

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of the MSE on the batch ``(x, y)`` w.r.t. ``w``."""
        residual = x @ self.w - y
        return (2.0 / x.shape[0]) * (x.T @ residual)

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        """One full-batch gradient-descent update on ``(x, y)``."""
        g = self.gradient(x, y)
        if self.cfg.weight_decay:
            g = g + 2.0 * self.cfg.weight_decay * self.w
        self.w -= self.cfg.lr * g

    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int,
            on_epoch=None) -> None:
        """Run ``epochs`` full-batch gradient steps over the same fixed batch.

        Parameters
        ----------
        x, y : the training set (drawn once by the teacher).
        epochs : number of passes; each pass uses the *whole* training set.
        on_epoch : optional callback ``on_epoch(epoch_index)`` called after every
            update, with ``epoch_index`` running from 1 to ``epochs``.
        """
        for epoch in range(1, epochs + 1):
            self.step(x, y)
            if on_epoch is not None:
                on_epoch(epoch)

    # ------------------------------------------------------------------
    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error of the current weights on ``(x, y)``."""
        residual = x @ self.w - y
        return float(np.mean(residual**2))

    def analytic_mse(self, w_target: np.ndarray, noise_variance: float = 0.0) -> float:
        """Expected MSE on *infinite* fresh data: ``||w - w_target||^2 + noise_variance``.

        Also equals the expected test loss, since the student is evaluated on
        data drawn from the teacher's distribution.  ``noise_variance`` is the
        label-noise *variance* ``sigma^2``, not the standard deviation.
        """
        return float(np.sum((self.w - w_target) ** 2) + noise_variance)


# ----------------------------------------------------------------------
def ridge_solution(x: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> np.ndarray:
    """Closed-form minimiser of MSE (+ optional L2), used as a sanity check."""
    n = x.shape[1]
    if weight_decay > 0:
        a = x.T @ x + weight_decay * x.shape[0] * np.eye(n)
    else:
        a = x.T @ x
        a = a + 1e-12 * np.eye(n)  # tiny jitter so singular systems still solve
    return np.linalg.solve(a, x.T @ y)
