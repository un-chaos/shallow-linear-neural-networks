"""Full-batch training loop.

Per epoch the student performs **one** gradient-descent step on the entire
fixed training set.  The training loss is recorded every epoch; the test loss is
recomputed only every ``test_every`` epochs -- and each time on a **brand-new**
dataset drawn by the teacher, never on the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import ExperimentConfig
from data import Dataset, Teacher
from model import Student


@dataclass
class TrainingHistory:
    """Everything a run produces, ready for CSV export and plotting."""

    train_epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_epoch: list[int] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_analytic: list[float] = field(default_factory=list)
    weight_distance: list[float] = field(default_factory=list)

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1]

    @property
    def final_test_loss(self) -> float:
        return self.test_loss[-1]

    @property
    def best_test_loss(self) -> float:
        return min(self.test_loss)


def evaluate(student: Student, teacher: Teacher, cfg: ExperimentConfig) -> tuple[float, float, float]:
    """Draw a fresh test set and return (mse, analytic_mse, ||w - w_bar||)."""
    test_set = teacher.sample(cfg.test_size)
    mse = student.mse(test_set.x, test_set.y)
    analytic = student.analytic_mse(teacher.w_bar, teacher.noise_floor())
    distance = float(np.linalg.norm(student.w - teacher.w_bar))
    return mse, analytic, distance


def run_training(student: Student, teacher: Teacher, train_set: Dataset,
                 cfg: ExperimentConfig) -> TrainingHistory:
    """Train ``student`` on ``train_set`` for ``cfg.epoch`` full-batch steps."""
    hist = TrainingHistory()
    test_epochs = _test_schedule(cfg)

    # Baseline evaluation before any update: the student's initial risk.
    _record_test(hist, student, teacher, cfg, epoch=0, baseline=True)

    for epoch in range(1, cfg.epoch + 1):
        student.step(train_set.x, train_set.y)

        hist.train_epoch.append(epoch)
        hist.train_loss.append(student.mse(train_set.x, train_set.y))

        if epoch in test_epochs:
            _record_test(hist, student, teacher, cfg, epoch=epoch)

        if cfg.verbose and _should_report(epoch, cfg):
            print(
                f"  epoch {epoch:>7d}/{cfg.epoch}  "
                f"train_loss={hist.train_loss[-1]:.6e}  "
                f"test_loss={hist.test_loss[-1]:.6e}"
            )

    return hist


# ----------------------------------------------------------------------
def _test_schedule(cfg: ExperimentConfig) -> set[int]:
    """Epoch indices at which the test loss is measured (always includes the last)."""
    epochs = set(range(cfg.test_every, cfg.epoch + 1, cfg.test_every))
    epochs.add(cfg.epoch)
    return epochs


def _record_test(hist: TrainingHistory, student: Student, teacher: Teacher,
                 cfg: ExperimentConfig, epoch: int, baseline: bool = False) -> None:
    mse, analytic, distance = evaluate(student, teacher, cfg)
    hist.test_epoch.append(epoch)
    hist.test_loss.append(mse)
    hist.test_analytic.append(analytic)
    hist.weight_distance.append(distance)
    if baseline and cfg.verbose:
        print(f"  epoch {epoch:>7d}        (init)  train_loss=--          "
              f"test_loss={mse:.6e}")


def _should_report(epoch: int, cfg: ExperimentConfig) -> bool:
    """Print roughly 20 progress lines over the whole run."""
    stride = max(1, cfg.epoch // 20)
    return epoch % stride == 0 or epoch == cfg.epoch
