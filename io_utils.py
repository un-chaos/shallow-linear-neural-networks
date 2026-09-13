"""Result persistence: CSV export and figure saving helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from train import TrainingHistory

CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "test_loss",
    "test_loss_analytic",
    "weight_distance",
]


def save_history_csv(hist: TrainingHistory, path: Path) -> Path:
    """Write one tidy row per epoch.

    ``train_loss`` is present for every epoch; ``test_loss`` is filled in only on
    the epochs where a fresh test set was drawn (blank elsewhere).  The analytic
    column is the exact expected test MSE ``||w - w_bar||^2 + sigma^2`` on
    infinite fresh data -- a scale reference, not a measurement.

    An epoch-0 row holds the untrained baseline: ``train_loss`` empty,
    ``test_loss`` measured.
    """
    train = dict(zip(hist.train_epoch, hist.train_loss))
    test = dict(zip(hist.test_epoch, hist.test_loss))
    analytic = dict(zip(hist.test_epoch, hist.test_analytic))
    distance = dict(zip(hist.test_epoch, hist.weight_distance))

    epochs = sorted(set(train) | set(test) | {0})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for epoch in epochs:
            writer.writerow([
                epoch,
                _fmt(train.get(epoch)),
                _fmt(test.get(epoch)),
                _fmt(analytic.get(epoch)),
                _fmt(distance.get(epoch)),
            ])
    return path


def save_metadata(cfg: ExperimentConfig, hist: TrainingHistory, path: Path,
                  extra: dict | None = None) -> Path:
    """Persist the full configuration plus summary statistics as JSON."""
    payload = {
        "config": cfg.as_dict(),
        "summary": {
            "final_train_loss": hist.final_train_loss,
            "final_test_loss": hist.final_test_loss,
            "best_test_loss": hist.best_test_loss,
            "final_test_loss_analytic": hist.test_analytic[-1],
            "final_weight_distance": hist.weight_distance[-1],
            "n_train_points_recorded": len(hist.train_epoch),
            "n_test_points_recorded": len(hist.test_epoch),
        },
    }
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return repr(float(value)) if not np.isfinite(value) else f"{float(value):.12g}"
