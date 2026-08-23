"""Regression and communication metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RegressionMetrics:
    mse_standardized: float
    rmse_standardized: float
    mae_standardized: float
    rmse_physical: float
    mae_physical: float
    samples: int


def regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_scale: float,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    error = predictions.float() - targets.float()
    if valid_mask is not None:
        valid = valid_mask.to(dtype=torch.bool, device=error.device)
        if valid.shape != error.shape:
            raise ValueError("valid_mask must align with predictions")
        if not bool(valid.any()):
            raise ValueError("valid_mask contains no valid targets")
        error = error[valid]
    mse = float(error.square().mean())
    rmse = float(np.sqrt(mse))
    mae = float(error.abs().mean())
    return asdict(
        RegressionMetrics(
            mse_standardized=mse,
            rmse_standardized=rmse,
            mae_standardized=mae,
            rmse_physical=rmse * float(target_scale),
            mae_physical=mae * float(target_scale),
            samples=int(error.numel()),
        )
    )
