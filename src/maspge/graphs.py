"""Directed graph builders using adjacency[receiver, sender]."""

from __future__ import annotations

import numpy as np
import torch


def all_share_adjacency(unit_count: int) -> np.ndarray:
    adjacency = np.ones((unit_count, unit_count), dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def static_knn_adjacency(
    coordinates_xy: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_xy must have shape [unit, 2]")
    unit_count = len(coordinates)
    if not 0 < k < unit_count:
        raise ValueError("k must be between 1 and unit_count - 1")
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.square(delta).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    adjacency = np.zeros((unit_count, unit_count), dtype=np.float32)
    for receiver in range(unit_count):
        senders = np.argsort(distance[receiver], kind="stable")[:k]
        adjacency[receiver, senders] = 1.0
    return adjacency


def random_k_incoming_adjacency(
    unit_count: int,
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    if not 0 < k < unit_count:
        raise ValueError("k must be between 1 and unit_count - 1")
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((unit_count, unit_count), dtype=np.float32)
    for receiver in range(unit_count):
        candidates = np.delete(np.arange(unit_count), receiver)
        senders = rng.choice(candidates, size=k, replace=False)
        adjacency[receiver, senders] = 1.0
    return adjacency


def normalize_by_incoming_degree(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [batch, unit, unit]")
    values = adjacency.float()
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1.0)
