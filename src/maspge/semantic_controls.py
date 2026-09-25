"""Parameter-matched controls for functional-role semantics.

The controls preserve every observed scalar and the complete MASPGE-RS
architecture.  They only replace the physical sensor-to-role contract with a
deterministic deranged assignment whose role-bin sizes match the real one.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def role_slot_derangement(
    role_feature_counts: tuple[int, ...],
    *,
    seed: int,
    maximum_attempts: int = 10_000,
) -> torch.Tensor:
    """Return source indices that move every valid slot to another role bin."""

    counts = tuple(int(value) for value in role_feature_counts)
    if any(value < 0 for value in counts):
        raise ValueError("role feature counts must be non-negative")
    active = [index for index, count in enumerate(counts) if count > 0]
    if len(active) < 2:
        raise ValueError("role derangement requires at least two active roles")
    role_ids = torch.cat(
        [torch.full((counts[index],), index, dtype=torch.long) for index in active]
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(maximum_attempts)):
        permutation = torch.randperm(role_ids.numel(), generator=generator)
        if torch.all(role_ids.index_select(0, permutation) != role_ids):
            return permutation
    raise RuntimeError(
        "could not construct a role-slot derangement; check role-bin balance"
    )


def permute_role_slots(
    x_role: torch.Tensor,
    role_feature_counts: tuple[int, ...],
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Repartition valid role slots without dropping or duplicating values."""

    if x_role.ndim not in {4, 5}:
        raise ValueError("x_role must be [H,U,R,F] or [B,H,U,R,F]")
    counts = tuple(int(value) for value in role_feature_counts)
    if x_role.shape[-2] != len(counts):
        raise ValueError("role count does not match x_role")
    if any(count > x_role.shape[-1] for count in counts):
        raise ValueError("role feature count exceeds the fixed slot dimension")
    active = [index for index, count in enumerate(counts) if count > 0]
    flattened = torch.cat(
        [x_role[..., index, : counts[index]] for index in active], dim=-1
    )
    permutation = permutation.to(device=x_role.device, dtype=torch.long)
    if permutation.ndim != 1 or permutation.numel() != flattened.shape[-1]:
        raise ValueError("permutation length does not match valid role slots")
    if not torch.equal(
        torch.sort(permutation).values,
        torch.arange(permutation.numel(), device=x_role.device),
    ):
        raise ValueError("permutation must contain every valid slot exactly once")
    shuffled = flattened.index_select(-1, permutation)
    output = torch.zeros_like(x_role)
    start = 0
    for role_index in active:
        count = counts[role_index]
        output[..., role_index, :count] = shuffled[..., start : start + count]
        start += count
    return output


class RoleContractControlDataset(Dataset):
    """Dataset view using a fixed, parameter-free pseudo-role contract."""

    def __init__(
        self,
        base: Dataset,
        *,
        role_feature_counts: tuple[int, ...],
        permutation_seed: int,
    ) -> None:
        self.base = base
        self.role_feature_counts = tuple(int(x) for x in role_feature_counts)
        self.permutation_seed = int(permutation_seed)
        self.permutation = role_slot_derangement(
            self.role_feature_counts, seed=self.permutation_seed
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        item["x_role"] = permute_role_slots(
            item["x_role"], self.role_feature_counts, self.permutation
        )
        return item
