import torch

from maspge.semantic_controls import permute_role_slots, role_slot_derangement


def test_role_slot_derangement_changes_every_role_bin() -> None:
    counts = (3, 2, 2, 0)
    permutation = role_slot_derangement(counts, seed=20260828)
    role_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    assert torch.all(role_ids[permutation] != role_ids)
    assert sorted(permutation.tolist()) == list(range(sum(counts)))


def test_permute_role_slots_preserves_all_valid_values() -> None:
    counts = (3, 2, 2, 0)
    x_role = torch.zeros(4, 2, 4, 3)
    values = torch.arange(1, 4 * 2 * sum(counts) + 1, dtype=torch.float32)
    start = 0
    for role, count in enumerate(counts):
        if count:
            size = 4 * 2 * count
            x_role[..., role, :count] = values[start : start + size].reshape(4, 2, count)
            start += size
    permutation = role_slot_derangement(counts, seed=20260828)
    output = permute_role_slots(x_role, counts, permutation)
    before = torch.cat([x_role[..., role, :count].reshape(-1) for role, count in enumerate(counts) if count])
    after = torch.cat([output[..., role, :count].reshape(-1) for role, count in enumerate(counts) if count])
    assert torch.equal(torch.sort(before).values, torch.sort(after).values)
    assert not torch.equal(before, after)
    assert torch.count_nonzero(output[..., 3, :]) == 0
