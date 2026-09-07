import torch

from boltz.model.modules.trunkv2 import _record_seeded_msa_indices


def test_record_seeded_msa_indices_match_single_record_sampling() -> None:
    seeds = torch.tensor([17, 29], dtype=torch.int64)

    batched = _record_seeded_msa_indices(128, 32, seeds, recycling_step=2)
    separate = torch.stack(
        [
            _record_seeded_msa_indices(
                128,
                32,
                seed.view(1),
                recycling_step=2,
            )[0]
            for seed in seeds
        ]
    )

    assert torch.equal(batched, separate)


def test_record_seeded_msa_indices_match_legacy_global_generator() -> None:
    seed = 17
    torch.random.default_generator.manual_seed(seed)
    for _ in range(3):
        legacy = torch.randperm(128)

    actual = _record_seeded_msa_indices(
        128,
        32,
        torch.tensor([seed], dtype=torch.int64),
        recycling_step=2,
    )[0]

    assert torch.equal(actual, legacy[:32])
