from pathlib import Path

import torch

from boltz.data import const
from boltz.data.types import AffinityInfo, ChainInfo, Manifest, Record, StructureInfo
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.main import filter_inputs_affinity
from boltz.model.modules.affinity import pool_affinity_pairs


def _make_affinity_record(record_id: str = "target") -> Record:
    return Record(
        id=record_id,
        structure=StructureInfo(),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["PROTEIN"],
                cluster_id=0,
                msa_id=-1,
                num_residues=10,
                entity_id=0,
            ),
            ChainInfo(
                chain_id=1,
                chain_name="L",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id=1,
                msa_id=-1,
                num_residues=1,
                entity_id=1,
            ),
        ],
        interfaces=[],
        affinity=AffinityInfo(
            chain_id=1,
            mw=123.4,
            chain_name="L",
            output_id=record_id,
        ),
    )


def test_filter_inputs_affinity_skips_existing_output(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "target"
    pred_dir.mkdir(parents=True)
    (pred_dir / "affinity_target.json").write_text("{}")
    manifest = Manifest(records=[_make_affinity_record()])

    filtered = filter_inputs_affinity(
        manifest,
        tmp_path,
        override=False,
        require_pre_affinity=False,
    )

    assert filtered.records == []


def test_filter_inputs_affinity_keeps_existing_output_with_override(
    tmp_path: Path,
) -> None:
    pred_dir = tmp_path / "predictions" / "target"
    pred_dir.mkdir(parents=True)
    (pred_dir / "affinity_target.json").write_text("{}")
    manifest = Manifest(records=[_make_affinity_record()])

    filtered = filter_inputs_affinity(
        manifest,
        tmp_path,
        override=True,
        require_pre_affinity=False,
    )

    assert [record.id for record in filtered.records] == ["target"]


def test_filter_inputs_affinity_uses_output_tag(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "target"
    pred_dir.mkdir(parents=True)
    tagged = pred_dir / "affinity_target__distogram_normal_random__run_seed17.json"
    tagged.write_text("{}")
    manifest = Manifest(records=[_make_affinity_record()])

    filtered = filter_inputs_affinity(
        manifest,
        tmp_path,
        distogram_modes=("normal_random",),
        output_tag="seed17",
        require_pre_affinity=False,
    )

    assert filtered.records == []


def test_affinity_writer_accepts_embedding_output(tmp_path: Path) -> None:
    writer = BoltzAffinityWriter(
        data_dir=tmp_path,
        output_dir=tmp_path / "predictions",
        write_embeddings=True,
    )

    assert writer.write_embeddings is True


def test_affinity_writer_accepts_output_tag(tmp_path: Path) -> None:
    writer = BoltzAffinityWriter(
        data_dir=tmp_path,
        output_dir=tmp_path / "predictions",
        output_tag="seed17",
    )

    assert writer.output_tag == "seed17"


def test_pool_affinity_pairs_matches_cross_pair_head_pooling() -> None:
    z = torch.arange(18, dtype=torch.float32).reshape(1, 3, 3, 2)
    feats = {
        "token_pad_mask": torch.ones(1, 3),
        "mol_type": torch.tensor([[0, 0, 3]]),
        "affinity_token_mask": torch.tensor([[False, False, True]]),
    }

    pooled = pool_affinity_pairs(z, feats)
    expected = z[0, [0, 1, 2, 2], [2, 2, 0, 1]].mean(dim=0).unsqueeze(0)

    torch.testing.assert_close(pooled, expected)
