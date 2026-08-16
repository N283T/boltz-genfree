from pathlib import Path

import numpy as np
import pytest
import torch

from boltz.data import const
from boltz.data.types import (
    AffinityInfo,
    ChainInfo,
    Record,
    StructureInfo,
)
from boltz.data.write.writer import BoltzAffinityWriter


def _make_record(record_id: str, affinity: AffinityInfo | None = None) -> Record:
    return Record(
        id=record_id,
        structure=StructureInfo(),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id=0,
                msa_id=-1,
                num_residues=1,
            )
        ],
        interfaces=[],
        affinity=affinity,
    )


def test_affinity_writer_uses_per_copy_output_name(tmp_path: Path):
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
    )

    batch = {
        "record": [
            _make_record(
                "record_a",
                affinity=AffinityInfo(
                    chain_id=0,
                    mw=123.4,
                    chain_name="L2",
                    output_id="record_a_L2",
                ),
            )
        ],
    }
    prediction = {
        "exception": False,
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    import json

    path = tmp_path / "out" / "record_a" / "affinity_record_a_L2.json"
    with path.open() as f:
        affinity = json.load(f)

    assert affinity["binder_chain"] == "L2"
    assert affinity["affinity_pred_value"] == pytest.approx(1.25)


def test_affinity_writer_writes_mode_outputs_and_distogram_artifact(
    tmp_path: Path,
):
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        experiment_metadata={"seed": 7},
    )
    record = _make_record(
        "record_a",
        affinity=AffinityInfo(
            chain_id=0,
            mw=123.4,
            chain_name="L2",
            output_id="record_a_L2",
        ),
    )
    mode_prediction = {
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
        "affinity_distogram_bins": torch.tensor([[[0, 63], [63, 0]]]),
        "affinity_distogram_intervened_pair_mask": torch.tensor(
            [[[False, True], [True, False]]]
        ),
    }
    prediction = {
        "exception": False,
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
        "affinity_by_distogram_mode": {"constant_far": mode_prediction},
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch={"record": [record]},
        batch_idx=0,
        dataloader_idx=0,
    )

    import json

    stem = "affinity_record_a_L2__distogram_constant_far"
    with (tmp_path / "out" / "record_a" / f"{stem}.json").open() as f:
        summary = json.load(f)
    artifact = np.load(tmp_path / "out" / "record_a" / f"{stem}.npz")

    assert summary["distogram_intervention"] == "constant_far"
    assert summary["conditioner_added"] is True
    assert summary["effective_distogram"]["intervened_pair_count"] == 2
    assert summary["experiment"] == {"seed": 7}
    assert artifact["bins"].dtype == np.uint8
    assert artifact["intervened_pair_mask"].sum() == 2


def test_affinity_writer_writes_pooled_representations(tmp_path: Path):
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        write_affinity_representations=True,
    )
    record = _make_record(
        "record_a",
        affinity=AffinityInfo(
            chain_id=0,
            mw=123.4,
            chain_name="L2",
            output_id="record_a_L2",
        ),
    )
    mode_prediction = {
        "affinity_pred_value": torch.tensor([1.25]),
        "affinity_probability_binary": torch.tensor([0.75]),
        "affinity_distogram_bins": torch.zeros(1, 2, 2, dtype=torch.long),
        "affinity_distogram_intervened_pair_mask": torch.zeros(
            1, 2, 2, dtype=torch.bool
        ),
        "affinity_representation_member1_pooled_conditioned": torch.tensor(
            [[1.0, 2.0]]
        ),
        "affinity_representation_member2_pooled_conditioned": torch.tensor(
            [[3.0, 4.0]]
        ),
    }
    prediction = {
        "exception": False,
        "affinity_pred_value": torch.tensor([1.25]),
        "affinity_probability_binary": torch.tensor([0.75]),
        "affinity_by_distogram_mode": {"control": mode_prediction},
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch={"record": [record]},
        batch_idx=0,
        dataloader_idx=0,
    )

    path = (
        tmp_path
        / "out"
        / "record_a"
        / "affinity_record_a_L2__distogram_control__pooled_representations.npz"
    )
    with np.load(path) as data:
        np.testing.assert_array_equal(
            data["member1_pooled_conditioned"], np.asarray([1.0, 2.0])
        )
        assert data["representation_stage"] == "affinity_cross_pair_mean"


def test_affinity_writer_minimal_screening_output_deduplicates_run_metadata(
    tmp_path: Path,
):
    input_a = tmp_path / "inputs" / "record_a.yaml"
    input_b = tmp_path / "inputs" / "record_b.yaml"
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        experiment_metadata={
            "seed": 7,
            "use_kernels": True,
            "input_sources": [str(input_a), str(input_b)],
        },
        output_tag="minimal",
        screening_minimal_output=True,
    )
    record = _make_record(
        "record_a",
        affinity=AffinityInfo(
            chain_id=0,
            mw=123.4,
            chain_name="L2",
            output_id="record_a_L2",
        ),
    )
    mode_prediction = {
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
        "affinity_distogram_bins": torch.zeros((1, 2, 2), dtype=torch.long),
        "affinity_distogram_intervened_pair_mask": torch.ones(
            (1, 2, 2), dtype=torch.bool
        ),
    }
    prediction = {
        "exception": False,
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
        "affinity_by_distogram_mode": {"zero_distogram_features": mode_prediction},
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch={"record": [record]},
        batch_idx=0,
        dataloader_idx=0,
    )

    import json

    out_dir = tmp_path / "out"
    record_dir = out_dir / "record_a"
    mode_path = record_dir / (
        "affinity_record_a_L2__distogram_zero_distogram_features__run_minimal.json"
    )
    summary = json.loads(mode_path.read_text())
    run_metadata = json.loads(
        (out_dir / "affinity_run_metadata__run_minimal.json").read_text()
    )

    assert sorted(path.name for path in record_dir.iterdir()) == [mode_path.name]
    assert summary["experiment"]["input_source"] == str(input_a)
    assert "input_sources" not in summary["experiment"]
    assert "effective_distogram" not in summary
    assert summary["experiment"]["output_layout"] == ("affinity_screening_minimal_v1")
    assert summary["experiment"]["run_metadata_file"] == (
        "../affinity_run_metadata__run_minimal.json"
    )
    assert run_metadata["experiment"]["input_sources"] == [
        str(input_a),
        str(input_b),
    ]


def test_affinity_writer_minimal_output_supports_batches(tmp_path: Path):
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        output_tag="batch",
        screening_minimal_output=True,
    )
    records = [
        _make_record(
            f"record_{suffix}",
            affinity=AffinityInfo(
                chain_id=0,
                mw=123.4,
                chain_name="L2",
                output_id=f"record_{suffix}_L2",
            ),
        )
        for suffix in ("a", "b")
    ]
    mode_prediction = {
        "affinity_pred_value": torch.tensor([[1.25], [2.5]]),
        "affinity_probability_binary": torch.tensor([[0.75], [0.5]]),
        "affinity_distogram_bins": torch.zeros((2, 2, 2), dtype=torch.long),
        "affinity_distogram_intervened_pair_mask": torch.ones(
            (2, 2, 2), dtype=torch.bool
        ),
    }
    prediction = {
        "exception": False,
        **{
            key: mode_prediction[key]
            for key in ("affinity_pred_value", "affinity_probability_binary")
        },
        "affinity_by_distogram_mode": {"zero_distogram_features": mode_prediction},
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=[],
        batch={"record": records},
        batch_idx=0,
        dataloader_idx=0,
    )

    import json

    summaries = []
    for record in records:
        path = (
            tmp_path
            / "out"
            / record.id
            / (
                f"affinity_{record.affinity.output_id}"
                "__distogram_zero_distogram_features__run_batch.json"
            )
        )
        summaries.append(json.loads(path.read_text()))

    assert [row["affinity_pred_value"] for row in summaries] == [1.25, 2.5]
    assert [row["affinity_probability_binary"] for row in summaries] == [
        0.75,
        0.5,
    ]


def test_affinity_writer_reports_completed_and_failed_batches(tmp_path: Path) -> None:
    updates: list[tuple[list[str], bool]] = []
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        progress_callback=lambda record_ids, failed: updates.append(
            (record_ids, failed)
        ),
    )
    record = _make_record(
        "record_a",
        affinity=AffinityInfo(
            chain_id=0,
            mw=123.4,
            chain_name="L2",
            output_id="record_a_L2",
        ),
    )
    successful = {
        "exception": False,
        "affinity_pred_value": torch.tensor(1.25),
        "affinity_probability_binary": torch.tensor(0.75),
    }

    writer.write_on_batch_end(
        None,
        None,
        successful,
        [],
        {"record": [record]},
        0,
        0,
    )
    writer.write_on_batch_end(
        None,
        None,
        {"exception": True},
        [],
        {"record": [record]},
        1,
        0,
    )

    assert updates == [(["record_a"], False), (["record_a"], True)]
