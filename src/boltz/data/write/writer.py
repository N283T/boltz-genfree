import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Record, get_affinity_output_id


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        experiment_metadata: Optional[dict[str, object]] = None,
        write_embeddings: bool = False,
        write_affinity_representations: bool = False,
        output_tag: Optional[str] = None,
        screening_minimal_output: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_metadata = experiment_metadata or {}
        self.write_embeddings = write_embeddings
        self.write_affinity_representations = write_affinity_representations
        self.output_tag = output_tag
        self.screening_minimal_output = screening_minimal_output
        self.run_metadata_path: Optional[Path] = None
        if self.screening_minimal_output:
            tag_suffix = f"__run_{self.output_tag}" if self.output_tag else ""
            self.run_metadata_path = (
                self.output_dir / f"affinity_run_metadata{tag_suffix}.json"
            )
            payload = {
                "schema_version": 1,
                "output_layout": "affinity_screening_minimal_v1",
                "experiment": self.experiment_metadata,
            }
            with self.run_metadata_path.open("w") as handle:
                handle.write(json.dumps(payload, indent=4))

    def _record_experiment_metadata(self, record: Record) -> dict[str, object]:
        """Return compact per-record metadata with shared run metadata reference."""
        if not self.screening_minimal_output:
            return self.experiment_metadata
        metadata: dict[str, object] = {
            "output_layout": "affinity_screening_minimal_v1",
        }
        input_sources = self.experiment_metadata.get("input_sources", [])
        if isinstance(input_sources, list):
            matching_sources = [
                source
                for source in input_sources
                if isinstance(source, str) and Path(source).stem == record.id
            ]
            if len(matching_sources) == 1:
                metadata["input_source"] = matching_sources[0]
        if self.run_metadata_path is not None:
            metadata["run_metadata_file"] = f"../{self.run_metadata_path.name}"
        return metadata

    @staticmethod
    def _summary(
        prediction: dict[str, Tensor],
        record: Record,
        record_idx: int = 0,
    ) -> dict[str, object]:
        """Build the legacy scalar affinity summary for one prediction."""
        def scalar(key: str) -> float:
            return prediction[key].reshape(-1)[record_idx].item()

        summary: dict[str, object] = {
            "affinity_pred_value": scalar("affinity_pred_value"),
            "affinity_probability_binary": scalar(
                "affinity_probability_binary"
            ),
        }
        if record.affinity and record.affinity.chain_name:
            summary["binder_chain"] = record.affinity.chain_name
        if "affinity_pred_value1" in prediction:
            summary.update(
                {
                    "affinity_pred_value1": scalar("affinity_pred_value1"),
                    "affinity_probability_binary1": scalar(
                        "affinity_probability_binary1"
                    ),
                    "affinity_pred_value2": scalar("affinity_pred_value2"),
                    "affinity_probability_binary2": scalar(
                        "affinity_probability_binary2"
                    ),
                }
            )
        return summary

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        for record_idx, record in enumerate(batch["record"]):
            struct_dir = self.output_dir / record.id
            struct_dir.mkdir(exist_ok=True)
            output_id = get_affinity_output_id(record)

            affinity_summary = self._summary(prediction, record, record_idx)
            legacy_suffix = f"__run_{self.output_tag}" if self.output_tag else ""
            if not self.screening_minimal_output:
                # Preserve the original output for compatibility. It represents
                # the first requested distogram mode.
                path = struct_dir / f"affinity_{output_id}{legacy_suffix}.json"
                with path.open("w") as f:
                    f.write(json.dumps(affinity_summary, indent=4))

            for mode, mode_prediction in prediction.get(
                "affinity_by_distogram_mode",
                {},
            ).items():
                mode_summary = self._summary(mode_prediction, record, record_idx)
                mode_summary.update(
                    {
                        "distogram_intervention": mode,
                        "conditioner_added": mode != "zero_conditioning",
                        "experiment": self._record_experiment_metadata(record),
                    }
                )
                bins = None
                pair_mask = None
                if not self.screening_minimal_output:
                    bins = mode_prediction["affinity_distogram_bins"][
                        record_idx : record_idx + 1
                    ].detach().cpu()
                    pair_mask = mode_prediction[
                        "affinity_distogram_intervened_pair_mask"
                    ][record_idx : record_idx + 1].detach().cpu()
                    num_bins = max(int(bins.max().item()) + 1, 64)
                    histogram = torch.bincount(
                        bins.reshape(-1),
                        minlength=num_bins,
                    ).tolist()
                    mode_summary["effective_distogram"] = {
                        "shape": list(bins.shape),
                        "min_bin": int(bins.min().item()),
                        "max_bin": int(bins.max().item()),
                        "intervened_pair_count": int(pair_mask.sum().item()),
                        "bin_histogram": histogram,
                    }
                tag_suffix = f"__run_{self.output_tag}" if self.output_tag else ""
                mode_stem = f"affinity_{output_id}__distogram_{mode}{tag_suffix}"
                with (struct_dir / f"{mode_stem}.json").open("w") as f:
                    f.write(json.dumps(mode_summary, indent=4))
                if not self.screening_minimal_output:
                    assert bins is not None
                    assert pair_mask is not None
                    np.savez_compressed(
                        struct_dir / f"{mode_stem}.npz",
                        bins=bins.numpy().astype(np.uint8),
                        intervened_pair_mask=pair_mask.numpy().astype(bool),
                    )
                if self.write_affinity_representations:
                    representations = {
                        key.removeprefix("affinity_representation_"): value[
                            record_idx
                        ]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        for key, value in mode_prediction.items()
                        if key.startswith("affinity_representation_")
                    }
                    np.savez_compressed(
                        struct_dir / f"{mode_stem}__pooled_representations.npz",
                        **representations,
                        representation_stage=np.asarray(
                            "affinity_cross_pair_mean"
                        ),
                    )

        if self.write_embeddings and "s" in prediction and "z" in prediction:
            token_mask = batch["token_pad_mask"][0].detach().cpu().bool()
            token_idx = batch["resume_token_idx"][0].detach().cpu()[token_mask]
            s = prediction["s"].float().cpu()
            z = prediction["z"].float().cpu()
            if s.ndim == 3:
                s = s[0]
            if z.ndim == 4:
                z = z[0]
            s = s[token_mask]
            z = z[token_mask][:, token_mask]
            np.savez_compressed(
                struct_dir / f"affinity_trunk_embeddings_{record.id}.npz",
                s=s.numpy(),
                z=z.numpy(),
                token_idx=token_idx.numpy(),
                embedding_stage=np.asarray("affinity_trunk_cropped"),
            )
            np.savez_compressed(
                struct_dir / f"affinity_crop_{record.id}.npz",
                token_idx=token_idx.numpy(),
                crop_source=np.asarray(
                    self.experiment_metadata.get(
                        "affinity_crop_mode",
                        (
                            "explicit_token_indices"
                            if self.experiment_metadata.get(
                                "affinity_crop_indices_dir"
                            )
                            else "prepared_geometry"
                        ),
                    )
                ),
                input_source=np.asarray(
                    self.experiment_metadata.get(
                        "affinity_input_source",
                        "pre_affinity_generated_complex",
                    )
                ),
                generated_coordinates_read=np.asarray(
                    not self.experiment_metadata.get(
                        "affinity_structure_free",
                        False,
                    )
                ),
                sequence_window_start=np.asarray(
                    self.experiment_metadata.get(
                        "affinity_sequence_window_start",
                        -1,
                    )
                ),
                sequence_window_size=np.asarray(
                    self.experiment_metadata.get(
                        "affinity_sequence_window_size",
                        -1,
                    )
                ),
                structure_free_crop_label=np.asarray(
                    self.experiment_metadata.get(
                        "affinity_structure_free_crop_label",
                        "",
                    )
                ),
            )
            context_keys = (
                "s_inputs",
                "z",
                "coords",
                "token_pad_mask",
                "mol_type",
                "affinity_token_mask",
                "token_to_rep_atom",
                "affinity_mw",
            )
            np.savez_compressed(
                struct_dir / f"affinity_head_context_{record.id}.npz",
                **{
                    key: prediction[f"affinity_context_{key}"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for key in context_keys
                },
                context_stage=np.asarray("affinity_head_input"),
            )

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
