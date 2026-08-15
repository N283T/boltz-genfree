import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    StructureV2,
    Tokenized,
)


def prepare_ligand_reference_conformers(
    molecules: dict,
    tokenized: Tokenized,
    mode: str,
) -> dict:
    """Prepare ligand-local conformers used by the atom input encoder."""
    if mode in {"etkdg", "all_zero"}:
        return molecules
    if mode != "rdkit_2d":
        msg = f"Unsupported ligand reference conformer mode: {mode}"
        raise ValueError(msg)

    ligand_names = set(
        tokenized.tokens[
            tokenized.tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        ]["res_name"].tolist()
    )
    prepared = dict(molecules)
    for name in ligand_names:
        mol = Chem.Mol(molecules[name])
        mol.RemoveAllConformers()
        rdDepictor.Compute2DCoords(mol)
        prepared[name] = mol
    return prepared


def apply_ligand_reference_position_intervention(
    features: dict[str, Tensor],
    mode: str,
) -> dict[str, Tensor]:
    """Apply an exact post-featurization intervention to ligand ref_pos."""
    if mode in {"etkdg", "rdkit_2d"}:
        return features
    if mode != "all_zero":
        msg = f"Unsupported ligand reference conformer mode: {mode}"
        raise ValueError(msg)

    atom_to_token = features["atom_to_token"].bool()
    affinity_token_mask = features["affinity_token_mask"].bool()
    ligand_atom_mask = torch.any(
        atom_to_token & affinity_token_mask.unsqueeze(0),
        dim=1,
    )
    ref_pos = features["ref_pos"].clone()
    ref_pos[ligand_atom_mask] = 0.0
    return {**features, "ref_pos": ref_pos}


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
    template_dir: Optional[Path] = None,
    extra_mols_dir: Optional[Path] = None,
    affinity: bool = False,
    affinity_structure_free: bool = False,
    load_msa: bool = True,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.
    constraints_dir : Optional[Path]
        The path to the constraints directory.
    template_dir : Optional[Path]
        The path to the template directory.
    extra_mols_dir : Optional[Path]
        The path to the extra molecules directory.
    affinity : bool
        Whether to load the affinity data.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    if affinity and not affinity_structure_free:
        structure = StructureV2.load(
            target_dir / record.id / f"pre_affinity_{record.id}.npz"
        )
    else:
        structure = StructureV2.load(target_dir / f"{record.id}.npz")

    msas = {}
    if load_msa:
        for chain in record.chains:
            msa_id = chain.msa_id
            # Load the MSA for this chain, if any
            if msa_id != -1:
                msa = MSA.load(msa_dir / f"{msa_id}.npz")
                msas[chain.chain_id] = msa

    # Load templates
    templates = None
    if record.templates and template_dir is not None:
        templates = {}
        for template_info in record.templates:
            template_id = template_info.name
            template_path = template_dir / f"{record.id}_{template_id}.npz"
            template = StructureV2.load(template_path)
            templates[template_id] = template

    # Load residue constraints
    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(
            constraints_dir / f"{record.id}.npz"
        )

    # Load extra molecules
    extra_mols = {}
    if extra_mols_dir is not None:
        extra_mol_path = extra_mols_dir / f"{record.id}.pkl"
        if extra_mol_path.exists():
            with extra_mol_path.open("rb") as f:
                extra_mols = pickle.load(f)  # noqa: S301

    return Input(
        structure,
        msas,
        record=record,
        residue_constraints=residue_constraints,
        templates=templates,
        extra_mols=extra_mols,
    )


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        affinity_crop_indices_dir: Optional[Path] = None,
        affinity_receptor_crop_indices_dir: Optional[Path] = None,
        affinity_structure_free: bool = False,
        affinity_structure_free_single_sequence: bool = False,
        affinity_sequence_window_start: int = 0,
        affinity_ligand_ref_pos_mode: str = "etkdg",
        affinity_require_full_input: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(self.mol_dir)
        self._mol_cache: dict = {}
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        self.affinity_crop_indices_dir = affinity_crop_indices_dir
        self.affinity_receptor_crop_indices_dir = affinity_receptor_crop_indices_dir
        self.affinity_structure_free = affinity_structure_free
        self.affinity_structure_free_single_sequence = (
            affinity_structure_free_single_sequence
        )
        self.affinity_sequence_window_start = affinity_sequence_window_start
        self.affinity_ligand_ref_pos_mode = affinity_ligand_ref_pos_mode
        self.affinity_require_full_input = affinity_require_full_input
        if self.affinity:
            self.cropper = AffinityCropper()

    def _load_affinity_crop_indices(self, record_id: str) -> Optional[np.ndarray]:
        if self.affinity_crop_indices_dir is None:
            return None
        candidates = (
            self.affinity_crop_indices_dir
            / record_id
            / f"affinity_crop_{record_id}.npz",
            self.affinity_crop_indices_dir / f"affinity_crop_{record_id}.npz",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            msg = f"Missing explicit affinity crop for {record_id}: {candidates[0]}"
            raise FileNotFoundError(msg)
        with np.load(path) as data:
            if "token_idx" not in data:
                msg = f"Affinity crop must contain token_idx: {path}"
                raise ValueError(msg)
            return data["token_idx"].copy()

    def _load_affinity_receptor_crop_indices(
        self,
        record_id: str,
    ) -> Optional[np.ndarray]:
        if self.affinity_receptor_crop_indices_dir is None:
            return None
        path = (
            self.affinity_receptor_crop_indices_dir / f"affinity_crop_{record_id}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing reference receptor crop for {record_id}: {path}"
            )
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("record_id") != record_id:
            raise ValueError(f"Reference receptor crop record ID mismatch: {path}")
        indices = document.get("receptor_token_idx")
        if not isinstance(indices, list) or not all(
            isinstance(index, int) and not isinstance(index, bool) for index in indices
        ):
            raise ValueError(
                f"Reference receptor crop must contain integer receptor_token_idx: {path}"
            )
        return np.asarray(indices, dtype=np.int64)

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        Raises
        ------
        Exception
            If any pipeline stage fails for this record. During inference
            failures must surface rather than silently returning a different
            record's results.

        """
        # Get record
        record = self.manifest.records[idx]

        # Finalize input data
        input_data = load_input(
            record=record,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
            affinity=self.affinity,
            affinity_structure_free=self.affinity_structure_free,
            load_msa=not self.affinity_structure_free_single_sequence,
        )

        # Tokenize structure
        tokenized = self.tokenizer.tokenize(input_data)

        if self.affinity:
            if self.affinity_structure_free:
                crop_indices = self._load_affinity_crop_indices(record.id)
                receptor_crop_indices = self._load_affinity_receptor_crop_indices(
                    record.id
                )
                if crop_indices is not None and receptor_crop_indices is not None:
                    raise ValueError(
                        "Cannot combine full-token and receptor-only affinity crops."
                    )
                if self.affinity_require_full_input:
                    if (
                        crop_indices is not None
                        or receptor_crop_indices is not None
                        or self.affinity_sequence_window_start != 0
                    ):
                        msg = "Full-screen inference does not accept crop overrides."
                        raise ValueError(msg)
                    tokenized = self.cropper.crop_full_screen(
                        tokenized,
                        max_tokens=256,
                        max_atoms=2048,
                    )
                elif receptor_crop_indices is not None:
                    tokenized = self.cropper.crop_by_receptor_token_indices(
                        tokenized,
                        receptor_crop_indices,
                        max_tokens=256,
                        max_atoms=2048,
                    )
                elif crop_indices is None:
                    tokenized = self.cropper.crop_sequence_window(
                        tokenized,
                        protein_start=self.affinity_sequence_window_start,
                        max_tokens=256,
                        max_atoms=2048,
                    )
                else:
                    tokenized = self.cropper.crop_by_token_indices(
                        tokenized,
                        crop_indices,
                    )
            else:
                crop_indices = self._load_affinity_crop_indices(record.id)
                if crop_indices is None:
                    tokenized = self.cropper.crop(
                        tokenized,
                        max_tokens=256,
                        max_atoms=2048,
                    )
                else:
                    tokenized = self.cropper.crop_by_token_indices(
                        tokenized,
                        crop_indices,
                    )
            resume_token_idx = torch.from_numpy(
                tokenized.tokens["token_idx"].copy()
            ).long()

        # Load conformers
        molecules = {}
        molecules.update(self.canonicals)
        molecules.update(input_data.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        uncached = mol_names - set(self._mol_cache.keys())
        if uncached:
            self._mol_cache.update(load_molecules(self.mol_dir, uncached))
        molecules.update({k: self._mol_cache[k] for k in mol_names})
        if self.affinity_ligand_ref_pos_mode != "etkdg":
            molecules = prepare_ligand_reference_conformers(
                molecules,
                tokenized,
                self.affinity_ligand_ref_pos_mode,
            )

        # Inference specific options
        options = record.inference_options
        if options is None:
            pocket_constraints, contact_constraints = None, None
        else:
            pocket_constraints, contact_constraints = (
                options.pocket_constraints,
                options.contact_constraints,
            )

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=pocket_constraints,
            inference_contact_constraints=contact_constraints,
            compute_constraint_features=True,
            override_method=self.override_method,
            compute_affinity=self.affinity,
            compute_distogram_target=not self.affinity_structure_free,
            use_structure_coordinates=not self.affinity_structure_free,
        )
        features = apply_ligand_reference_position_intervention(
            features,
            self.affinity_ligand_ref_pos_mode,
        )

        if self.affinity:
            features["resume_token_idx"] = resume_token_idx

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class Boltz2InferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz2 inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        num_workers: int,
        batch_size: int = 1,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        affinity_crop_indices_dir: Optional[Path] = None,
        affinity_receptor_crop_indices_dir: Optional[Path] = None,
        affinity_structure_free: bool = False,
        affinity_structure_free_single_sequence: bool = False,
        affinity_sequence_window_start: int = 0,
        affinity_ligand_ref_pos_mode: str = "etkdg",
        affinity_require_full_input: bool = False,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        num_workers : int
            The number of workers to use.
        batch_size : int
            The number of records to process in each prediction batch.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.
        extra_mols_dir : Optional[Path]
            The path to the extra molecules directory.
        override_method : Optional[str]
            The method to override.
        pin_memory : bool
            Whether to use pinned memory for DataLoader.

        """
        super().__init__()
        if affinity and batch_size != 1 and not affinity_structure_free:
            msg = "Batched affinity inference requires affinity_structure_free."
            raise ValueError(msg)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        self.affinity_crop_indices_dir = affinity_crop_indices_dir
        self.affinity_receptor_crop_indices_dir = affinity_receptor_crop_indices_dir
        self.affinity_structure_free = affinity_structure_free
        self.affinity_structure_free_single_sequence = (
            affinity_structure_free_single_sequence
        )
        self.affinity_sequence_window_start = affinity_sequence_window_start
        self.affinity_ligand_ref_pos_mode = affinity_ligand_ref_pos_mode
        self.affinity_require_full_input = affinity_require_full_input
        self.pin_memory = pin_memory

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            mol_dir=self.mol_dir,
            constraints_dir=self.constraints_dir,
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
            override_method=self.override_method,
            affinity=self.affinity,
            affinity_crop_indices_dir=self.affinity_crop_indices_dir,
            affinity_receptor_crop_indices_dir=(
                self.affinity_receptor_crop_indices_dir
            ),
            affinity_structure_free=self.affinity_structure_free,
            affinity_structure_free_single_sequence=(
                self.affinity_structure_free_single_sequence
            ),
            affinity_sequence_window_start=self.affinity_sequence_window_start,
            affinity_ligand_ref_pos_mode=self.affinity_ligand_ref_pos_mode,
            affinity_require_full_input=self.affinity_require_full_input,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch
