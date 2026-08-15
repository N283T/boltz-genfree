import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data import const
from boltz.data.module.inferencev2 import (
    apply_ligand_reference_position_intervention,
    prepare_ligand_reference_conformers,
)


class _Tokenized:
    tokens = np.array(
        [
            (const.chain_type_ids["PROTEIN"], "ALA"),
            (const.chain_type_ids["NONPOLYMER"], "LIG"),
        ],
        dtype=[("mol_type", "i1"), ("res_name", "U5")],
    )


def _molecule_with_3d_conformer() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) >= 0
    return mol


def test_rdkit_2d_replaces_only_ligand_conformer() -> None:
    protein = _molecule_with_3d_conformer()
    ligand = _molecule_with_3d_conformer()
    original_ligand_z = np.array(
        [
            ligand.GetConformer().GetAtomPosition(i).z
            for i in range(ligand.GetNumAtoms())
        ]
    )

    prepared = prepare_ligand_reference_conformers(
        {"ALA": protein, "LIG": ligand},
        _Tokenized(),
        "rdkit_2d",
    )

    ligand_z = np.array(
        [
            prepared["LIG"].GetConformer().GetAtomPosition(i).z
            for i in range(prepared["LIG"].GetNumAtoms())
        ]
    )
    assert np.any(np.abs(original_ligand_z) > 1e-6)
    assert np.allclose(ligand_z, 0.0)
    assert prepared["LIG"] is not ligand
    assert prepared["ALA"] is protein


def test_default_mode_reuses_original_molecules() -> None:
    molecules = {
        "ALA": _molecule_with_3d_conformer(),
        "LIG": _molecule_with_3d_conformer(),
    }

    assert (
        prepare_ligand_reference_conformers(molecules, _Tokenized(), "etkdg")
        is molecules
    )


def test_skip_mode_reuses_original_molecules() -> None:
    molecules = {
        "ALA": _molecule_with_3d_conformer(),
        "LIG": _molecule_with_3d_conformer(),
    }

    assert (
        prepare_ligand_reference_conformers(molecules, _Tokenized(), "skip")
        is molecules
    )


def test_seeded_etkdg_is_deterministic_and_replaces_only_ligand() -> None:
    molecules = {
        "ALA": _molecule_with_3d_conformer(),
        "LIG": _molecule_with_3d_conformer(),
    }

    first = prepare_ligand_reference_conformers(
        molecules,
        _Tokenized(),
        "etkdg",
        etkdg_seed=17,
    )
    second = prepare_ligand_reference_conformers(
        molecules,
        _Tokenized(),
        "etkdg",
        etkdg_seed=17,
    )
    different_seed = prepare_ligand_reference_conformers(
        molecules,
        _Tokenized(),
        "etkdg",
        etkdg_seed=18,
    )

    assert first["ALA"] is molecules["ALA"]
    assert first["LIG"] is not molecules["LIG"]
    assert np.allclose(
        first["LIG"].GetConformer().GetPositions(),
        second["LIG"].GetConformer().GetPositions(),
    )
    assert not np.allclose(
        first["LIG"].GetConformer().GetPositions(),
        different_seed["LIG"].GetConformer().GetPositions(),
    )


def test_skip_is_applied_after_featurization_to_ligand_only() -> None:
    features = {
        "ref_pos": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ),
        "atom_to_token": torch.tensor(
            [[1, 0], [0, 1], [0, 1]],
            dtype=torch.bool,
        ),
        "affinity_token_mask": torch.tensor([0, 1], dtype=torch.bool),
    }

    intervened = apply_ligand_reference_position_intervention(features, "skip")

    assert torch.equal(intervened["ref_pos"][0], features["ref_pos"][0])
    assert torch.count_nonzero(intervened["ref_pos"][1:]) == 0
    assert torch.count_nonzero(features["ref_pos"]) > 0
