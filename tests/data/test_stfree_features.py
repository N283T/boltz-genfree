"""Feature-generation tests for the structure-free inference route."""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Conformer

from boltz.data import const
from boltz.data.feature.featurizerv2 import process_atom_features
from boltz.data.types import (
    AtomV2,
    BondV2,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    StructureV2,
    TokenBondV2,
    Tokenized,
    TokenV2,
)


def _single_atom_ligand() -> tuple[Tokenized, dict[str, Chem.Mol]]:
    molecule = Chem.MolFromSmiles("C")
    assert molecule is not None
    molecule.GetAtomWithIdx(0).SetProp("name", "C1")
    molecule.AddConformer(Conformer(1))

    atoms = np.zeros(1, dtype=AtomV2)
    atoms[0]["name"] = "C1"
    atoms[0]["element"] = 6
    atoms[0]["is_present"] = True

    residues = np.zeros(1, dtype=Residue)
    residues[0]["name"] = "LIG"
    residues[0]["atom_num"] = 1
    residues[0]["is_present"] = True

    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = "B"
    chains[0]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
    chains[0]["atom_num"] = 1
    chains[0]["res_num"] = 1

    coords = np.zeros(1, dtype=Coords)
    ensemble = np.zeros(1, dtype=Ensemble)
    ensemble[0]["atom_num"] = 1
    structure = StructureV2(
        atoms=atoms,
        bonds=np.array([], dtype=BondV2),
        residues=residues,
        chains=chains,
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
        coords=coords,
        ensemble=ensemble,
    )

    tokens = np.zeros(1, dtype=TokenV2)
    tokens[0]["atom_num"] = 1
    tokens[0]["res_name"] = "LIG"
    tokens[0]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
    tokens[0]["resolved_mask"] = True
    tokens[0]["disto_mask"] = True
    tokens[0]["affinity_mask"] = True
    tokenized = Tokenized(
        tokens=tokens,
        bonds=np.array([], dtype=TokenBondV2),
        structure=structure,
        msa={},
    )
    return tokenized, {"LIG": molecule}


def test_structure_free_atom_features_omit_target_distogram() -> None:
    tokenized, molecules = _single_atom_ligand()

    features = process_atom_features(
        data=tokenized,
        random=np.random.default_rng(42),
        molecules=molecules,
        ensemble_features={"ensemble_ref_idxs": np.array([0])},
        compute_distogram_target=False,
        use_structure_coordinates=False,
    )

    assert "disto_target" not in features
    assert "disto_coords_ensemble" not in features


def test_regular_atom_features_keep_target_distogram() -> None:
    tokenized, molecules = _single_atom_ligand()

    features = process_atom_features(
        data=tokenized,
        random=np.random.default_rng(42),
        molecules=molecules,
        ensemble_features={"ensemble_ref_idxs": np.array([0])},
        compute_distogram_target=True,
    )

    assert features["disto_target"].shape == (1, 1, 1, 64)
    assert features["disto_coords_ensemble"].shape == (1, 1, 3)
