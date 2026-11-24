"""Ligand atom order and names exactly as GenFree's parser assigns them (needs RDKit).

``boltz.data.parse.schema`` standardises an affinity ligand's SMILES, parses it, adds
hydrogens and names the heavy atoms ``<ELEMENT><n>`` with ``n`` counting each element in
RDKit canonical-rank order. The names are a model input (the atom encoder embeds them), and
the heavy-atom order is the ligand token order. The WebGPU featuriser has no RDKit, so this
tool writes both as JSON for ``cli.js --ligand-atoms``::

    uv run python webgpu/tools/ligand_atoms.py 'Oc1ccccc1' > ligand.json
"""

from __future__ import annotations

import argparse
import json

from rdkit import Chem

from boltz.data.parse.schema import standardize


def ligand_atoms(smiles: str) -> dict[str, object]:
    standardized = standardize(smiles)
    if standardized is None:
        raise ValueError(f"cannot standardise {smiles}")
    mol = Chem.MolFromSmiles(standardized)
    mol.UpdatePropertyCache(strict=True)
    mol = Chem.AddHs(mol)
    ranks = Chem.CanonicalRankAtoms(mol)
    counts: dict[str, int] = {}
    names = {}
    for idx in sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i]):
        symbol = mol.GetAtomWithIdx(idx).GetSymbol().upper()
        counts[symbol] = counts.get(symbol, 0) + 1
        names[idx] = f"{symbol}{counts[symbol]}"
    heavy = Chem.RemoveHs(mol, sanitize=False)
    return {
        "smiles": standardized,
        "atomNames": [names[i] for i in range(heavy.GetNumAtoms())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles")
    args = parser.parse_args()
    print(json.dumps(ligand_atoms(args.smiles)))


if __name__ == "__main__":
    main()
