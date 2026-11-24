"""Build a SMILES corpus and the PyTorch-side expected ligand atoms for it.

Writes ``corpus.json``: a list of ``{smiles, source, bucket, expected}`` where
``expected`` is ``ligand_atoms.ligand_atoms(smiles)`` (or ``{"error": ...}``).
``check_canonical.mjs`` runs the pure-JS port over the same list and compares.

    .venv/bin/python webgpu/tools/ligand_atoms_corpus.py --out corpus.json --ccd 4000
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ligand_atoms import ligand_atoms

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]

ORGANIC = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Se"}

HAND = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "OC(=O)c1ccccc1OC(C)=O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "c1ccc2c(c1)cccc2",
    "C1=CC=C2C(=C1)C=CC=C2",
    "O=C(O)c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
    "Cc1ccc(cc1)S(=O)(=O)Nc1ncccn1",
    "c1ccc(cc1)-c1ccccc1",
    "C1CCCCC1",
    "C1CC2CCC1C2",
    "C12C3C4C1C5C2C3C45",
    "C1C2CC3CC1CC(C2)C3",
    "N#Cc1ccccc1",
    "CC(C)(C)c1ccc(O)cc1",
    "Nc1ncnc2[nH]cnc12",
    "c1cc[nH]c1",
    "c1ccoc1",
    "c1ccsc1",
    "O=C1CCCN1",
    "CCN(CC)CC",
    "Clc1ccc(Cl)cc1Br",
    "FC(F)(F)c1ccccc1",
    "OCC(O)CO",
    "CS(C)=O",
    "C=CC=C",
    "C#CC#C",
    "c1ccc2[nH]c3ccccc3c2c1",
    "O=C1c2ccccc2C(=O)c2ccccc12",
    "Cc1nc2ccccc2[nH]1",
    "CC1=CC(=O)C=CC1=O",
    "O=c1cc[nH]cc1",
    "Oc1ccncc1",
    "C1=CC=CC=C1",
    "c1ccc2ccccc2c1",
    "c1ccc2cc3ccccc3cc2c1",
    "C1=CC2=CC=CC2=C1",
    "c1cc2ccc1CC2",
    "O=S(=O)(N)c1ccc(N)cc1",
    "CC(C)NCC(O)COc1cccc2ccccc12",
    "Cc1cc(C)n(n1)C",
    "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC",
    "CCOC(=O)C1=C(C)NC(C)=C(C1c1cccc(c1)N(=O)=O)C(=O)OC",
    "C(C)(C)(C)OC(=O)N1CCCCC1",
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "N[C@@H](C)C(=O)O",
    "C/C=C/C",
    "C[N+](C)(C)C",
    "[Na+].[Cl-]",
    "CC(=O)[O-].[Na+]",
    "C[N+](=O)[O-]",
    "CC(=O)N(C)C",
    "[2H]C",
    "[nH]1cccc1",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "O=C1NC(=O)c2ccccc12",
    "c1csc2c1cccc2",
    "c1ccc2occc2c1",
    "C1COCCN1",
    "c1ccc(cc1)C1(CC1)N",
    "CCCCCCCCCCCCCCCC(=O)O",
    # drug-like, written in arbitrary (non-canonical) atom order
    "CC(=O)Nc1ccc(O)cc1",
    "CN1CCN(CC1)c1ccc(cc1)Nc1nccc(n1)-c1cccnc1",
    "Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)NC(=O)c1ccc(CN2CCN(C)CC2)cc1",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "CCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(O)=O",
    "O=C(Nc1ccc(cc1)C(F)(F)F)Nc1ccc(Oc2ccnc(C(=O)NC)c2)cc1",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(CC1)CCOCC(O)=O",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "OC(=O)c1ccccc1O",
    "Nc1ccc(cc1)S(=O)(=O)Nc1ccccn1",
    "C1CCC2(CC1)CCCC2",
    "c1ccc2c(c1)oc1ccccc12",
    "O=C1CCCCC1",
    "N#CC(C#N)=C1C=CC=CN1",
    "C=C1C=CC=CC=C1",
    "O=C1C=CC=CC=C1",
    "Cc1cc2c(cc1C)n(cn2)C1OC(CO)C(O)C1O",
    "CC(C)(C)c1cc(C(C)(C)C)c(O)c(c1)C",
    "[2H]C([2H])([2H])O",
    "C[13CH2]CC",
    "CC[18OH]",
    "[O-][N+](=O)c1ccccc1",
    "O=N(=O)c1ccccc1",
    "C[N+](C)(C)CC(=O)[O-]",
    "NCC(=O)[O-]",
    "[NH3+]CC(=O)[O-]",
    "CC(=O)[O-]",
    "c1cc[n+](C)cc1",
    "C[n+]1ccccc1.[Cl-]",
    "CS(=O)(=O)C",
    "O=S(C)c1ccccc1",
    "CC(=O)NC(=N)N",
    "OC(=N)C",
    "C(=O)(O)C(O)=O",
    "c1ccc2c(c1)C(=O)c1ccccc1C2=O",
    "Oc1ncccc1",
    "[H]C([H])([H])C([H])([H])O",
    "[H]c1ccccc1",
    "c1ccccc1[H]",
]


def bucket(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsable"
    parts = []
    if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
        parts.append("inorganic")
    if "." in Chem.MolToSmiles(mol):
        parts.append("multifrag")
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        parts.append("charged")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        parts.append("isotope")
    iso = Chem.MolToSmiles(mol)
    if any(c in iso for c in "@/\\"):
        parts.append("stereo")
    return "+".join(parts) if parts else "plain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ccd", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants", type=int, default=1, help="random writings per CCD molecule"
    )
    parser.add_argument("--mols", default=str(Path.home() / ".boltz" / "mols"))
    parser.add_argument("--screen", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    entries: list[tuple[str, str]] = [(s, "hand") for s in HAND]

    screen = (
        Path(args.screen)
        if args.screen
        else (Path(__file__).resolve().parents[2] / "examples" / "screen_ligands.csv")
    )
    if screen.exists():
        with screen.open() as handle:
            for row in csv.DictReader(handle):
                entries.append((row["smiles"], "screen"))

    files = sorted(Path(args.mols).glob("*.pkl"))
    rng.shuffle(files)
    for path in files[: args.ccd]:
        try:
            with path.open("rb") as handle:
                mol = pickle.load(handle)
            mol = Chem.RemoveHs(mol)
            smi = Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001
            continue
        if not smi or mol.GetNumAtoms() > 128 or mol.GetNumAtoms() < 2:
            continue
        entries.append((smi, "ccd"))
        # The input SMILES is arbitrary: also feed a randomly ordered writing.
        for _ in range(args.variants):
            try:
                entries.append((Chem.MolToSmiles(mol, doRandom=True), "ccd-random"))
            except Exception:  # noqa: BLE001
                pass
        # Kekule writing with an arbitrary atom order (exercises the JS Kekulisation).
        try:
            kekule = Chem.Mol(mol)
            Chem.Kekulize(kekule, clearAromaticFlags=True)
            entries.append(
                (
                    Chem.MolToSmiles(kekule, kekuleSmiles=True, doRandom=True),
                    "ccd-kekule",
                )
            )
        except Exception:  # noqa: BLE001
            pass

    out = []
    seen = set()
    for smi, source in entries:
        if smi in seen:
            continue
        seen.add(smi)
        try:
            expected = ligand_atoms(smi)
        except BaseException as exc:  # noqa: BLE001
            expected = {"error": f"{type(exc).__name__}: {exc}"}
        b = bucket(smi)
        plain_in = Chem.MolFromSmiles(smi)
        changed = False
        if plain_in is not None and "smiles" in expected:
            changed = Chem.MolToSmiles(plain_in) != expected["smiles"]
        out.append(
            {
                "smiles": smi,
                "source": source,
                "bucket": b,
                "changed": changed,
                "expected": expected,
            }
        )
    Path(args.out).write_text(json.dumps(out))
    counts: dict[str, int] = {}
    for item in out:
        counts[item["bucket"]] = counts.get(item["bucket"], 0) + 1
    print(len(out), "entries", json.dumps(counts, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
