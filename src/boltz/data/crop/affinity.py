from dataclasses import replace
from typing import Optional

import numpy as np

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


class AffinityCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(
        self,
        neighborhood_size: int = 10,
        max_tokens_protein: int = 200,
    ) -> None:
        """Initialize the cropper.

        Parameters
        ----------
        neighborhood_size : int
            Modulates the type of cropping to be performed.
            Smaller neighborhoods result in more spatial
            cropping. Larger neighborhoods result in more
            continuous cropping.

        """
        self.neighborhood_size = neighborhood_size
        self.max_tokens_protein = max_tokens_protein

    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # compute minimum distance to ligand
        ligand_coords = valid_tokens[valid_tokens["affinity_mask"]]["center_coords"]
        dists = np.min(
            np.sum(
                (valid_tokens["center_coords"][:, None] - ligand_coords[None]) ** 2,
                axis=-1,
            )
            ** 0.5,
            axis=1,
        )

        indices = np.argsort(dists)

        # Select cropped indices
        cropped: set[int] = set()
        total_atoms = 0

        # protein tokens
        cropped_protein: set[int] = set()
        ligand_ids = set(
            valid_tokens[
                valid_tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]["token_idx"]
        )

        for idx in indices:
            # Get the token
            token = valid_tokens[idx]

            # Get all tokens from this chain
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= self.neighborhood_size:
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighborhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                min_idx = token["res_idx"] - self.neighborhood_size
                max_idx = token["res_idx"] + self.neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                # Start by adding just the query token
                new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                min_idx = max_idx = token["res_idx"]
                while new_tokens.size < self.neighborhood_size:
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    new_tokens = max_token_set
                    new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                    new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]

            # Compute new tokens and new atoms
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_tokens = token_data[list(new_indices)]
            new_atoms = np.sum(new_tokens["atom_num"])

            # Stop if we exceed the max number of tokens or atoms
            if (
                (len(new_indices) > (max_tokens - len(cropped)))
                or ((max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms))
                or (
                    len(cropped_protein | new_indices - ligand_ids)
                    > self.max_tokens_protein
                )
            ):
                break

            # Add new indices
            cropped.update(new_indices)
            total_atoms += new_atoms

            # Add protein indices
            cropped_protein.update(new_indices - ligand_ids)

        # Get the cropped tokens sorted by index
        token_data = token_data[sorted(cropped)]

        # Only keep bonds within the cropped tokens
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        # Return the cropped tokens
        return replace(data, tokens=token_data, bonds=token_bonds)

    def crop_sequence_prefix(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: Optional[int] = None,
    ) -> Tokenized:
        """Select binder tokens and a deterministic receptor sequence prefix.

        This crop deliberately ignores coordinates and resolved masks. It is a
        prior-free baseline for affinity inference without a generated complex
        structure, not a pocket-selection heuristic.
        """
        return self.crop_sequence_window(
            data,
            protein_start=0,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )

    def crop_full_screen(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: Optional[int] = None,
    ) -> Tokenized:
        """Select the full receptor and ligand or fail without truncation."""
        cropped = self.crop_sequence_prefix(
            data,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )
        if cropped.tokens.size != data.tokens.size:
            msg = (
                "Full-screen input was truncated during affinity token selection: "
                f"kept {cropped.tokens.size}/{data.tokens.size} tokens."
            )
            raise ValueError(msg)
        return cropped

    def crop_sequence_window(
        self,
        data: Tokenized,
        protein_start: int,
        max_tokens: int,
        max_atoms: Optional[int] = None,
    ) -> Tokenized:
        """Select binder tokens and a deterministic receptor sequence window."""
        if protein_start < 0:
            msg = "Sequence affinity window start must be non-negative."
            raise ValueError(msg)

        token_data = data.tokens
        binder_tokens = token_data[token_data["affinity_mask"]]
        if not binder_tokens.size:
            msg = "Sequence affinity crop found no binder tokens."
            raise ValueError(msg)

        binder_atoms = int(np.sum(binder_tokens["atom_num"]))
        if binder_tokens.size > max_tokens or (
            max_atoms is not None and binder_atoms > max_atoms
        ):
            msg = "Binder alone exceeds the sequence affinity crop limits."
            raise ValueError(msg)

        protein_tokens = token_data[
            token_data["mol_type"] == const.chain_type_ids["PROTEIN"]
        ]
        if protein_start >= protein_tokens.size:
            msg = (
                f"Sequence affinity window start {protein_start} is outside "
                f"the {protein_tokens.size}-token receptor."
            )
            raise ValueError(msg)
        protein_tokens = protein_tokens[
            protein_start : protein_start + self.max_tokens_protein
        ]
        selected_ids = {int(idx) for idx in binder_tokens["token_idx"]}
        total_atoms = binder_atoms
        protein_count = 0
        for token in protein_tokens:
            token_id = int(token["token_idx"])
            if token_id in selected_ids:
                continue
            token_atoms = int(token["atom_num"])
            if len(selected_ids) + 1 > max_tokens:
                break
            if max_atoms is not None and total_atoms + token_atoms > max_atoms:
                break
            selected_ids.add(token_id)
            total_atoms += token_atoms
            protein_count += 1

        if protein_count == 0:
            msg = "Sequence affinity crop found no protein receptor tokens."
            raise ValueError(msg)
        return self.crop_by_token_indices(
            data,
            np.asarray(sorted(selected_ids), dtype=np.int64),
        )

    @staticmethod
    def crop_by_token_indices(data: Tokenized, token_indices: np.ndarray) -> Tokenized:
        """Apply an explicit crop expressed in original token IDs."""
        requested = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        if requested.size == 0:
            msg = "Explicit affinity crop contains no token IDs."
            raise ValueError(msg)
        if np.unique(requested).size != requested.size:
            msg = "Explicit affinity crop contains duplicate token IDs."
            raise ValueError(msg)

        token_data = data.tokens
        available = {int(idx) for idx in token_data["token_idx"]}
        missing = [int(idx) for idx in requested if int(idx) not in available]
        if missing:
            msg = f"Explicit affinity crop references missing token IDs: {missing[:10]}"
            raise ValueError(msg)

        # Preserve canonical token order so feature and saved-embedding axes align.
        selected = np.isin(token_data["token_idx"], requested)
        token_data = token_data[selected]
        if not np.any(token_data["affinity_mask"]):
            msg = "Explicit affinity crop does not contain the binder."
            raise ValueError(msg)

        indices = token_data["token_idx"]
        token_bonds = data.bonds[np.isin(data.bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]
        return replace(data, tokens=token_data, bonds=token_bonds)
