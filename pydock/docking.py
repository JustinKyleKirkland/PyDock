import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .scoring import MolecularScorer
from .utils import prepare_molecule, read_pdb


class MoleculeDocker:
    def __init__(self, receptor: Union[Chem.Mol, str], grid_spacing: float = 2.0, add_hydrogens: bool = False):
        """
        Initialize the MoleculeDocker with a receptor molecule.

        Args:
            receptor (Union[Chem.Mol, str]): RDKit molecule object or path to PDB file
            grid_spacing (float): Spacing between grid points in Angstroms (default: 2.0)
            add_hydrogens (bool): Whether to add hydrogen atoms to molecules. Default is False to preserve original structure.
        """
        if isinstance(receptor, str):
            # If receptor is a PDB file path, read and prepare it
            mol = read_pdb(receptor)
            if mol is None:
                raise ValueError(f"Failed to read receptor PDB file: {receptor}")
            self.receptor = prepare_molecule(mol, add_hydrogens=add_hydrogens)
            if self.receptor is None:
                raise ValueError(f"Failed to prepare receptor molecule from: {receptor}")
        else:
            # If receptor is already a molecule, prepare it
            self.receptor = prepare_molecule(receptor, add_hydrogens=add_hydrogens)
            if self.receptor is None:
                raise ValueError("Failed to prepare receptor molecule")

        self.grid_spacing = grid_spacing
        self.add_hydrogens = add_hydrogens
        self.grid_points = self._generate_grid_points()
        self.scorer = MolecularScorer()

        # Add hydrogens if requested
        if add_hydrogens:
            self.receptor = Chem.AddHs(self.receptor)

        # Generate 3D conformer for receptor if needed
        if not self.receptor.GetNumConformers():
            self._generate_3d_conformer(self.receptor)

        # Initialize logging
        self.log_file = "docking_scores.log"
        self._initialize_log()

    def _initialize_log(self):
        """Initialize the log file with headers."""
        with open(self.log_file, "w") as f:
            f.write("Docking Scores Log\n")
            f.write("=================\n\n")
            f.write("Conformer\tTotal Energy\tVDW Energy\tElectrostatic\tH-Bond\tSolvent\tEntropy\n")
            f.write("-" * 80 + "\n")

    def _log_conformer_score(self, conformer_id: int, score_details: Dict[str, float]):
        """Log the score details for a conformer."""
        with open(self.log_file, "a") as f:
            f.write(f"{conformer_id}\t")
            f.write(f"{score_details['total_energy']:.2f}\t")
            f.write(f"{score_details['vdw_energy']:.2f}\t")
            f.write(f"{score_details['electrostatic_energy']:.2f}\t")
            f.write(f"{score_details['hbond_energy']:.2f}\t")
            f.write(f"{score_details['solvent_energy']:.2f}\t")
            f.write(f"{score_details['entropy_penalty']:.2f}\n")

    def _generate_grid_points(self) -> np.ndarray:
        """Generate grid points around the receptor."""
        # Get receptor bounds
        conf = self.receptor.GetConformer()
        bounds = np.array(
            [
                [conf.GetAtomPosition(i).x for i in range(self.receptor.GetNumAtoms())],
                [conf.GetAtomPosition(i).y for i in range(self.receptor.GetNumAtoms())],
                [conf.GetAtomPosition(i).z for i in range(self.receptor.GetNumAtoms())],
            ]
        )

        # Add padding (reduced from 5.0 to 3.0)
        padding = 3.0
        min_coords = bounds.min(axis=1) - padding
        max_coords = bounds.max(axis=1) + padding

        # Generate grid
        x = np.arange(min_coords[0], max_coords[0], self.grid_spacing)
        y = np.arange(min_coords[1], max_coords[1], self.grid_spacing)
        z = np.arange(min_coords[2], max_coords[2], self.grid_spacing)

        # Limit the number of grid points to prevent excessive computation
        max_points = 1000
        points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        if len(points) > max_points:
            # Randomly sample points if we have too many
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        logging.info(f"Generated {len(points)} grid points")
        return points

    def _generate_3d_conformer(self, mol: Chem.Mol) -> bool:
        """Generate a 3D conformer for a molecule using RDKit's ETKDG method."""
        try:
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            # Optimize the conformer
            AllChem.MMFFOptimizeMolecule(mol)
            return True
        except Exception as e:
            logging.error(f"Error generating 3D conformer: {e}")
            return False

    def _generate_3d_conformers(self, mol: Chem.Mol, num_conformers: int) -> bool:
        """Generate multiple 3D conformers for a molecule."""
        try:
            # Generate multiple conformers
            AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=42)
            # Optimize each conformer
            for i in range(mol.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            return True
        except Exception as e:
            logging.error(f"Error generating 3D conformers: {e}")
            return False

    def _translate_ligand(self, ligand: Chem.Mol, conf_id: int, point: np.ndarray) -> Chem.Mol:
        """Translate a ligand to a specific grid point.

        Args:
            ligand (Chem.Mol): The ligand molecule
            conf_id (int): The conformer ID to translate
            point (np.ndarray): The target grid point coordinates

        Returns:
            Chem.Mol: A new molecule with the translated conformer
        """
        # Create a copy of the ligand
        translated_ligand = Chem.Mol(ligand)

        # Get the conformer
        conf = translated_ligand.GetConformer(conf_id)

        # Translate each atom to the new position
        for i in range(translated_ligand.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + point[0], pos.y + point[1], pos.z + point[2]))

        return translated_ligand

    def dock_ligand(self, ligand: Chem.Mol, num_conformers: int = 100) -> Tuple[Optional[Chem.Mol], Dict[str, float]]:
        """Dock a ligand to the receptor using grid search and scoring."""
        # Add hydrogens if requested
        if self.add_hydrogens:
            ligand = Chem.AddHs(ligand)

        # Generate 3D conformers
        if not self._generate_3d_conformers(ligand, num_conformers):
            raise ValueError("Failed to generate 3D conformers for ligand")

        best_score = float("inf")
        best_conformer = None
        best_score_details = None

        total_iterations = len(self.grid_points) * ligand.GetNumConformers()
        current_iteration = 0

        # Try each conformer
        for conf_id in range(ligand.GetNumConformers()):
            # Try each grid point
            for point in self.grid_points:
                current_iteration += 1

                # Log progress every 100 iterations
                if current_iteration % 100 == 0:
                    progress = (current_iteration / total_iterations) * 100
                    logging.info(f"Docking progress: {progress:.1f}% ({current_iteration}/{total_iterations})")

                # Translate ligand to grid point
                translated_ligand = self._translate_ligand(ligand, conf_id, point)

                # Calculate score
                score_details = self.scorer.calculate_total_score(self.receptor, translated_ligand)

                # Log the score for this conformer
                self._log_conformer_score(conf_id, score_details)

                # Update best score if better
                if score_details["total_energy"] < best_score:
                    best_score = score_details["total_energy"]
                    best_conformer = translated_ligand
                    best_score_details = score_details
                    logging.info(f"New best score found: {best_score:.2f} kcal/mol")

                # Early termination if score is very good
                if score_details["total_energy"] < -10.0:
                    logging.info("Early termination: Found very good score")
                    return best_conformer, best_score_details

        logging.info("Docking completed. Best score: {:.2f} kcal/mol".format(best_score))
        return best_conformer, best_score_details

    def _calculate_score(self, ligand: Chem.Mol) -> float:
        """
        Calculate a simple distance-based score between ligand and receptor.
        Lower score is better.
        """
        score = 0.0
        ligand_conf = ligand.GetConformer()
        receptor_conf = self.receptor.GetConformer()

        for i in range(ligand.GetNumAtoms()):
            for j in range(self.receptor.GetNumAtoms()):
                dist = np.linalg.norm(
                    np.array(ligand_conf.GetAtomPosition(i)) - np.array(receptor_conf.GetAtomPosition(j))
                )
                score += 1.0 / (dist + 1.0)  # Add 1.0 to avoid division by zero

        return score
