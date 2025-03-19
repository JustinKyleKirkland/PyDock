import logging
import os

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from pydock.docking import MoleculeDocker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("docking.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")


def create_test_molecules():
    """Create test molecules for docking."""
    try:
        # Create a simple benzene molecule as receptor
        logger.info("Creating receptor molecule (benzene)")
        receptor_smiles = "c1ccccc1"
        receptor = Chem.MolFromSmiles(receptor_smiles)
        if receptor is None:
            raise ValueError("Failed to create receptor molecule from SMILES")

        # Add hydrogens and generate 3D coordinates
        receptor = Chem.AddHs(receptor)
        AllChem.EmbedMolecule(receptor, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(receptor)
        logger.info(f"Receptor created successfully with {receptor.GetNumAtoms()} atoms (including hydrogens)")

        # Create a simple toluene molecule as ligand
        logger.info("Creating ligand molecule (toluene)")
        ligand_smiles = "Cc1ccccc1"
        ligand = Chem.MolFromSmiles(ligand_smiles)
        if ligand is None:
            raise ValueError("Failed to create ligand molecule from SMILES")

        # Add hydrogens and generate 3D coordinates
        ligand = Chem.AddHs(ligand)
        AllChem.EmbedMolecule(ligand, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(ligand)
        logger.info(f"Ligand created successfully with {ligand.GetNumAtoms()} atoms (including hydrogens)")

        return receptor, ligand
    except Exception as e:
        logger.error(f"Error creating test molecules: {e}")
        raise


def save_molecule(mol: Chem.Mol, filename: str):
    """Save a molecule to a PDB file."""
    try:
        writer = Chem.PDBWriter(filename)
        writer.write(mol)
        writer.close()
        logger.info(f"Molecule saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving molecule to {filename}: {e}")
        raise


def main():
    """Main function to demonstrate molecular docking."""
    try:
        # Create output directory
        os.makedirs("output", exist_ok=True)

        # Create test molecules
        receptor, ligand = create_test_molecules()

        # Save initial molecules
        save_molecule(receptor, "output/receptor.pdb")
        save_molecule(ligand, "output/ligand.pdb")

        # Initialize docker with grid spacing and hydrogen addition
        docker = MoleculeDocker(receptor, grid_spacing=2.0, add_hydrogens=False)

        # Dock ligand with 100 conformers
        docked_pose, score_details = docker.dock_ligand(ligand, num_conformers=300)

        if docked_pose is not None:
            # Save the docked pose
            save_molecule(docked_pose, "output/docked_pose.pdb")

            # Log the best score details
            logger.info("Best docking pose found:")
            logger.info(f"Total Energy: {score_details['total_energy']:.2f} kcal/mol")
            logger.info(f"VDW Energy: {score_details['vdw_energy']:.2f} kcal/mol")
            logger.info(f"Electrostatic Energy: {score_details['electrostatic_energy']:.2f} kcal/mol")
            logger.info(f"H-Bond Energy: {score_details['hbond_energy']:.2f} kcal/mol")
            logger.info(f"Solvent Energy: {score_details['solvent_energy']:.2f} kcal/mol")
            logger.info(f"Entropy Penalty: {score_details['entropy_penalty']:.2f} kcal/mol")

            logger.info(f"\nDetailed scores for all conformers have been logged to: {docker.log_file}")
        else:
            logger.error("Failed to find a valid docking pose")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
