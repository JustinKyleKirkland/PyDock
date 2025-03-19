from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def read_pdb(pdb_file: str, remove_hydrogens: bool = True) -> Optional[Chem.Mol]:
    """
    Read a PDB file and return an RDKit molecule.

    Args:
        pdb_file (str): Path to the PDB file
        remove_hydrogens (bool): Whether to remove hydrogen atoms

    Returns:
        Optional[Chem.Mol]: RDKit molecule object or None if reading fails
    """
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=remove_hydrogens)
        if mol is None:
            raise ValueError(f"Failed to read PDB file: {pdb_file}")
        return mol
    except Exception as e:
        print(f"Error reading PDB file: {e}")
        return None


def prepare_molecule(mol: Chem.Mol, add_hydrogens: bool = False) -> Optional[Chem.Mol]:
    """
    Prepare a molecule for docking by optionally adding hydrogens and generating 3D coordinates.

    Args:
        mol (Chem.Mol): RDKit molecule to prepare
        add_hydrogens (bool): Whether to add hydrogen atoms. Default is False to preserve original structure.

    Returns:
        Optional[Chem.Mol]: Prepared molecule or None if preparation fails
    """
    try:
        # Create a copy to avoid modifying the original
        prepared_mol = Chem.Mol(mol)

        # Add hydrogens if requested
        if add_hydrogens:
            prepared_mol = Chem.AddHs(prepared_mol)

        # Generate 3D coordinates if the molecule doesn't have them
        if not prepared_mol.GetNumConformers():
            AllChem.EmbedMolecule(prepared_mol, randomSeed=42)
            # Optimize the structure
            AllChem.MMFFOptimizeMolecule(prepared_mol)

        return prepared_mol
    except Exception as e:
        print(f"Error preparing molecule: {e}")
        return None


def read_receptor_ligand(receptor_pdb: str, ligand_pdb: str) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    Read receptor and ligand from PDB files and prepare them for docking.

    Args:
        receptor_pdb (str): Path to receptor PDB file
        ligand_pdb (str): Path to ligand PDB file

    Returns:
        Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]: Tuple of (receptor, ligand) molecules
    """
    receptor = read_pdb(receptor_pdb)
    ligand = read_pdb(ligand_pdb)

    if receptor is None or ligand is None:
        return None, None

    # Prepare molecules
    receptor = prepare_molecule(receptor, add_hydrogens=True)
    ligand = prepare_molecule(ligand, add_hydrogens=True)

    return receptor, ligand
