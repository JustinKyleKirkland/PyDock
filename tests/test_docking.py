import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pydock.docking import MoleculeDocker


def create_test_molecules():
    """Create simple test molecules for docking."""
    # Create a simple benzene molecule as receptor
    receptor_smiles = "c1ccccc1"
    receptor = Chem.MolFromSmiles(receptor_smiles)
    AllChem.EmbedMolecule(receptor)

    # Create a simple toluene molecule as ligand
    ligand_smiles = "Cc1ccccc1"
    ligand = Chem.MolFromSmiles(ligand_smiles)
    AllChem.EmbedMolecule(ligand)

    return receptor, ligand


def create_test_pdb_files(tmp_path):
    """Create temporary PDB files for testing."""
    receptor, ligand = create_test_molecules()

    receptor_path = os.path.join(tmp_path, "receptor.pdb")
    ligand_path = os.path.join(tmp_path, "ligand.pdb")

    Chem.PDBWriter(receptor_path).write(receptor)
    Chem.PDBWriter(ligand_path).write(ligand)

    return receptor_path, ligand_path


def test_molecule_docker_initialization():
    """Test MoleculeDocker initialization."""
    receptor, _ = create_test_molecules()
    docker = MoleculeDocker(receptor)
    assert docker.receptor is not None
    assert docker.grid_spacing == 1.0
    assert len(docker.grid_points) > 0


def test_grid_points_generation():
    """Test grid points generation."""
    receptor, _ = create_test_molecules()
    docker = MoleculeDocker(receptor)
    assert docker.grid_points.shape[1] == 3  # Should have x, y, z coordinates
    assert len(docker.grid_points) > 0


def test_dock_ligand():
    """Test ligand docking."""
    receptor, ligand = create_test_molecules()
    docker = MoleculeDocker(receptor)
    docked_pose, score = docker.dock_ligand(ligand, num_conformers=2)

    assert docked_pose is not None
    assert isinstance(score, float)
    assert score > 0


def test_invalid_receptor():
    """Test handling of invalid receptor."""
    with pytest.raises(Exception):
        MoleculeDocker(None)


def test_invalid_ligand():
    """Test handling of invalid ligand."""
    receptor, _ = create_test_molecules()
    docker = MoleculeDocker(receptor)
    with pytest.raises(Exception):
        docker.dock_ligand(None)


def test_pdb_file_handling(tmp_path):
    """Test handling of PDB files."""
    receptor_path, ligand_path = create_test_pdb_files(tmp_path)

    # Test reading PDB files
    docker = MoleculeDocker(receptor_path)
    assert docker.receptor is not None

    docked_pose, score = docker.dock_ligand(ligand_path)
    assert docked_pose is not None
    assert isinstance(score, float)
    assert score > 0


def test_invalid_pdb_file():
    """Test handling of invalid PDB files."""
    with pytest.raises(Exception):
        MoleculeDocker("nonexistent.pdb")

    receptor, _ = create_test_molecules()
    docker = MoleculeDocker(receptor)
    with pytest.raises(Exception):
        docker.dock_ligand("nonexistent.pdb")
