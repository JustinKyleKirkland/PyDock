# PyDock

A lightweight molecular docking program in Python that uses RDKit for molecular operations and implements a simple grid-based docking algorithm.

## Features

- Simple grid-based molecular docking
- Support for multiple ligand conformers
- Basic distance-based scoring function
- Easy-to-use Python API
- Support for PDB file input
- Flexible hydrogen handling (preserve original structure or add hydrogens)
- Automatic 3D coordinate generation when needed

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pydock.git
cd pydock
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Here are examples of how to use PyDock:

### Using SMILES strings (for simple molecules):

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from pydock.docking import MoleculeDocker

# Create receptor molecule (example with benzene)
receptor_smiles = "c1ccccc1"
receptor = Chem.MolFromSmiles(receptor_smiles)
AllChem.EmbedMolecule(receptor)

# Create ligand molecule (example with toluene)
ligand_smiles = "Cc1ccccc1"
ligand = Chem.MolFromSmiles(ligand_smiles)
AllChem.EmbedMolecule(ligand)

# Initialize the docker (optionally add hydrogens)
docker = MoleculeDocker(receptor, add_hydrogens=False)  # Default is False to preserve original structure

# Perform docking
docked_pose, score = docker.dock_ligand(ligand, num_conformers=10)

# The docked_pose contains the best pose found, and score is the docking score
# (lower is better)
```

### Using PDB files:

```python
from pydock.docking import MoleculeDocker

# Initialize the docker with PDB files (optionally add hydrogens)
docker = MoleculeDocker("receptor.pdb", add_hydrogens=False)  # Default is False to preserve original structure

# Perform docking with a ligand PDB file
docked_pose, score = docker.dock_ligand("ligand.pdb", num_conformers=10)

# The docked_pose contains the best pose found, and score is the docking score
# (lower is better)
```

## Running Tests

To run the unit tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
