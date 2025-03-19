from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges


class MolecularScorer:
    def __init__(self):
        """Initialize the molecular scorer with force field parameters."""
        # Van der Waals parameters (Lennard-Jones) - AMBER-like parameters
        self.vdw_params = {
            "C": {"epsilon": 0.15, "sigma": 3.5, "well_depth": 0.15},
            "N": {"epsilon": 0.16, "sigma": 3.25, "well_depth": 0.16},
            "O": {"epsilon": 0.17, "sigma": 3.0, "well_depth": 0.17},
            "H": {"epsilon": 0.02, "sigma": 2.5, "well_depth": 0.02},
            "S": {"epsilon": 0.17, "sigma": 3.55, "well_depth": 0.17},
            "P": {"epsilon": 0.16, "sigma": 3.7, "well_depth": 0.16},
            "F": {"epsilon": 0.06, "sigma": 2.9, "well_depth": 0.06},
            "Cl": {"epsilon": 0.12, "sigma": 3.4, "well_depth": 0.12},
            "Br": {"epsilon": 0.12, "sigma": 3.5, "well_depth": 0.12},
            "I": {"epsilon": 0.12, "sigma": 3.7, "well_depth": 0.12},
        }

        # Electrostatic parameters
        self.coulomb_constant = 332.0  # kcal/mol * Å/e²
        self.dielectric_params = {
            "initial": 1.0,  # Dielectric constant at contact
            "final": 80.0,  # Dielectric constant at infinity
            "slope": 0.5,  # Rate of dielectric increase
            "cutoff": 8.0,  # Distance cutoff in Å
        }

        # Hydrogen bonding parameters with angle dependence
        self.hbond_params = {
            "O-H": {"energy": -5.0, "distance": 1.8, "angle": 120, "angle_tolerance": 30, "distance_tolerance": 0.5},
            "N-H": {"energy": -4.0, "distance": 1.9, "angle": 120, "angle_tolerance": 30, "distance_tolerance": 0.5},
            "S-H": {"energy": -3.0, "distance": 2.0, "angle": 120, "angle_tolerance": 30, "distance_tolerance": 0.5},
        }

        # Solvent parameters with improved surface tension model
        self.solvent_params = {
            "surface_tension": 0.005,  # kcal/mol/Å²
            "probe_radius": 1.4,  # Å
            "hydrophobic_factor": 0.5,  # Factor for hydrophobic surface area
            "polar_factor": 1.0,  # Factor for polar surface area
            "cutoff": 8.0,  # Å
        }

        # Entropy parameters with conformational entropy
        self.entropy_params = {
            "rotational_entropy": 0.5,  # kcal/mol/rotatable bond
            "translational_entropy": 1.0,  # kcal/mol
            "conformational_entropy": 0.3,  # kcal/mol/rotatable bond
            "ring_penalty": 0.2,  # kcal/mol/ring
            "torsion_penalty": 0.1,  # kcal/mol/torsion
        }

    def _calculate_distance_dependent_dielectric(self, distance: float) -> float:
        """Calculate distance-dependent dielectric constant."""
        if distance < 0.1:
            return self.dielectric_params["initial"]
        if distance > self.dielectric_params["cutoff"]:
            return self.dielectric_params["final"]

        # Smooth transition between initial and final dielectric
        return self.dielectric_params["initial"] + (
            self.dielectric_params["final"] - self.dielectric_params["initial"]
        ) * (1 - np.exp(-self.dielectric_params["slope"] * distance))

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi

    def calculate_vdw_energy(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate Van der Waals energy using improved Lennard-Jones potential."""
        energy = 0.0
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()

        for i in range(mol1.GetNumAtoms()):
            atom1 = mol1.GetAtomWithIdx(i)
            symbol1 = atom1.GetSymbol()
            pos1 = conf1.GetAtomPosition(i)

            for j in range(mol2.GetNumAtoms()):
                atom2 = mol2.GetAtomWithIdx(j)
                symbol2 = atom2.GetSymbol()
                pos2 = conf2.GetAtomPosition(j)

                # Get VDW parameters
                params1 = self.vdw_params.get(symbol1, {"epsilon": 0.1, "sigma": 3.0, "well_depth": 0.1})
                params2 = self.vdw_params.get(symbol2, {"epsilon": 0.1, "sigma": 3.0, "well_depth": 0.1})

                # Calculate distance
                dist = np.linalg.norm(np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z]))

                # Skip if distance is too large
                if dist > 8.0:  # Cutoff distance
                    continue

                # Lennard-Jones potential with improved parameters
                sigma = (params1["sigma"] + params2["sigma"]) / 2
                epsilon = np.sqrt(params1["epsilon"] * params2["epsilon"])
                well_depth = np.sqrt(params1["well_depth"] * params2["well_depth"])

                if dist < 0.1:  # Avoid division by zero
                    continue

                r6 = (sigma / dist) ** 6
                r12 = r6**2
                energy += 4 * epsilon * (r12 - r6) * well_depth

        return energy

    def calculate_electrostatic_energy(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate electrostatic energy using distance-dependent dielectric."""
        energy = 0.0

        # Calculate partial charges if not present
        if not mol1.HasProp("_GasteigerCharges"):
            rdPartialCharges.ComputeGasteigerCharges(mol1)
        if not mol2.HasProp("_GasteigerCharges"):
            rdPartialCharges.ComputeGasteigerCharges(mol2)

        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()

        for i in range(mol1.GetNumAtoms()):
            atom1 = mol1.GetAtomWithIdx(i)
            charge1 = float(atom1.GetProp("_GasteigerCharge"))
            pos1 = conf1.GetAtomPosition(i)

            for j in range(mol2.GetNumAtoms()):
                atom2 = mol2.GetAtomWithIdx(j)
                charge2 = float(atom2.GetProp("_GasteigerCharge"))
                pos2 = conf2.GetAtomPosition(j)

                # Calculate distance
                dist = np.linalg.norm(np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z]))

                if dist < 0.1:  # Avoid division by zero
                    continue

                # Skip if distance is too large
                if dist > self.dielectric_params["cutoff"]:
                    continue

                # Calculate distance-dependent dielectric
                dielectric = self._calculate_distance_dependent_dielectric(dist)

                # Coulomb's law with distance-dependent dielectric
                energy += self.coulomb_constant * charge1 * charge2 / (dist * dielectric)

        return energy

    def calculate_hbond_energy(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate hydrogen bonding energy with angle dependence."""
        energy = 0.0
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()

        for i in range(mol1.GetNumAtoms()):
            atom1 = mol1.GetAtomWithIdx(i)
            symbol1 = atom1.GetSymbol()
            pos1 = conf1.GetAtomPosition(i)

            # Check if atom1 can form H-bonds
            if symbol1 not in ["O", "N", "S"]:
                continue

            # Get bonded hydrogens
            bonded_hs = []
            for bond in atom1.GetBonds():
                if bond.GetOtherAtom(atom1).GetSymbol() == "H":
                    bonded_hs.append(bond.GetOtherAtom(atom1))

            for j in range(mol2.GetNumAtoms()):
                atom2 = mol2.GetAtomWithIdx(j)
                symbol2 = atom2.GetSymbol()
                pos2 = conf2.GetAtomPosition(j)

                # Check if atom2 can form H-bonds
                if symbol2 not in ["O", "N", "S"]:
                    continue

                # Calculate distance
                dist = np.linalg.norm(np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z]))

                # Get H-bond parameters
                bond_type = f"{symbol1}-H" if symbol1 in ["O", "N", "S"] else f"{symbol2}-H"
                params = self.hbond_params.get(bond_type, {"energy": -2.0, "distance": 2.0, "angle": 120})

                # Calculate H-bond energy based on distance and angle
                if abs(dist - params["distance"]) < params["distance_tolerance"]:
                    # Calculate angle between H-bond donor and acceptor
                    for h_atom in bonded_hs:
                        h_pos = conf1.GetAtomPosition(h_atom.GetIdx())
                        v1 = np.array([pos1.x - h_pos.x, pos1.y - h_pos.y, pos1.z - h_pos.z])
                        v2 = np.array([pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z])
                        angle = self._calculate_angle_between_vectors(v1, v2)

                        # Calculate angle-dependent energy
                        if abs(angle - params["angle"]) < params["angle_tolerance"]:
                            angle_factor = 1.0 - abs(angle - params["angle"]) / params["angle_tolerance"]
                            energy += params["energy"] * angle_factor

        return energy

    def calculate_solvent_energy(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate solvent effects using improved surface area model."""
        # Atomic radii in Angstroms
        atomic_radii = {
            "C": 1.7,
            "N": 1.55,
            "O": 1.52,
            "H": 1.2,
            "S": 1.8,
            "P": 1.8,
            "F": 1.47,
            "Cl": 1.75,
            "Br": 1.85,
            "I": 1.98,
        }

        energy = 0.0
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()

        # Calculate surface area for each molecule
        for i in range(mol1.GetNumAtoms()):
            atom1 = mol1.GetAtomWithIdx(i)
            symbol1 = atom1.GetSymbol()
            pos1 = conf1.GetAtomPosition(i)
            radius1 = atomic_radii.get(symbol1, 1.5)

            # Determine if atom is polar
            is_polar = symbol1 in ["O", "N", "S", "F", "Cl", "Br", "I"]

            for j in range(mol2.GetNumAtoms()):
                atom2 = mol2.GetAtomWithIdx(j)
                symbol2 = atom2.GetSymbol()
                pos2 = conf2.GetAtomPosition(j)
                radius2 = atomic_radii.get(symbol2, 1.5)

                # Skip if distance is too large
                dist = np.linalg.norm(np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z]))
                if dist > self.solvent_params["cutoff"]:
                    continue

                # If atoms are close enough to interact
                if dist < (radius1 + radius2 + 2.0):
                    # Calculate overlap
                    overlap = radius1 + radius2 - dist
                    if overlap > 0:
                        # Calculate buried surface area
                        buried_area = 4 * np.pi * (radius1**2 + radius2**2) * (overlap / (radius1 + radius2))

                        # Apply different factors for polar and non-polar surface area
                        if is_polar:
                            energy -= (
                                buried_area
                                * self.solvent_params["surface_tension"]
                                * self.solvent_params["polar_factor"]
                            )
                        else:
                            energy -= (
                                buried_area
                                * self.solvent_params["surface_tension"]
                                * self.solvent_params["hydrophobic_factor"]
                            )

        return energy

    def calculate_entropy_penalty(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate entropy penalty including conformational entropy."""
        # Count rotatable bonds
        rot_bonds1 = rdMolDescriptors.CalcNumRotatableBonds(mol1)
        rot_bonds2 = rdMolDescriptors.CalcNumRotatableBonds(mol2)

        # Count rings
        rings1 = len(mol1.GetRingInfo().AtomRings())
        rings2 = len(mol2.GetRingInfo().AtomRings())

        # Count torsions (using RDKit's proper method)
        torsions1 = len(
            [
                b
                for b in mol1.GetBonds()
                if b.GetBondType() == Chem.BondType.SINGLE
                and not b.GetBeginAtom().IsInRing()
                and not b.GetEndAtom().IsInRing()
            ]
        )
        torsions2 = len(
            [
                b
                for b in mol2.GetBonds()
                if b.GetBondType() == Chem.BondType.SINGLE
                and not b.GetBeginAtom().IsInRing()
                and not b.GetEndAtom().IsInRing()
            ]
        )

        # Calculate entropy penalties
        rotational_entropy = (rot_bonds1 + rot_bonds2) * self.entropy_params["rotational_entropy"]
        translational_entropy = self.entropy_params["translational_entropy"]
        conformational_entropy = (rot_bonds1 + rot_bonds2) * self.entropy_params["conformational_entropy"]
        ring_penalty = (rings1 + rings2) * self.entropy_params["ring_penalty"]
        torsion_penalty = (torsions1 + torsions2) * self.entropy_params["torsion_penalty"]

        return rotational_entropy + translational_entropy + conformational_entropy + ring_penalty + torsion_penalty

    def calculate_total_score(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Dict[str, float]:
        """Calculate total binding score including all components."""
        vdw_energy = self.calculate_vdw_energy(mol1, mol2)
        electrostatic_energy = self.calculate_electrostatic_energy(mol1, mol2)
        hbond_energy = self.calculate_hbond_energy(mol1, mol2)
        solvent_energy = self.calculate_solvent_energy(mol1, mol2)
        entropy_penalty = self.calculate_entropy_penalty(mol1, mol2)

        # Calculate total energy
        total_energy = vdw_energy + electrostatic_energy + hbond_energy + solvent_energy + entropy_penalty

        return {
            "total_energy": total_energy,
            "vdw_energy": vdw_energy,
            "electrostatic_energy": electrostatic_energy,
            "hbond_energy": hbond_energy,
            "solvent_energy": solvent_energy,
            "entropy_penalty": entropy_penalty,
        }
