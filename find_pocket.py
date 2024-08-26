from Bio import PDB
import numpy as np

def calculate_distance(atom1, atom2):
    """Calculate the Euclidean distance between two atoms."""
    diff_vector = atom1.coord - atom2.coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def find_residues_around_ligand(pdb_file, ligand_name, distance_cutoff):
    """Find amino acid residues around a specific ligand within a given distance cutoff."""
    # Load the PDB structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # Initialize lists to hold the results
    residues_within_cutoff = []

    # Iterate over all models in the structure (typically only one model)
    for model in structure:
        # Iterate over all chains in the model
        for chain in model:
            # Find the ligand in the chain
            ligand_residue = None
            for residue in chain:
                if residue.resname == ligand_name:
                    ligand_residue = residue
                    break

            if ligand_residue is None:
                continue

            # Iterate over all residues in the chain to find those within the distance cutoff
            for residue in chain:
                if residue == ligand_residue:
                    continue

                # Calculate the minimum distance between any atom in the ligand and the residue
                min_distance = min(calculate_distance(ligand_atom, residue_atom)
                                   for ligand_atom in ligand_residue
                                   for residue_atom in residue)

                if min_distance <= distance_cutoff:
                    residue_id = residue.id[1]
                    residues_within_cutoff.append((residue.resname, residue_id))

    return residues_within_cutoff

def main(pdb_file, ligand_name):
    # Find residues within 10Å and 5Å
    residues_within_10A = find_residues_around_ligand(pdb_file, ligand_name, 10.0)
    residues_within_5A = find_residues_around_ligand(pdb_file, ligand_name, 5.0)

    # Output the results
    print("Residues within 10Å:")
    for resname, resnum in residues_within_10A:
        print(f"{resname} {resnum}")

    print("\nResidues within 5Å:")
    for resname, resnum in residues_within_5A:
        print(f"{resname} {resnum}")

if __name__ == "__main__":
    # Example usage
    pdb_file = "example.pdb"  # Replace with your PDB file path
    ligand_name = "LIG"  # Replace with your ligand name
    main(pdb_file, ligand_name)
