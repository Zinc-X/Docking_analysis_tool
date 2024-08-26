import os
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, Superimposer, PDBIO
import numpy as np

def read_sdf_file(file_path):
    """
    Reads a molecular file and returns a molecule object.

    Parameters:
        file_path (str): Path to the molecular file. Supported formats include .mol, .mol2, .sdf, .smiles.

    Returns:
        Chem.Mol: A molecule object. Raises an error if the file cannot be read or if the format is unsupported.
    """
    
    # Ensure the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Extract the file extension
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == '.mol':
        mol = Chem.MolFromMolFile(file_path)
    elif file_extension == '.mol2':
        mol = Chem.MolFromMol2File(file_path)
    elif file_extension == '.sdf':
        # SDF files may contain multiple molecules; here we read only the first one
        supplier = Chem.SDMolSupplier(file_path)
        mol = next(iter(supplier), None)
    elif file_extension == '.smiles':
        with open(file_path, 'r') as f:
            smiles = f.read().strip()
        mol = Chem.MolFromSmiles(smiles)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    if mol is None:
        raise ValueError(f"Failed to read molecule from {file_path}")
    
    return mol

def read_pdb_file(pdb_path):
    """
    Reads a PDB file and returns a protein structure.

    Parameters:
        pdb_path (str): Path to the PDB file.

    Returns:
        Bio.PDB.Structure.Structure: A protein structure object.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    return structure

def combine_molecule_protein(mol, protein_structure):
    """
    Combines a small molecule with a protein structure.

    Parameters:
        mol (Chem.Mol): The small molecule.
        protein_structure (Bio.PDB.Structure.Structure): The protein structure.

    Returns:
        Bio.PDB.Structure.Structure: The combined structure of the molecule and protein.
    """
    # Convert the molecule to PDB format
    mol_pdb_block = Chem.MolToPDBBlock(mol)
    mol_structure = PDBParser(QUIET=True).get_structure('ligand', mol_pdb_block)
    
    # Combine the two structures
    combined_structure = protein_structure.copy()
    for chain in mol_structure.get_chains():
        combined_structure[0].add(chain)
    
    return combined_structure

def calculate_rmsd(mol1, mol2, mode="all_atoms"):
    """
    Calculates the RMSD between two molecules based on the specified mode.

    Parameters:
        mol1 (Chem.Mol): The first molecule.
        mol2 (Chem.Mol): The second molecule.
        mode (str): The mode for RMSD calculation. Can be "all_atoms", "heavy_atoms", or "alpha_c".

    Returns:
        float: The RMSD value.
    """
    if mode == "all_atoms":
        atoms1 = mol1.GetAtoms()
        atoms2 = mol2.GetAtoms()
    elif mode == "heavy_atoms":
        atoms1 = [atom for atom in mol1.GetAtoms() if atom.GetAtomicNum() > 1]
        atoms2 = [atom for atom in mol2.GetAtoms() if atom.GetAtomicNum() > 1]
    elif mode == "alpha_c":
        atoms1 = [atom for atom in mol1.GetAtoms() if atom.GetPDBResidueInfo().GetName().strip() == 'CA']
        atoms2 = [atom for atom in mol2.GetAtoms() if atom.GetPDBResidueInfo().GetName().strip() == 'CA']
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Get coordinates
    coords1 = np.array([atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in atoms1])
    coords2 = np.array([atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in atoms2])

    # Calculate RMSD
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

def superimpose_proteins(protein1, protein2):
    """
    Superimposes two protein structures.

    Parameters:
        protein1 (Bio.PDB.Structure.Structure): The first protein structure.
        protein2 (Bio.PDB.Structure.Structure): The second protein structure.

    Returns:
        float: The RMSD value after superimposition.
    """
    sup = Superimposer()
    sup.set_atoms(list(protein1.get_atoms()), list(protein2.get_atoms()))
    sup.apply(protein2.get_atoms())
    return sup.rms

def process_input_files(input1, input2, mode="all_atoms"):
    """
    Main processing function for calculating RMSD based on input files.

    Parameters:
        input1 (str): Path to the first input file (.sdf or .pdb).
        input2 (str): Path to the second input file (.pdb).
        mode (str): The mode for RMSD calculation. Can be "all_atoms", "heavy_atoms", or "alpha_c".

    Returns:
        tuple: A tuple containing the RMSD values and their mean if input1 is .sdf, or a single RMSD value if input1 is .pdb.
    """
    if isinstance(input1, str) and input1.endswith('.sdf'):
        molecules = read_sdf_file(input1)
        protein_structure = read_pdb_file(input2)
        
        rmsd_values = []
        for mol in molecules:
            combined_structure = combine_molecule_protein(mol, protein_structure)
            rmsd = calculate_rmsd(mol, combined_structure, mode)
            rmsd_values.append(rmsd)
        
        rmsd_mean = np.mean(rmsd_values)
        return rmsd_values, rmsd_mean
    
    elif isinstance(input1, str) and input1.endswith('.pdb'):
        protein_structure1 = read_pdb_file(input1)
        protein_structure2 = read_pdb_file(input2)
        
        rms = superimpose_proteins(protein_structure1, protein_structure2)
        rmsd = calculate_rmsd(protein_structure1, protein_structure2, mode)
        
        return rms, rmsd

# Example usage
if __name__ == "__main__":
    input1 = "input1.sdf"  # or "input1.pdb"
    input2 = "input2.pdb"
    mode = "all_atoms"  # Can be "all_atoms", "heavy_atoms", or "alpha_c"
    
    result = process_input_files(input1, input2, mode)
    print(f"RMSD result: {result}")
