import glob
import argparse
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import warnings
from Bio import BiopythonWarning

def three_to_one(code):
    """Convert three-letter amino acid code to one-letter code."""
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return aa_map.get(code, 'X')  # Return 'X' if the code is not found

def pdb_to_fasta(pdb_file_dir, chain_id1, chain_id2):
    """Convert a PDB file to a FASTA file."""

    sample_pdb = glob.glob(f'{pdb_file_dir}/*.pdb')[0]

    pdb_id = sample_pdb.split("/")[-1].split(".")[0]

    parser = PDBParser()
    structure = parser.get_structure("structure", sample_pdb)

    with open(f"{pdb_id}_merged.fasta", "w") as file:
        for chain_id in [chain_id1, chain_id2]:
            chain = structure[0][chain_id]
            sequence = ""

            # Get the sequence of the chain
            for residue in chain:
                if not is_aa(residue.get_resname()):
                    continue
                sequence += three_to_one(residue.get_resname())

            # Write the sequence to the file with chain ID
            file.write(f">{chain_id}\n")
            file.write(sequence + "\n")

def main():
    parser = argparse.ArgumentParser(description='Convert a PDB file to a FASTA file for specified chains.')
    parser.add_argument('pdb_file_path', type=str, help='Path to the PDB file directory')
    parser.add_argument('chain_id1', type=str, help='Chain ID for the first sequence')
    parser.add_argument('chain_id2', type=str, help='Chain ID for the second sequence')
    args = parser.parse_args()

    warnings.simplefilter('ignore', BiopythonWarning)
    pdb_to_fasta(args.pdb_file_path, args.chain_id1, args.chain_id2)

if __name__ == "__main__":
    main()
