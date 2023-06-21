import argparse
import os
import re
from pdb2sql import pdb2sql

One2ThreeDict = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
    'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'B': 'ASX', 'U': 'SEC', 'Z': 'GLX'
}
Three2OneDict = {v: k for k, v in One2ThreeDict.items()}


def get_fasta(pdb_dir, pdb, output_dir):
    pdb_path = os.path.join(pdb_dir, pdb)
    sqldb = pdb2sql(pdb_path)
    for chain_id in ('A', 'B'):
        # Get the unique residues
        residues = sqldb.get_residues(chainID=chain_id)
        # Get the one-letter residue code
        seq = ''
        count = 0
        for residue in residues:
            seq += Three2OneDict[residue[1]]
            count += 1
            if count == 79:
                seq += '\n'
                count = 0
        # Write the file
        case_id = re.split('_|\.', os.path.basename(pdb))[0]
        out_dir = os.path.join(output_dir, case_id)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f'{case_id}.{chain_id}.fasta')
        with open(fname, 'w') as f:
            f.write(f'>{case_id}.{chain_id}\n')
            f.write(seq)


def combine_fasta_files(pdb_dir, output_fasta_name):
    fasta_dir = os.path.join(pdb_dir, 'fasta_files')
    os.makedirs(fasta_dir, exist_ok=True)
    os.system(f"sed -n 'p' {fasta_dir}/*.fasta > {output_fasta_name}")


def main(pdb_dir, output_fasta_name):
    files = os.listdir(pdb_dir)
    pdbs = [f for f in files if f.endswith('.pdb')]

    # Generate FASTA files
    for pdb in pdbs:
        get_fasta(pdb_dir, pdb, output_dir='fasta_files')

    # Combine the FASTA files
    combine_fasta_files(pdb_dir, output_fasta_name)
    print('Fasta files generated for all PDB files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_dir', help='Path to the directory containing PDB files')
    parser.add_argument('output_fasta_name', help='Name of the combined output FASTA file')
    args = parser.parse_args()

    main(args.pdb_dir, args.output_fasta_name)
