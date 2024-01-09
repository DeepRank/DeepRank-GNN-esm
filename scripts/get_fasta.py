import argparse
import os
from pdb2sql import pdb2sql

One2ThreeDict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
                 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
                 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                 'B': 'ASX', 'U': 'SEC', 'Z': 'GLX'}

Three2OneDict = {v: k for k, v in One2ThreeDict.items()}


def get_fasta(pdb_path, output_fasta_dir):
    sqldb = pdb2sql(pdb_path)
    
    for chain_id in ('A', 'B'):
        residues = sqldb.get_residues(chainID=chain_id)
        seq = ''.join(Three2OneDict[residue[1]] for residue in residues)
        
        case_id = os.path.basename(pdb_path).split('.')[0]
        output_file = os.path.join(output_fasta_dir, f'{case_id}.{chain_id}.fasta')

        with open(output_file, 'w') as f:
            f.write(f'>{case_id}.{chain_id}\n')
            f.write('\n'.join([seq[i:i+79] for i in range(0, len(seq), 79)]))


def combine_fasta_files(fasta_dir, output_fasta_name):
    os.system(f"sed -n 'p' {fasta_dir}/*.fasta > {output_fasta_name}")


def main(pdb_dir, output_fasta_name):
    fasta_dir = os.path.join(pdb_dir, 'fasta_files')
    os.makedirs(fasta_dir, exist_ok=True)

    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            get_fasta(pdb_path, output_fasta_dir=fasta_dir)

    combine_fasta_files(fasta_dir, output_fasta_name)
    print('Fasta files generated for all PDB files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_dir', help='Path to the directory containing PDB files')
    parser.add_argument('output_fasta_name', help='Name of the combined output FASTA file')
    args = parser.parse_args()

    main(args.pdb_dir, args.output_fasta_name)
